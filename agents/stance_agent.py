"""Stance detection agent (owned by Member D).

Purpose: label each evidence snippet SUPPORTS/REFUTES/NEUTRAL toward its
claim, with a one-line rationale, judging only from the snippets (no
outside knowledge). Prompts live in core.prompts.

Two entrypoints:
- stance(claim, evidence): one claim, one LLM call (kept for direct use/tests).
- stance_batch(pairs): ALL claims judged in ONE LLM call - this is what
  agents/graph.py uses. Judging N claims per verify used to mean N separate
  LLM calls (the main driver of API-quota burn); batching collapses that to
  a single call regardless of claim count.
"""

from __future__ import annotations

import time
from typing import Optional, Protocol, runtime_checkable

from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from core.prompts import STANCE_BATCH_PROMPT, STANCE_PROMPT
from core.schemas import StanceResult
from utils.logging_conf import get_logger

logger = get_logger(__name__)

MAX_EVIDENCE_SNIPPETS_PER_CLAIM = 8


class StanceDetectionError(Exception):
    """Raised when stance detection fails after all retries are exhausted."""


@runtime_checkable
class ClaimLike(Protocol):
    claim_id: str
    text: str


@runtime_checkable
class EvidenceLike(Protocol):
    evidence_id: str
    snippet: str
    source_name: Optional[str]


class _StanceDetectionResponse(BaseModel):
    """Wrapper model for the single-claim structured output - a single
    top-level object is more reliable for tool-calling than a bare list."""

    results: list[StanceResult] = Field(default_factory=list)


class _StanceItem(BaseModel):
    """One batched result. `ref` is a short label (E1, E2, ...) the model
    echoes back - far more reliable than asking it to reproduce the full
    claim_id + evidence_id strings verbatim (Llama drops/rewrites those)."""

    ref: str
    stance: str  # SUPPORTS | REFUTES | NEUTRAL
    rationale: str = ""


class _StanceBatchResponse(BaseModel):
    results: list[_StanceItem] = Field(default_factory=list)


def _format_snippet_block(evidence_items: list[EvidenceLike]) -> str:
    lines = []
    for i, item in enumerate(evidence_items, start=1):
        source = item.source_name or "unknown source"
        lines.append(f'{i}. (evidence_id={item.evidence_id}, source="{source}"): "{item.snippet}"')
    return "\n".join(lines)


class StanceAgent:
    """Judges the stance of evidence snippets toward claims using an LLM.

    Parameters
    ----------
    llm: A LangChain chat model. Defaults to ChatGroq from app.config
        settings. Injectable for testing or to swap models.
    max_snippets_per_claim: Hard cap on snippets sent to the model per claim.
    max_retries: Attempts before raising StanceDetectionError.
    retry_backoff_seconds: Base delay between retries (exponential, capped
        at 8s).
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        max_snippets_per_claim: int = MAX_EVIDENCE_SNIPPETS_PER_CLAIM,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._max_snippets_per_claim = max_snippets_per_claim
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._llm: BaseChatModel = llm or self._build_default_llm()
        self._structured_llm = self._llm.with_structured_output(_StanceDetectionResponse)
        self._structured_batch_llm = self._llm.with_structured_output(_StanceBatchResponse)

    @staticmethod
    def _build_default_llm() -> BaseChatModel:
        if not settings.groq_api_key:
            raise StanceDetectionError(
                "GROQ_API_KEY is not set. Configure it in your .env file "
                "before running stance detection."
            )
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
        )

    # -- single-claim path (one LLM call per claim) --------------------------

    def stance(self, claim: ClaimLike, evidence: list[EvidenceLike]) -> list[StanceResult]:
        self._validate_claim(claim)

        if not evidence:
            logger.info("no evidence for claim %s - skipping stance call", claim.claim_id)
            return []

        truncated_evidence = self._enforce_snippet_limit(evidence)
        messages = [HumanMessage(content=STANCE_PROMPT.format(
            claim_text=claim.text, numbered_snippets=_format_snippet_block(truncated_evidence)
        ))]
        response = self._invoke_with_retry(messages, self._structured_llm, _StanceDetectionResponse)
        results = self._align_results(response.results, [(claim, truncated_evidence)])

        logger.info("stance detection succeeded: %d results for claim %s", len(results), claim.claim_id)
        return results

    # -- batched path (ONE LLM call for all claims) --------------------------

    def stance_batch(
        self, claim_evidence_pairs: list[tuple[ClaimLike, list[EvidenceLike]]]
    ) -> list[StanceResult]:
        """Judge every (claim, its-evidence) pair in a single LLM call.
        Pairs with no evidence or empty claim text are skipped (no call is
        made if nothing is judgeable)."""
        active = [
            (claim, self._enforce_snippet_limit(evidence))
            for claim, evidence in claim_evidence_pairs
            if evidence and claim.text and claim.text.strip()
        ]
        if not active:
            logger.info("no judgeable claim/evidence pairs - skipping stance call")
            return []

        # Label every evidence item with a short ref (E1, E2, ...) that the
        # model can echo reliably, and remember which real (claim, evidence)
        # each ref maps back to.
        ref_map: dict[str, tuple[ClaimLike, EvidenceLike]] = {}
        blocks: list[str] = []
        n = 0
        for claim, evidence in active:
            lines = []
            for item in evidence:
                n += 1
                ref = f"E{n}"
                ref_map[ref] = (claim, item)
                source = item.source_name or "unknown source"
                lines.append(f'  {ref} (source="{source}"): "{item.snippet}"')
            blocks.append(f'Claim {claim.claim_id}: "{claim.text}"\n' + "\n".join(lines))

        messages = [HumanMessage(content=STANCE_BATCH_PROMPT.format(claim_blocks="\n\n".join(blocks)))]
        response = self._invoke_with_retry(messages, self._structured_batch_llm, _StanceBatchResponse)

        stance_by_ref = {item.ref.strip().upper(): item for item in response.results}
        results: list[StanceResult] = []
        for ref, (claim, item) in ref_map.items():
            judged = stance_by_ref.get(ref)
            stance = judged.stance.strip().upper() if judged else "NEUTRAL"
            if stance not in {"SUPPORTS", "REFUTES", "NEUTRAL"}:
                stance = "NEUTRAL"
            results.append(StanceResult(
                claim_id=claim.claim_id,
                evidence_id=item.evidence_id,
                stance=stance,
                rationale=(judged.rationale if judged else "no stance returned by model; defaulted to NEUTRAL"),
            ))

        logger.info("batched stance detection succeeded: %d results across %d claims", len(results), len(active))
        return results

    # -- internals -----------------------------------------------------------

    def _validate_claim(self, claim: ClaimLike) -> None:
        if not claim.text or not claim.text.strip():
            raise StanceDetectionError("cannot judge stance for a claim with empty text")

    def _enforce_snippet_limit(self, evidence: list[EvidenceLike]) -> list[EvidenceLike]:
        if len(evidence) > self._max_snippets_per_claim:
            logger.warning(
                "evidence list (%d) exceeds per-claim cap (%d), truncating",
                len(evidence), self._max_snippets_per_claim,
            )
            return evidence[: self._max_snippets_per_claim]
        return evidence

    def _invoke_with_retry(self, messages: list[HumanMessage], chain, response_type):
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                result = chain.invoke(messages)
                if isinstance(result, response_type):
                    return result
                return response_type.model_validate(result)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning("stance detection returned invalid output (attempt %d): %s", attempt, exc)
            except Exception as exc:  # broad by design, retried
                last_error = exc
                logger.warning("stance detection call failed (attempt %d): %s", attempt, exc)

            if attempt < self._max_retries:
                time.sleep(min(self._retry_backoff_seconds * (2 ** (attempt - 1)), 8.0))

        raise StanceDetectionError(
            f"stance detection failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    def _align_results(
        self,
        results: list[StanceResult],
        pairs: list[tuple[ClaimLike, list[EvidenceLike]]],
    ) -> list[StanceResult]:
        """Rebuilds by (claim_id, evidence_id) so downstream code can always
        rely on exactly one result per evidence item across all claims,
        defaulting to NEUTRAL for anything the model dropped."""
        by_key = {(r.claim_id, r.evidence_id): r for r in results}
        aligned: list[StanceResult] = []

        for claim, evidence in pairs:
            for item in evidence:
                result = by_key.get((claim.claim_id, item.evidence_id))
                if result is None:
                    logger.warning(
                        "model did not return a stance for evidence %s (claim %s) - defaulting to NEUTRAL",
                        item.evidence_id, claim.claim_id,
                    )
                    result = StanceResult(
                        claim_id=claim.claim_id,
                        evidence_id=item.evidence_id,
                        stance="NEUTRAL",
                        rationale="no stance returned by model; defaulted to NEUTRAL",
                    )
                aligned.append(result)

        return aligned


_default_agent: Optional[StanceAgent] = None


def _agent() -> StanceAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = StanceAgent()
    return _default_agent


def stance(claim: ClaimLike, evidence: list[EvidenceLike]) -> list[StanceResult]:
    """Single-claim entrypoint: stance(claim, evidence) -> list[StanceResult]."""
    return _agent().stance(claim, evidence)


def stance_batch(
    claim_evidence_pairs: list[tuple[ClaimLike, list[EvidenceLike]]]
) -> list[StanceResult]:
    """Batched entrypoint agents/graph.py calls: one LLM call for all claims."""
    return _agent().stance_batch(claim_evidence_pairs)
