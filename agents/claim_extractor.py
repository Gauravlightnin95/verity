"""Claim extraction agent (owned by Member D).

Purpose: one LLM call (structured output) decomposing an article into
atomic, checkable claims with entities/location/date, and classifying
genre (report/opinion/satire) - opinion/satire get checkability="low".
Prompt lives in core.prompts.CLAIM_EXTRACTION_PROMPT.

Every sentence the LLM returns is then scored by a local check-worthiness
classifier and only the check-worthy factual ones are passed on:

    LLM structured output -> N raw sentences
            |
            v
    check-worthiness classifier (local, one call per sentence)
            |
            +--> claim.csv : claim_id, claim, label   for ALL N
            |
            v
    keep label == "Check-worthy Factual"  ->  list[Claim]

Input: article (ArticleContent). Output: list[Claim] (core.schemas) - the
same element type as before the filter existed, only shorter.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from core.prompts import CLAIM_EXTRACTION_PROMPT
from core.schemas import Claim
from utils.logging_conf import get_logger

logger = get_logger(__name__)

# A ceiling, not a target: the check-worthiness filter below is what
# actually limits how many claims reach retrieval/stance, so this only has
# to stop a runaway response.
MAX_CLAIMS_PER_ARTICLE = 100

# ClaimBuster-trained, 3 classes. The label strings below are specific to
# this model - swap the two together or the filter silently drops
# everything (see CLAIM_FILTER_README.md section 2 for the probe to run
# before adopting a replacement).
_CLAIM_DETECTOR_MODEL = "Nithiwat/mdeberta-v3-base_claimbuster"

_LABEL_CHECKWORTHY = "Check-worthy Factual"  # verifiable and worth checking
_LABEL_UNIMPORTANT = "Unimportant Factual"  # true but trivial
_LABEL_NON_FACTUAL = "Non-factual"  # opinion, prediction, question

# Not a model label: what _label_claims records when the classifier could
# not be loaded or a call failed. Distinguishing it from a real label is
# what lets extract_claims tell "nothing was check-worthy" apart from
# "nothing was classified at all".
_LABEL_UNAVAILABLE = "classifier-unavailable"

# The whole widening knob: adding _LABEL_UNIMPORTANT.casefold() here also
# verifies trivially-true statements.
_CLAIM_LABELS = frozenset({_LABEL_CHECKWORTHY.casefold()})

# Per-run debug artefact, rewritten from scratch each time (no history by
# design) and never fatal if it can't be written.
_CLAIMS_CSV_PATH = Path("claim.csv").resolve()
_CLAIMS_CSV_COLUMNS = ("claim_id", "claim", "label")

# text -> check-worthiness label. The default is the lazily loaded HF
# pipeline; tests inject their own to stay offline.
ClaimClassifier = Callable[[str], str]


class ClaimExtractionError(Exception):
    """Raised when claim extraction fails after all retries are exhausted."""


class ArticleLike(Protocol):
    """Minimal structural contract for an article this module can process -
    satisfied by core.schemas.ArticleContent without importing it directly."""

    title: Optional[str]
    body: str
    author: Optional[str]
    domain: Optional[str]


class _ClaimExtractionResponse(BaseModel):
    """Wrapper model for structured output - a single top-level object is
    more reliable for tool-calling than a bare list."""

    claims: list[Claim] = Field(default_factory=list, max_length=MAX_CLAIMS_PER_ARTICLE)


_ARTICLE_BLOCK_TEMPLATE = """\
Article title: {title}
Article author: {author}
Article source domain: {domain}

Article body:
\"\"\"
{body}
\"\"\"\
"""


class _CheckWorthinessClassifier:
    """The default classifier: a Hugging Face text-classification pipeline
    over _CLAIM_DETECTOR_MODEL, loaded on first use.

    transformers/torch are core dependencies, but the import still happens
    inside the call rather than at module scope: importing torch costs
    seconds, and a partial or broken install must not take down this
    module (and therefore agents/graph.py) at import time. A failed load
    is remembered so the download is not retried once per claim.
    """

    def __init__(self, model_name: str = _CLAIM_DETECTOR_MODEL) -> None:
        self._model_name = model_name
        self._pipeline = None
        self._load_failed = False

    def __call__(self, text: str) -> str:
        pipeline_fn = self._ensure_loaded()
        if pipeline_fn is None:
            return _LABEL_UNAVAILABLE

        try:
            predictions = pipeline_fn(text, truncation=True)
        except Exception as exc:  # a per-sentence failure, not a dead model
            logger.warning("check-worthiness classification failed: %s", exc)
            return _LABEL_UNAVAILABLE

        if isinstance(predictions, list):
            predictions = predictions[0] if predictions else None
        if not isinstance(predictions, dict) or "label" not in predictions:
            logger.warning("check-worthiness classifier returned no label: %r", predictions)
            return _LABEL_UNAVAILABLE
        return str(predictions["label"])

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._load_failed:
            return None

        try:
            # Lazy on purpose - see the class docstring.
            from transformers import pipeline

            logger.debug("loading check-worthiness classifier %s", self._model_name)
            self._pipeline = pipeline("text-classification", model=self._model_name)
        except Exception as exc:
            logger.warning(
                "could not load check-worthiness classifier %s: %s", self._model_name, exc
            )
            self._load_failed = True
            return None
        return self._pipeline


class ClaimExtractor:
    """Extracts check-worthy factual claims from an article using an LLM.

    Parameters
    ----------
    llm: A LangChain chat model. Defaults to ChatGroq from app.config
        settings. Injectable for testing or to swap models.
    max_claims: Hard cap on claims the LLM step may return, before
        check-worthiness filtering.
    max_retries: Attempts before raising ClaimExtractionError.
    retry_backoff_seconds: Base delay between retries (exponential, capped
        at 30s).
    claim_classifier: text -> check-worthiness label. Defaults to the
        lazily loaded local classifier; inject a callable to keep tests
        offline or to swap detectors.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        max_claims: int = MAX_CLAIMS_PER_ARTICLE,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        claim_classifier: Optional[ClaimClassifier] = None,
    ) -> None:
        self._max_claims = max_claims
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._llm: BaseChatModel = llm or self._build_default_llm()
        self._structured_llm = self._llm.with_structured_output(_ClaimExtractionResponse)
        self._claim_classifier: ClaimClassifier = claim_classifier or _CheckWorthinessClassifier()

    @staticmethod
    def _build_default_llm() -> BaseChatModel:
        if not settings.groq_api_key:
            raise ClaimExtractionError(
                "GROQ_API_KEY is not set. Configure it in your .env file "
                "before running claim extraction."
            )
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
        )

    def extract_claims(self, article: ArticleLike) -> list[Claim]:
        self._validate_article(article)

        messages = self._build_messages(article)
        response = self._invoke_with_retry(messages)
        claims = self._enforce_claim_limit(response.claims)

        # Classify once, record everything, then filter: the CSV is written
        # before the filter so dropped sentences stay auditable.
        labeled = self._label_claims(claims)
        self._write_claims_csv(labeled)

        # A strict filter over unlabeled output drops every claim, which is
        # indistinguishable from a clean run that found nothing - so say so
        # instead of returning [].
        if labeled and all(label == _LABEL_UNAVAILABLE for _, label in labeled):
            raise ClaimExtractionError(
                "check-worthiness classifier could not label any claim, so none "
                "can be passed on (transformers/torch are core dependencies - "
                "repair the environment with: uv sync)"
            )

        kept = [claim for claim, label in labeled if self._is_claim(label)]

        logger.info(
            "claim extraction succeeded: %d extracted, %d check-worthy for %s",
            len(claims),
            len(kept),
            article.domain,
        )
        return kept

    def _validate_article(self, article: ArticleLike) -> None:
        if not article.body or not article.body.strip():
            raise ClaimExtractionError("cannot extract claims from an article with an empty body")

    def _build_messages(self, article: ArticleLike) -> list[HumanMessage]:
        article_block = _ARTICLE_BLOCK_TEMPLATE.format(
            title=article.title or "(no title)",
            author=article.author or "(unknown author)",
            domain=article.domain or "(unknown domain)",
            body=article.body,
        )
        prompt = CLAIM_EXTRACTION_PROMPT.format(max_claims=self._max_claims, article_text=article_block)
        return [HumanMessage(content=prompt)]

    def _invoke_with_retry(self, messages: list[HumanMessage]) -> _ClaimExtractionResponse:
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._structured_llm.invoke(messages)
                if isinstance(result, _ClaimExtractionResponse):
                    return result
                return _ClaimExtractionResponse.model_validate(result)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning("claim extraction returned invalid output (attempt %d): %s", attempt, exc)
            except Exception as exc:  # broad by design, retried
                last_error = exc
                logger.warning("claim extraction call failed (attempt %d): %s", attempt, exc)

            if attempt < self._max_retries:
                backoff = min(self._retry_backoff_seconds * (2 ** (attempt - 1)), 30.0)
                time.sleep(backoff)

        raise ClaimExtractionError(
            f"claim extraction failed after {self._max_retries} attempts: {last_error}"
        ) from last_error

    def _enforce_claim_limit(self, claims: list[Claim]) -> list[Claim]:
        if len(claims) > self._max_claims:
            logger.warning("model returned %d claims, truncating to %d", len(claims), self._max_claims)
            return claims[: self._max_claims]
        return claims

    def _label_claims(self, claims: list[Claim]) -> list[tuple[Claim, str]]:
        """One classifier call per claim, feeding both the CSV and the filter."""
        labeled: list[tuple[Claim, str]] = []
        for claim in claims:
            try:
                label = self._claim_classifier(claim.text)
            except Exception as exc:  # an injected classifier may raise anything
                logger.warning("could not classify claim %s: %s", claim.claim_id, exc)
                label = _LABEL_UNAVAILABLE
            labeled.append((claim, label or _LABEL_UNAVAILABLE))
        return labeled

    @staticmethod
    def _is_claim(label: str) -> bool:
        """Strict: only a check-worthy factual label passes."""
        return label.strip().casefold() in _CLAIM_LABELS

    @staticmethod
    def _write_claims_csv(labeled: list[tuple[Claim, str]]) -> None:
        """Snapshot every classified sentence, kept or dropped. Opened "w":
        a per-run snapshot, no history by design. A debug artefact, so a
        failure here is logged and never fatal."""
        try:
            with open(_CLAIMS_CSV_PATH, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(_CLAIMS_CSV_COLUMNS)
                for claim, label in labeled:
                    writer.writerow([claim.claim_id, claim.text, label])
        except Exception as exc:
            logger.warning("could not write %s: %s", _CLAIMS_CSV_PATH, exc)


_default_extractor: Optional[ClaimExtractor] = None


def extract_claims(article: ArticleLike) -> list[Claim]:
    """Module-level entrypoint agents/graph.py calls: extract_claims(article) -> list[Claim]."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = ClaimExtractor()
    return _default_extractor.extract_claims(article)
