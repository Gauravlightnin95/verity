"""Judge agent: asks the LLM to write the human-readable verdict, hard-
constrained to never contradict core.fusion's deterministic draft.

The LLM's structured output is deliberately SLIM (_JudgeNotes: label +
caveats only, a few hundred tokens). It never echoes back evidence
citations, signals, per-claim rollups, or confidence - those are copied
from the fusion draft verbatim. This matters twice over:
- correctness: asking Llama to regurgitate ~20 full citations through a
  tool call made it truncate mid-JSON and fail with tool_use_failed on
  every evidence-rich article;
- integrity: the LLM is structurally incapable of inventing/dropping
  evidence or inflating a signal weight, rather than merely told not to.

If the LLM call fails, or it tries to change the label (other than a
more-cautious downgrade to UNVERIFIABLE), this falls back to the fusion
verdict unchanged and logs why - so "the pipeline works with zero API
keys" stays structurally guaranteed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings
from core.prompts import JUDGE_PROMPT
from core.schemas import CheckLog, EvidenceItem, StanceResult, Verdict, VerdictLabel


class _JudgeNotes(BaseModel):
    """The only things the judge LLM may produce: its label confirmation
    (or UNVERIFIABLE downgrade) and the polished human-readable caveats."""

    label: VerdictLabel
    caveats: list[str] = Field(default_factory=list)


def _build_chain(model_name: str):
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model_name, api_key=settings.groq_api_key, temperature=0
    ).with_structured_output(_JudgeNotes)


def _fallback(fusion_verdict: Verdict, reason: str, status: str = "failed") -> Verdict:
    result = fusion_verdict.model_copy(deep=True)
    result.checks_performed = [
        *result.checks_performed,
        CheckLog(agent_name="judge", status=status, note=reason),
    ]
    return result


def judge(
    fusion_verdict: Verdict,
    evidence: list[EvidenceItem],
    stance_results: list[StanceResult],
    model: str | None = None,
) -> Verdict:
    # No evidence -> fusion already forced UNVERIFIABLE and the judge could
    # only cite nothing, so skip the LLM call entirely. Saves a call on
    # every no-coverage run (common when search returns nothing).
    if not evidence:
        return _fallback(fusion_verdict, "no evidence to judge - using fusion verdict", status="skipped")

    model_name = model or settings.llm_model
    # Compact inputs: draft verdict without the bulky citation/log arrays,
    # plus one line per evidence item and per stance.
    prompt = JUDGE_PROMPT.format(
        draft_verdict_json=fusion_verdict.model_dump_json(
            indent=2, exclude={"checks_performed", "evidence_citations"}
        ),
        evidence_snippets="\n".join(
            f"[{e.evidence_id}] ({e.source_name}, tier {e.source_tier}): {e.snippet}" for e in evidence
        ),
        stance_results="\n".join(
            f"{s.claim_id} <- {s.evidence_id}: {s.stance} ({s.rationale})" for s in stance_results
        )
        or "(none)",
    )

    try:
        chain = _build_chain(model_name)
        notes = chain.invoke(prompt)
        if not isinstance(notes, _JudgeNotes):
            notes = _JudgeNotes.model_validate(notes)
    except Exception as exc:
        return _fallback(fusion_verdict, f"LLM call failed, using fusion verdict: {exc}")

    if notes.label != fusion_verdict.label and notes.label != VerdictLabel.UNVERIFIABLE:
        return _fallback(
            fusion_verdict,
            "LLM output violated a hard rule, using fusion verdict: "
            f"LLM label {notes.label!r} contradicts fusion label {fusion_verdict.label!r}",
        )

    final = fusion_verdict.model_copy(deep=True)
    final.label = notes.label
    if notes.caveats:  # empty LLM caveats must not erase fusion's transparency trail
        final.caveats = notes.caveats
    final.checks_performed = [*final.checks_performed, CheckLog(agent_name="judge", status="ok")]
    return final
