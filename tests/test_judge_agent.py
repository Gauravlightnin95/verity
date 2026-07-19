"""Tests agents.judge_agent's LLM integration by mocking at the
_build_chain() seam - not the HTTP layer - mirroring how Member B's
respx-based tests fake things at a seam this project controls.
No test in this file requires a real GROQ_API_KEY."""

from datetime import datetime, timezone

import agents.judge_agent as judge_agent
from agents.judge_agent import _JudgeNotes
from core.schemas import EvidenceItem, PerClaimVerdict, Signal, Verdict, VerdictLabel

_NOW = datetime(2026, 7, 14, tzinfo=timezone.utc)

# Non-empty evidence so judge() actually runs the LLM path (it skips the
# call entirely when there is no evidence - see test_judge_skips_llm_...).
_EVIDENCE = [
    EvidenceItem(
        evidence_id="e1",
        claim_id="c1",
        snippet="x",
        url="https://example.com",
        source_name="Example",
        source_tier=1,
        retrieved_at=_NOW,
    )
]


def _fusion_verdict(label=VerdictLabel.TRUE) -> Verdict:
    return Verdict(
        label=label,
        confidence=0.8,
        per_claim=[PerClaimVerdict(claim_id="c1", label=label, supporting=1, refuting=0)],
        evidence_citations=list(_EVIDENCE),
        signals=[Signal(name="source_credibility", score=0.9, weight=0.10, note="tier-1")],
        caveats=["auto caveat from fusion"],
        checks_performed=[],
    )


class _FakeChain:
    def __init__(self, result):
        self._result = result

    def invoke(self, _prompt):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def test_judge_applies_polished_caveats_and_keeps_label(monkeypatch):
    fusion_verdict = _fusion_verdict()
    notes = _JudgeNotes(label=VerdictLabel.TRUE, caveats=["polished by the LLM"])
    monkeypatch.setattr(judge_agent, "_build_chain", lambda model_name: _FakeChain(notes))

    result = judge_agent.judge(fusion_verdict, _EVIDENCE, [])

    assert result.label == VerdictLabel.TRUE
    assert result.caveats == ["polished by the LLM"]
    assert result.checks_performed[-1].status == "ok"


def test_judge_cannot_touch_citations_signals_or_confidence(monkeypatch):
    # Structural guarantee: the LLM only returns label + caveats, so
    # citations, signals, per-claim rollups, and confidence are copied
    # from the fusion draft verbatim no matter what.
    fusion_verdict = _fusion_verdict()
    notes = _JudgeNotes(label=VerdictLabel.TRUE, caveats=["reworded"])
    monkeypatch.setattr(judge_agent, "_build_chain", lambda model_name: _FakeChain(notes))

    result = judge_agent.judge(fusion_verdict, _EVIDENCE, [])

    assert result.evidence_citations == fusion_verdict.evidence_citations
    assert result.signals == fusion_verdict.signals
    assert result.per_claim == fusion_verdict.per_claim
    assert result.confidence == fusion_verdict.confidence


def test_judge_falls_back_when_llm_contradicts_the_fusion_label(monkeypatch):
    fusion_verdict = _fusion_verdict(label=VerdictLabel.TRUE)
    notes = _JudgeNotes(label=VerdictLabel.FALSE, caveats=[])
    monkeypatch.setattr(judge_agent, "_build_chain", lambda model_name: _FakeChain(notes))

    result = judge_agent.judge(fusion_verdict, _EVIDENCE, [])

    assert result.label == VerdictLabel.TRUE  # fusion's label wins
    assert result.checks_performed[-1].status == "failed"
    assert result.caveats == ["auto caveat from fusion"]  # untouched


def test_judge_allows_llm_to_downgrade_to_unverifiable(monkeypatch):
    fusion_verdict = _fusion_verdict(label=VerdictLabel.TRUE)
    notes = _JudgeNotes(label=VerdictLabel.UNVERIFIABLE, caveats=["evidence too thin"])
    monkeypatch.setattr(judge_agent, "_build_chain", lambda model_name: _FakeChain(notes))

    result = judge_agent.judge(fusion_verdict, _EVIDENCE, [])

    assert result.label == VerdictLabel.UNVERIFIABLE


def test_judge_keeps_fusion_caveats_when_llm_returns_none(monkeypatch):
    # An empty caveat list from the LLM must not erase the transparency
    # trail fusion built (skipped checks, unverifiable claims, ...).
    fusion_verdict = _fusion_verdict()
    notes = _JudgeNotes(label=VerdictLabel.TRUE, caveats=[])
    monkeypatch.setattr(judge_agent, "_build_chain", lambda model_name: _FakeChain(notes))

    result = judge_agent.judge(fusion_verdict, _EVIDENCE, [])

    assert result.caveats == ["auto caveat from fusion"]


def test_judge_falls_back_when_the_llm_call_raises(monkeypatch):
    fusion_verdict = _fusion_verdict()
    monkeypatch.setattr(
        judge_agent, "_build_chain", lambda model_name: _FakeChain(RuntimeError("no API key"))
    )

    result = judge_agent.judge(fusion_verdict, _EVIDENCE, [])

    assert result.label == fusion_verdict.label
    assert result.checks_performed[-1].status == "failed"
    assert "no API key" in result.checks_performed[-1].note


def test_judge_skips_llm_when_there_is_no_evidence(monkeypatch):
    # With zero evidence, fusion already forced UNVERIFIABLE and the judge
    # can cite nothing - so it must NOT call the LLM at all.
    fusion_verdict = _fusion_verdict(label=VerdictLabel.UNVERIFIABLE)

    def _boom(model_name):
        raise AssertionError("judge must not build a chain when evidence is empty")

    monkeypatch.setattr(judge_agent, "_build_chain", _boom)

    result = judge_agent.judge(fusion_verdict, [], [])

    assert result.label == VerdictLabel.UNVERIFIABLE
    assert result.checks_performed[-1].status == "skipped"


def test_judge_works_with_zero_api_keys_configured(monkeypatch):
    monkeypatch.setattr(judge_agent.settings, "groq_api_key", None)
    fusion_verdict = _fusion_verdict()

    def _raise_missing_key(model_name):
        raise RuntimeError("GROQ_API_KEY not set")

    monkeypatch.setattr(judge_agent, "_build_chain", _raise_missing_key)

    result = judge_agent.judge(fusion_verdict, _EVIDENCE, [])

    assert result.label == fusion_verdict.label
