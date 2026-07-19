"""
tests/test_stance_agent.py
=============================
Tests for `agents/stance_agent.py` (D.2).

Claude is never called for real here — `StanceAgent` is constructed with a
`FakeStructuredChatModel` (see `tests/fakes.py`), keeping these tests fast,
free, and fully offline.
"""

from __future__ import annotations

import pytest

from agents.stance_agent import (
    StanceAgent,
    StanceDetectionError,
    StanceResult,
    _StanceDetectionResponse,
)
from tests.conftest import FakeEvidence
from tests.fakes import FakeStructuredChatModel


def _response_from_fixture(fixture: list[dict]) -> _StanceDetectionResponse:
    return _StanceDetectionResponse(results=[StanceResult.model_validate(item) for item in fixture])


class TestStanceAgentSuccess:
    def test_stance_returns_mixed_results_from_fixture(
        self, fake_claim, fake_evidence_list, stance_results_sample_1
    ):
        # fixture has 4 results (e1-e4); align the fake evidence list to match
        evidence = fake_evidence_list + [FakeEvidence("e4", "A related but inconclusive snippet.")]
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(stance_results_sample_1)])
        agent = StanceAgent(llm=llm)

        results = agent.stance(fake_claim, evidence)

        assert len(results) == 4
        stances = {r.evidence_id: r.stance for r in results}
        assert stances["e1"] == "SUPPORTS"
        assert stances["e3"] == "REFUTES"
        assert stances["e4"] == "NEUTRAL"

    def test_stance_honesty_rule_in_prompt(self, fake_claim, fake_evidence_list):
        seen = {}

        def capture(messages):
            seen["system"] = messages[0].content

        llm = FakeStructuredChatModel(
            responses=[_StanceDetectionResponse(results=[])], on_invoke=capture
        )
        agent = StanceAgent(llm=llm)
        agent.stance(fake_claim, fake_evidence_list)

        assert "judge ONLY from the snippets" in seen["system"]
        assert "Do not use your own background knowledge" in seen["system"]


class TestStanceAgentEdgeCases:
    def test_empty_evidence_short_circuits_without_calling_llm(self, fake_claim):
        llm = FakeStructuredChatModel(responses=[])
        agent = StanceAgent(llm=llm)

        results = agent.stance(fake_claim, [])

        assert results == []
        assert llm.call_count == 0

    def test_empty_claim_text_raises(self, fake_evidence_list):
        class BlankClaim:
            claim_id = "c1"
            text = "   "

        llm = FakeStructuredChatModel(responses=[])
        agent = StanceAgent(llm=llm)

        with pytest.raises(StanceDetectionError, match="empty text"):
            agent.stance(BlankClaim(), fake_evidence_list)

    def test_evidence_over_cap_is_truncated_before_calling_model(self, fake_claim):
        nine_items = [FakeEvidence(f"e{i}", f"Snippet number {i}") for i in range(1, 10)]
        captured = {}

        def capture(messages):
            captured["human"] = messages[0].content

        llm = FakeStructuredChatModel(
            responses=[_StanceDetectionResponse(results=[])], on_invoke=capture
        )
        agent = StanceAgent(llm=llm, max_snippets_per_claim=8)
        agent.stance(fake_claim, nine_items)

        assert "evidence_id=e9" not in captured["human"]
        assert "evidence_id=e1" in captured["human"]

    def test_missing_result_defaults_to_neutral(self, fake_claim, fake_evidence_list):
        # Model only returns a stance for e1; e2 and e3 are missing.
        partial = _StanceDetectionResponse(
            results=[
                StanceResult(claim_id="c1", evidence_id="e1", stance="SUPPORTS", rationale="Confirmed.")
            ]
        )
        llm = FakeStructuredChatModel(responses=[partial])
        agent = StanceAgent(llm=llm)

        results = agent.stance(fake_claim, fake_evidence_list)

        assert len(results) == 3  # one per evidence item, always
        by_id = {r.evidence_id: r for r in results}
        assert by_id["e1"].stance == "SUPPORTS"
        assert by_id["e2"].stance == "NEUTRAL"
        assert "defaulted to NEUTRAL" in by_id["e2"].rationale
        assert by_id["e3"].stance == "NEUTRAL"


class TestStanceAgentRetries:
    def test_retries_then_succeeds(self, fake_claim, fake_evidence_list, monkeypatch):
        monkeypatch.setattr("agents.stance_agent.time.sleep", lambda _seconds: None)

        good_response = _StanceDetectionResponse(
            results=[
                StanceResult(claim_id="c1", evidence_id=e.evidence_id, stance="NEUTRAL", rationale="ok")
                for e in fake_evidence_list
            ]
        )
        llm = FakeStructuredChatModel(
            responses=[RuntimeError("simulated transient API error"), good_response]
        )
        agent = StanceAgent(llm=llm, max_retries=3, retry_backoff_seconds=0.01)

        results = agent.stance(fake_claim, fake_evidence_list)

        assert len(results) == len(fake_evidence_list)
        assert llm.call_count == 2

    def test_all_retries_exhausted_raises(self, fake_claim, fake_evidence_list, monkeypatch):
        monkeypatch.setattr("agents.stance_agent.time.sleep", lambda _seconds: None)

        llm = FakeStructuredChatModel(responses=[RuntimeError("down")] * 3)
        agent = StanceAgent(llm=llm, max_retries=3, retry_backoff_seconds=0.01)

        with pytest.raises(StanceDetectionError, match="failed after 3 attempts"):
            agent.stance(fake_claim, fake_evidence_list)
