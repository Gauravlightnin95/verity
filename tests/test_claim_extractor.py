"""
tests/test_claim_extractor.py
================================
Tests for `agents/claim_extractor.py` (D.1).

Claude is never called for real here — `ClaimExtractor` is constructed
with a `FakeStructuredChatModel` (see `tests/fakes.py`), keeping these
tests fast, free, and fully offline, per the team's testing contract.
"""

from __future__ import annotations

import pytest

from agents.claim_extractor import (
    Claim,
    ClaimExtractionError,
    ClaimExtractor,
    _ClaimExtractionResponse,
)
from tests.fakes import FakeStructuredChatModel


def _response_from_fixture(fixture: list[dict]) -> _ClaimExtractionResponse:
    return _ClaimExtractionResponse(claims=[Claim.model_validate(item) for item in fixture])


class TestClaimExtractorSuccess:
    def test_extract_claims_returns_claims_from_fixture(self, fake_article, claims_sample_1):
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_1)])
        extractor = ClaimExtractor(llm=llm)

        claims = extractor.extract_claims(fake_article)

        assert len(claims) == 2
        assert claims[0].claim_id == "c1"
        assert claims[0].checkability == "high"
        assert claims[1].checkability == "medium"

    def test_extract_claims_opinion_genre_all_low_checkability(self, fake_article, claims_sample_2):
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_2)])
        extractor = ClaimExtractor(llm=llm)

        claims = extractor.extract_claims(fake_article)

        assert len(claims) == 3
        assert all(c.genre == "opinion" for c in claims)
        assert all(c.checkability == "low" for c in claims)

    def test_prompt_includes_claim_cap_and_honesty_rules(self, fake_article, claims_sample_1):
        seen_messages = {}

        def capture(messages):
            seen_messages["messages"] = messages

        llm = FakeStructuredChatModel(
            responses=[_response_from_fixture(claims_sample_1)], on_invoke=capture
        )
        extractor = ClaimExtractor(llm=llm, max_claims=5)
        extractor.extract_claims(fake_article)

        system_prompt = seen_messages["messages"][0].content
        assert "at most 5 claims" in system_prompt
        assert "Respond ONLY via the provided tool/schema" in system_prompt


class TestClaimExtractorValidation:
    def test_empty_article_body_raises(self, fake_article):
        fake_article.body = "   "
        llm = FakeStructuredChatModel(responses=[])
        extractor = ClaimExtractor(llm=llm)

        with pytest.raises(ClaimExtractionError, match="empty body"):
            extractor.extract_claims(fake_article)

    def test_over_limit_response_is_truncated(self, fake_article):
        # The wrapper schema's own `max_length` (tied to the team-wide
        # MAX_CLAIMS_PER_ARTICLE constant) blocks a response with more claims
        # than that ceiling from being constructed at all - so to exercise
        # ClaimExtractor's own defensive `_enforce_claim_limit` truncation,
        # build up to the schema ceiling and set a *tighter* per-instance
        # max_claims. This is a real scenario: a caller tightening the cap
        # below the team default without changing the shared schema.
        max_claims = _ClaimExtractionResponse(
            claims=[
                Claim(
                    claim_id=f"c{i}",
                    text=f"Claim number {i}",
                    entities=[],
                    checkability="medium",
                    genre="report",
                )
                for i in range(1, 4)  # 3 claims - the schema ceiling
            ],
        )
        llm = FakeStructuredChatModel(responses=[max_claims])
        extractor = ClaimExtractor(llm=llm, max_claims=2)  # tighter than the schema cap

        claims = extractor.extract_claims(fake_article)

        assert len(claims) == 2
        assert claims[-1].claim_id == "c2"


class TestClaimExtractorRetries:
    def test_retries_then_succeeds(self, fake_article, claims_sample_1, monkeypatch):
        monkeypatch.setattr("agents.claim_extractor.time.sleep", lambda _seconds: None)

        llm = FakeStructuredChatModel(
            responses=[
                RuntimeError("simulated transient API error"),
                _response_from_fixture(claims_sample_1),
            ]
        )
        extractor = ClaimExtractor(llm=llm, max_retries=3, retry_backoff_seconds=0.01)

        claims = extractor.extract_claims(fake_article)

        assert len(claims) == 2
        assert llm.call_count == 2

    def test_all_retries_exhausted_raises(self, fake_article, monkeypatch):
        monkeypatch.setattr("agents.claim_extractor.time.sleep", lambda _seconds: None)

        llm = FakeStructuredChatModel(
            responses=[RuntimeError("down"), RuntimeError("still down"), RuntimeError("still down")]
        )
        extractor = ClaimExtractor(llm=llm, max_retries=3, retry_backoff_seconds=0.01)

        with pytest.raises(ClaimExtractionError, match="failed after 3 attempts"):
            extractor.extract_claims(fake_article)

        assert llm.call_count == 3
