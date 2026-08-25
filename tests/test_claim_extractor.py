"""
tests/test_claim_extractor.py
================================
Tests for `agents/claim_extractor.py` (D.1).

Neither model is called for real here — `ClaimExtractor` is constructed
with a `FakeStructuredChatModel` for the LLM step and a
`FakeClaimClassifier` for the check-worthiness step (both in
`tests/fakes.py`), keeping these tests fast, free, and fully offline, per
the team's testing contract. The real classifier downloads a Hugging Face
model on first use, so injecting it is what keeps that contract true.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import pytest

from agents.claim_extractor import (
    Claim,
    ClaimExtractionError,
    ClaimExtractor,
    _ClaimExtractionResponse,
)
from tests.fakes import FakeClaimClassifier, FakeStructuredChatModel

CHECK_WORTHY = "Check-worthy Factual"
UNIMPORTANT = "Unimportant Factual"
NON_FACTUAL = "Non-factual"


def _response_from_fixture(fixture: list[dict]) -> _ClaimExtractionResponse:
    return _ClaimExtractionResponse(claims=[Claim.model_validate(item) for item in fixture])


def _extractor(
    llm: FakeStructuredChatModel,
    classifier: Optional[FakeClaimClassifier] = None,
    **kwargs,
) -> ClaimExtractor:
    """Default to a classifier that finds every claim check-worthy, so
    tests about the LLM step aren't also asserting on the filter."""
    return ClaimExtractor(
        llm=llm, claim_classifier=classifier or FakeClaimClassifier(), **kwargs
    )


@pytest.fixture(autouse=True)
def claims_csv(tmp_path, monkeypatch) -> Path:
    """Redirect the per-run claim.csv snapshot into tmp_path - the suite
    must not write into whatever directory pytest was launched from."""
    path = tmp_path / "claim.csv"
    monkeypatch.setattr("agents.claim_extractor._CLAIMS_CSV_PATH", path)
    return path


def _read_csv(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


class TestClaimExtractorSuccess:
    def test_extract_claims_returns_claims_from_fixture(self, fake_article, claims_sample_1):
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_1)])
        extractor = _extractor(llm)

        claims = extractor.extract_claims(fake_article)

        assert len(claims) == 2
        assert claims[0].claim_id == "c1"
        assert claims[0].checkability == "high"
        assert claims[1].checkability == "medium"

    def test_extract_claims_opinion_genre_all_low_checkability(self, fake_article, claims_sample_2):
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_2)])
        extractor = _extractor(llm)

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
        extractor = _extractor(llm, max_claims=5)
        extractor.extract_claims(fake_article)

        system_prompt = seen_messages["messages"][0].content
        assert "at most 5 claims" in system_prompt
        assert "Respond ONLY via the provided tool/schema" in system_prompt


class TestCheckWorthinessFilter:
    def test_only_check_worthy_claims_are_returned(self, fake_article, claims_sample_1):
        # c1 is a dated, verifiable fact; c2 is a prediction about the
        # future - exactly the split the filter exists to make.
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_1)])
        classifier = FakeClaimClassifier(labels=[CHECK_WORTHY, NON_FACTUAL])
        extractor = _extractor(llm, classifier)

        claims = extractor.extract_claims(fake_article)

        assert [c.claim_id for c in claims] == ["c1"]
        # One classifier call per claim, feeding both the CSV and the filter.
        assert classifier.call_count == 2

    def test_unimportant_factual_is_dropped_too(self, fake_article, claims_sample_1):
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_1)])
        extractor = _extractor(llm, FakeClaimClassifier(labels=[UNIMPORTANT]))

        assert extractor.extract_claims(fake_article) == []

    def test_no_check_worthy_claims_returns_empty_list(self, fake_article, claims_sample_2):
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_2)])
        extractor = _extractor(llm, FakeClaimClassifier(labels=[NON_FACTUAL]))

        claims = extractor.extract_claims(fake_article)

        # An opinion piece with nothing checkable in it is a legitimate
        # outcome, not an error - agents/graph.py turns it into a verdict.
        assert claims == []

    def test_csv_records_every_claim_including_dropped_ones(
        self, fake_article, claims_sample_1, claims_csv
    ):
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_1)])
        extractor = _extractor(llm, FakeClaimClassifier(labels=[CHECK_WORTHY, NON_FACTUAL]))

        extractor.extract_claims(fake_article)

        header, *rows = _read_csv(claims_csv)
        assert header == ["claim_id", "claim", "label"]
        assert [(row[0], row[2]) for row in rows] == [("c1", CHECK_WORTHY), ("c2", NON_FACTUAL)]

    def test_csv_is_a_snapshot_of_the_latest_run_only(
        self, fake_article, claims_sample_1, claims_sample_2, claims_csv
    ):
        llm = FakeStructuredChatModel(
            responses=[
                _response_from_fixture(claims_sample_1),
                _response_from_fixture(claims_sample_2),
            ]
        )
        extractor = _extractor(llm)

        extractor.extract_claims(fake_article)
        extractor.extract_claims(fake_article)

        _, *rows = _read_csv(claims_csv)
        assert len(rows) == 3  # the second run's three claims, not 2 + 3

    def test_csv_write_failure_is_not_fatal(self, fake_article, claims_sample_1, monkeypatch, tmp_path):
        # A directory where the file should be: open(..., "w") raises.
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        monkeypatch.setattr("agents.claim_extractor._CLAIMS_CSV_PATH", blocked)

        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_1)])
        extractor = _extractor(llm)

        assert len(extractor.extract_claims(fake_article)) == 2

    def test_classifier_unavailable_for_every_claim_raises(self, fake_article, claims_sample_1):
        # A dead classifier labels nothing, which a strict filter would
        # turn into an empty list - indistinguishable from a clean run
        # that found nothing checkable. It has to raise instead.
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_1)])
        extractor = _extractor(llm, FakeClaimClassifier(labels=[RuntimeError("no transformers")]))

        with pytest.raises(ClaimExtractionError, match="could not label any claim"):
            extractor.extract_claims(fake_article)

    def test_partially_unavailable_classifier_still_returns_what_it_labeled(
        self, fake_article, claims_sample_1
    ):
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_1)])
        extractor = _extractor(
            llm, FakeClaimClassifier(labels=[CHECK_WORTHY, RuntimeError("transient")])
        )

        claims = extractor.extract_claims(fake_article)

        assert [c.claim_id for c in claims] == ["c1"]

    def test_no_claims_at_all_does_not_raise(self, fake_article):
        # Nothing to label is not a dead classifier - the "all unavailable"
        # guard must not fire on an empty list.
        llm = FakeStructuredChatModel(responses=[_ClaimExtractionResponse(claims=[])])
        extractor = _extractor(llm)

        assert extractor.extract_claims(fake_article) == []

    def test_label_matching_ignores_case_and_whitespace(self, fake_article, claims_sample_1):
        llm = FakeStructuredChatModel(responses=[_response_from_fixture(claims_sample_1)])
        extractor = _extractor(llm, FakeClaimClassifier(labels=["  check-worthy factual "]))

        assert len(extractor.extract_claims(fake_article)) == 2


class TestClaimExtractorValidation:
    def test_empty_article_body_raises(self, fake_article):
        fake_article.body = "   "
        llm = FakeStructuredChatModel(responses=[])
        extractor = _extractor(llm)

        with pytest.raises(ClaimExtractionError, match="empty body"):
            extractor.extract_claims(fake_article)

    def test_over_limit_response_is_truncated(self, fake_article):
        # The wrapper schema's own `max_length` (tied to the team-wide
        # MAX_CLAIMS_PER_ARTICLE ceiling) blocks a response with more claims
        # than that from being constructed at all - so to exercise
        # ClaimExtractor's own defensive `_enforce_claim_limit` truncation,
        # set a *tighter* per-instance max_claims. This is a real scenario:
        # a caller tightening the cap below the team default without
        # changing the shared schema.
        response = _ClaimExtractionResponse(
            claims=[
                Claim(
                    claim_id=f"c{i}",
                    text=f"Claim number {i}",
                    entities=[],
                    checkability="medium",
                    genre="report",
                )
                for i in range(1, 4)
            ],
        )
        llm = FakeStructuredChatModel(responses=[response])
        classifier = FakeClaimClassifier()
        extractor = _extractor(llm, classifier, max_claims=2)  # tighter than the schema cap

        claims = extractor.extract_claims(fake_article)

        assert len(claims) == 2
        assert claims[-1].claim_id == "c2"
        # Truncation happens before classification: the dropped claim is
        # never paid for.
        assert classifier.call_count == 2


class TestClaimExtractorRetries:
    def test_retries_then_succeeds(self, fake_article, claims_sample_1, monkeypatch):
        monkeypatch.setattr("agents.claim_extractor.time.sleep", lambda _seconds: None)

        llm = FakeStructuredChatModel(
            responses=[
                RuntimeError("simulated transient API error"),
                _response_from_fixture(claims_sample_1),
            ]
        )
        extractor = _extractor(llm, max_retries=3, retry_backoff_seconds=0.01)

        claims = extractor.extract_claims(fake_article)

        assert len(claims) == 2
        assert llm.call_count == 2

    def test_all_retries_exhausted_raises(self, fake_article, monkeypatch):
        monkeypatch.setattr("agents.claim_extractor.time.sleep", lambda _seconds: None)

        llm = FakeStructuredChatModel(
            responses=[RuntimeError("down"), RuntimeError("still down"), RuntimeError("still down")]
        )
        extractor = _extractor(llm, max_retries=3, retry_backoff_seconds=0.01)

        with pytest.raises(ClaimExtractionError, match="failed after 3 attempts"):
            extractor.extract_claims(fake_article)

        assert llm.call_count == 3
