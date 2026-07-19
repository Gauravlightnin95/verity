"""Validates the fixture JSON samples (the docs' own examples) against
core.schemas, and exercises the schema-layer hard rules."""

import pytest
from pydantic import ValidationError

from core.schemas import (
    Claim,
    EvidenceItem,
    InputPayload,
    OCRResult,
    Signal,
    StanceResult,
    Verdict,
    VerdictLabel,
    VerityState,
)


def test_verdict_sample_validates(contract):
    data = contract("verdict_sample.json")
    verdict = Verdict.model_validate(data)
    assert verdict.label == VerdictLabel.MISLEADING
    assert verdict.per_claim[0].claim_id == "c1"
    assert verdict.evidence_citations[0].source_tier == 1


def test_ocr_result_sample_validates(contract):
    data = contract("ocr_result_sample.json")
    ocr = OCRResult.model_validate(data)
    assert len(ocr.regions) == 4
    assert ocr.device_used == "NPU"
    assert ocr.regions[0].bbox == (40, 22, 980, 110)


def test_claims_sample_validates(contract):
    data = contract("claims_sample.json")
    claims = [Claim.model_validate(c) for c in data]
    assert claims[0].checkability == "high"
    assert claims[1].event_date is None


def test_stance_results_sample_validates(contract):
    data = contract("stance_results_sample.json")
    stances = [StanceResult.model_validate(s) for s in data]
    assert stances[0].stance == "SUPPORTS"


def test_forensics_signals_sample_validates(contract):
    data = contract("forensics_signals_sample.json")
    signals = [Signal.model_validate(s) for s in data]
    assert signals[0].name == "halftone"
    assert signals[1].extras["publication_guess"] == "The Daily Example"


def test_verdict_without_citations_must_be_unverifiable():
    with pytest.raises(ValidationError):
        Verdict(label=VerdictLabel.TRUE, confidence=0.9, evidence_citations=[])

    # UNVERIFIABLE with no citations is fine.
    Verdict(label=VerdictLabel.UNVERIFIABLE, confidence=0.0, evidence_citations=[])


def test_input_payload_requires_matching_field():
    with pytest.raises(ValidationError):
        InputPayload(input_type="url", url=None)

    InputPayload(input_type="url", url="https://example.com/article")


def test_evidence_item_requires_valid_tier():
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="e1",
            claim_id="c1",
            snippet="x",
            url="https://example.com",
            source_name="Example",
            source_tier=4,
            retrieved_at="2026-07-14T10:22:00Z",
        )


def test_verity_state_defaults_are_empty():
    state = VerityState()
    assert state.claims == []
    assert state.evidence == []
    assert state.verdict is None
