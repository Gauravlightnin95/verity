"""Test-only fake capability functions, one per pipeline stage, matching
the real agents/*.py signatures exactly.

Now that agents/graph.py imports every real capability directly (the
mock/toggle system was removed once every real module was merged - see
CLAUDE.md / TEAM_GUIDE.md history), tests/conftest.py's
_fake_graph_capabilities fixture monkeypatches these onto agents.graph's
module namespace for test_graph.py / test_api.py, so the end-to-end graph
tests stay fast, deterministic, and require zero API keys/network - the
same role mocks/*.py used to serve, just scoped to tests instead of being
shipped as production code.

Shared-ID contract: extract_claims emits claim_id "c1"/"c2", retrieve and
stance are built to reference the same ids and evidence_id "e1"/"e2"/"e3",
and run_forensics emits Signal.name values ("halftone", "masthead_match")
matching core.fusion.NAME_TO_CATEGORY's aliases.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.schemas import (
    ArticleContent,
    Claim,
    EvidenceItem,
    InputPayload,
    OCRRegion,
    OCRResult,
    Signal,
    StanceResult,
)

_DISCLAIMER = (
    "stylistic signal only - AI-text detectors have known false-positive "
    "rates and cannot prove authorship"
)


def route(payload: InputPayload) -> dict:
    if payload.input_type == "text" and not (payload.text or "").strip():
        return {"reject_reason": "empty text input"}
    return {"language": "en", "reject_reason": None}


def scrape(url: str) -> ArticleContent:
    return ArticleContent(
        title="City Approves New Metro Line",
        body=(
            "The municipal corporation on Friday approved a new metro line "
            "connecting the northern suburbs to the city centre. Officials "
            "said the project, first proposed in 2024, will begin "
            "construction next year."
        ),
        author="Staff Reporter",
        publish_date="2026-07-11",
        domain="example-news.test",
        partial=False,
    )


def run_ocr(image_path: str) -> OCRResult:
    return OCRResult(
        regions=[
            OCRRegion(region_type="masthead", text="The Daily Example", confidence=0.96, bbox=(40, 22, 980, 110)),
            OCRRegion(region_type="dateline", text="New Delhi | 12 July 2026", confidence=0.91, bbox=(40, 118, 420, 150)),
            OCRRegion(region_type="headline", text="City Approves New Metro Line", confidence=0.94, bbox=(40, 160, 980, 230)),
            OCRRegion(region_type="body", text="The municipal corporation on Friday approved a new metro line...", confidence=0.88, bbox=(40, 240, 500, 900)),
        ],
        full_text=(
            "The Daily Example New Delhi | 12 July 2026 City Approves New "
            "Metro Line The municipal corporation on Friday approved a new "
            "metro line..."
        ),
        language="en",
        device_used="NPU",
        duration_ms=1840,
    )


def run_forensics(image_path: str, masthead_text: str | None = None) -> list[Signal]:
    return [
        Signal(
            name="halftone", score=0.82, weight=0.10,
            note="fake: print dot pattern detected in background patches",
            extras={"has_print_artifacts": True},
        ),
        Signal(
            name="masthead_match", score=0.91, weight=0.05,
            note="fake: closest match 'The Daily Example', similarity 0.91",
            extras={"publication_guess": "The Daily Example"},
        ),
    ]


def extract_claims(article: ArticleContent) -> list[Claim]:
    return [
        Claim(
            claim_id="c1",
            text="The municipal corporation approved a new metro line on 11 July 2026",
            entities=["municipal corporation", "metro line"],
            location="New Delhi",
            event_date="2026-07-11",
            checkability="high",
            genre="report",
        ),
        Claim(
            claim_id="c2",
            text="The project will be completed within 18 months",
            entities=["metro line"],
            location="New Delhi",
            event_date=None,
            checkability="medium",
            genre="report",
        ),
    ]


_RETRIEVED_AT = datetime(2026, 7, 14, 10, 22, 0, tzinfo=timezone.utc)

_EVIDENCE_BY_CLAIM: dict[str, list[EvidenceItem]] = {
    "c1": [
        EvidenceItem(
            evidence_id="e1", claim_id="c1",
            snippet="The ministry confirmed on Monday that the metro line was approved on 11 July.",
            url="https://pib.gov.in/example", source_name="PIB Fact Check", source_tier=1,
            published_date="2026-07-13", retrieved_at=_RETRIEVED_AT,
        ),
        EvidenceItem(
            evidence_id="e2", claim_id="c1",
            snippet="Coverage of the metro network expansion in the northern suburbs.",
            url="https://example-news.test/metro-expansion", source_name="Example News", source_tier=2,
            published_date="2026-07-12", retrieved_at=_RETRIEVED_AT,
        ),
    ],
    "c2": [
        EvidenceItem(
            evidence_id="e3", claim_id="c2",
            snippet="Officials declined to confirm an 18-month completion timeline.",
            url="https://factly.in/example", source_name="Factly", source_tier=1,
            published_date="2026-07-13", retrieved_at=_RETRIEVED_AT,
        ),
    ],
}


def retrieve(claims: list[Claim], publication_hint: str | None = None) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for claim in claims:
        items.extend(_EVIDENCE_BY_CLAIM.get(claim.claim_id, []))
    return items


_STANCE_BY_CLAIM: dict[str, list[tuple[str, str, str]]] = {
    "c1": [
        ("e1", "SUPPORTS", "Fake: reports the same approval and the same date."),
        ("e2", "NEUTRAL", "Fake: discusses the metro but not the approval itself."),
    ],
    "c2": [
        ("e3", "REFUTES", "Fake: officials declined to confirm the 18-month timeline."),
    ],
}


def stance(claim: Claim, evidence: list[EvidenceItem]) -> list[StanceResult]:
    evidence_ids = {e.evidence_id for e in evidence if e.claim_id == claim.claim_id}
    canned = _STANCE_BY_CLAIM.get(claim.claim_id, [])
    return [
        StanceResult(claim_id=claim.claim_id, evidence_id=eid, stance=label, rationale=rationale)
        for eid, label, rationale in canned
        if eid in evidence_ids
    ]


def stance_batch(
    claim_evidence_pairs: list[tuple[Claim, list[EvidenceItem]]]
) -> list[StanceResult]:
    """Batched fake: same canned stances, flattened across all claims."""
    results: list[StanceResult] = []
    for claim, evidence in claim_evidence_pairs:
        results.extend(stance(claim, evidence))
    return results


def score_credibility(evidence: list[EvidenceItem]) -> Signal:
    if not evidence:
        return Signal(name="source_credibility", score=0.0, weight=0.10, note="fake: no evidence to score")
    return Signal(
        name="source_credibility", score=0.8, weight=0.10,
        note="fake: mostly tier-1 sources (PIB Fact Check, Factly)",
    )


def check_temporal(claims: list[Claim], evidence: list[EvidenceItem]) -> Signal:
    return Signal(
        name="temporal_consistency", score=1.0, weight=0.05,
        note="fake: no old-news-as-new or future-dated anomalies detected",
    )


def detect_ai_text(text: str) -> Signal:
    if len((text or "").split()) < 150:
        return Signal(name="ai_text", score=0.5, weight=0.0, note=f"text under 150 words. {_DISCLAIMER}")
    return Signal(
        name="ai_text", score=0.85, weight=0.15,
        note=f"fake: low AI-generated probability. {_DISCLAIMER}",
    )
