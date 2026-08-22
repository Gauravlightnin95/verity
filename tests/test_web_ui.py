"""Tests for the web UI surface: the static mount, the index route, and the
POST /report PDF endpoint.

No network and no pipeline run - /report takes a Verdict the caller already
holds, so it is exercised directly with a constructed one.
"""

from datetime import datetime, timezone

import pytest

from app.report import build_pdf
from core.schemas import CheckLog, EvidenceItem, PerClaimVerdict, Signal, Verdict, VerdictLabel

_RETRIEVED = datetime(2026, 8, 22, 10, 0, 0, tzinfo=timezone.utc)


def _verdict(**overrides) -> Verdict:
    base = dict(
        label=VerdictLabel.MOSTLY_TRUE,
        confidence=0.61,
        per_claim=[PerClaimVerdict(claim_id="c1", label=VerdictLabel.TRUE, supporting=3, refuting=0)],
        evidence_citations=[EvidenceItem(
            evidence_id="e1", claim_id="c1", snippet="ISRO confirmed the landing.",
            url="https://isro.gov.in/x", source_name="ISRO", source_tier=1,
            published_date="2023-08-23", retrieved_at=_RETRIEVED,
        )],
        signals=[Signal(name="source_credibility", score=0.8, weight=0.10, note="mostly tier-1")],
        caveats=["One claim could not be checked."],
        checks_performed=[CheckLog(agent_name="retrieval", status="ok", duration_ms=27)],
    )
    base.update(overrides)
    return Verdict(**base)


# ---------------- PDF builder ----------------
def test_build_pdf_returns_a_real_pdf():
    data = build_pdf(_verdict())
    assert data.startswith(b"%PDF-")
    assert len(data) > 800


def test_build_pdf_handles_an_empty_verdict():
    # An UNVERIFIABLE run with nothing found must still export, not crash.
    data = build_pdf(_verdict(
        label=VerdictLabel.UNVERIFIABLE, confidence=0.0,
        per_claim=[], evidence_citations=[], signals=[], caveats=[], checks_performed=[],
    ))
    assert data.startswith(b"%PDF-")


def test_build_pdf_survives_markup_in_scraped_text():
    # Snippets come off the open web. ReportLab's Paragraph parses a small
    # HTML dialect, so an unescaped "<" from a scraped page would abort the
    # whole export.
    hostile = EvidenceItem(
        evidence_id="e1", claim_id="c1",
        snippet='<b>unclosed & <script>alert("x")</script> 5 < 6',
        url="https://example.com/<>", source_name="A & B <news>", source_tier=3,
        retrieved_at=_RETRIEVED,
    )
    data = build_pdf(_verdict(evidence_citations=[hostile], caveats=["1 < 2 & 3 > 2"]))
    assert data.startswith(b"%PDF-")


# ---------------- routes ----------------
def test_report_endpoint_returns_pdf_bytes(api_client):
    res = api_client.post("/report", json=_verdict().model_dump(mode="json"))
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF-")


def test_report_endpoint_rejects_a_malformed_verdict(api_client):
    res = api_client.post("/report", json={"label": "NOT_A_LABEL", "confidence": 2})
    assert res.status_code == 422


def test_index_serves_the_web_ui(api_client):
    res = api_client.get("/")
    assert res.status_code == 200
    assert "VERITY" in res.text
    # The page must load its own assets from the mount, not the design canvas.
    assert "/ui/app.js" in res.text
    assert "support.js" not in res.text


@pytest.mark.parametrize("path", ["/ui/styles.css", "/ui/app.js"])
def test_static_assets_are_served(api_client, path):
    res = api_client.get(path)
    assert res.status_code == 200
    assert res.content


def test_health_still_works_alongside_the_static_mount(api_client):
    assert api_client.get("/health").json() == {"status": "ok"}
