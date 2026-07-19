"""
tests/test_streamlit_app.py
==============================
Tests for the testable business logic inside `ui/streamlit_app.py` (D.4):
`VerifyClient` (HTTP calls to Member A's backend) and `PDFReportBuilder`.

`respx` intercepts `httpx` calls so these tests never touch a real network
or require Member A's backend to actually be running — the same offline
testing pattern Member B uses for its scraping/retrieval calls.

Streamlit's own widget rendering (`st.tabs`, `st.button`, etc.) is UI glue
best covered by manual QA / `streamlit.testing.v1.AppTest` in a later pass;
it is intentionally out of scope here so these tests stay fast and focused
on logic that can silently break (HTTP handling, retries, PDF generation).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from ui.streamlit_app import PDFReportBuilder, VerdictResponse, VerifyClient, VerifyRequestError

BACKEND_URL = "http://test-backend:8000"

SAMPLE_VERDICT = {
    "label": "MISLEADING",
    "confidence": 0.71,
    "per_claim": [
        {"claim_id": "c1", "label": "TRUE", "supporting": 3, "refuting": 0},
        {"claim_id": "c2", "label": "FALSE", "supporting": 0, "refuting": 2},
    ],
    "evidence_citations": [
        {
            "evidence_id": "e1",
            "claim_id": "c1",
            "snippet": "The ministry confirmed on Monday that...",
            "url": "https://pib.gov.in/example",
            "source_name": "PIB Fact Check",
            "source_tier": 1,
            "retrieved_at": "2026-07-14T10:22:00Z",
        }
    ],
    "signals": [
        {
            "name": "evidence_stance",
            "score": 0.62,
            "weight": 0.45,
            "note": "3 snippets support, 2 refute across claims",
            "extras": {},
        },
        {
            "name": "ai_text",
            "score": 0.58,
            "weight": 0.15,
            "note": "stylistic signal only - detectors can be wrong",
            "extras": {},
        },
    ],
    "caveats": ["Image checks skipped: text input"],
    "checks_performed": [
        {"agent_name": "retrieval", "status": "ok", "duration_ms": 2140, "note": "12 evidence items"}
    ],
}


class TestVerifyClientSuccess:
    @respx.mock
    def test_verify_text_returns_parsed_verdict(self):
        respx.post(f"{BACKEND_URL}/verify").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERDICT)
        )
        client = VerifyClient(base_url=BACKEND_URL)

        verdict = client.verify_text("Some article text")

        assert isinstance(verdict, VerdictResponse)
        assert verdict.label == "MISLEADING"
        assert verdict.confidence == 0.71
        assert len(verdict.per_claim) == 2

    @respx.mock
    def test_verify_url_sends_correct_payload(self):
        route = respx.post(f"{BACKEND_URL}/verify").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERDICT)
        )
        client = VerifyClient(base_url=BACKEND_URL)

        client.verify_url("https://example.com/article")

        # Sent as form-encoded data, matching the backend's Form(...)
        # fields - NOT JSON, which would silently fail to populate them.
        request = route.calls.last.request
        assert request.headers["content-type"] == "application/x-www-form-urlencoded"
        sent_body = request.content
        assert b"url=https%3A%2F%2Fexample.com%2Farticle" in sent_body

    @respx.mock
    def test_verify_image_sends_multipart_file(self):
        respx.post(f"{BACKEND_URL}/verify").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERDICT)
        )
        client = VerifyClient(base_url=BACKEND_URL)

        verdict = client.verify_image(b"fake-image-bytes", "clipping.jpg")

        assert verdict.label == "MISLEADING"


class TestVerifyClientFailureHandling:
    @respx.mock
    def test_retries_on_server_error_then_succeeds(self):
        route = respx.post(f"{BACKEND_URL}/verify").mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(200, json=SAMPLE_VERDICT),
            ]
        )
        client = VerifyClient(base_url=BACKEND_URL, max_retries=2, retry_backoff_seconds=0.01)

        verdict = client.verify_text("text")

        assert verdict.label == "MISLEADING"
        assert route.call_count == 2

    @respx.mock
    def test_raises_verify_request_error_after_exhausting_retries(self):
        respx.post(f"{BACKEND_URL}/verify").mock(return_value=httpx.Response(503))
        client = VerifyClient(base_url=BACKEND_URL, max_retries=2, retry_backoff_seconds=0.01)

        with pytest.raises(VerifyRequestError, match="could not reach backend"):
            client.verify_text("text")

    @respx.mock
    def test_connection_error_is_wrapped_as_verify_request_error(self):
        respx.post(f"{BACKEND_URL}/verify").mock(side_effect=httpx.ConnectError("refused"))
        client = VerifyClient(base_url=BACKEND_URL, max_retries=1, retry_backoff_seconds=0.01)

        with pytest.raises(VerifyRequestError):
            client.verify_text("text")


class TestVerdictResponseHelpers:
    def test_signal_lookup_by_name(self):
        verdict = VerdictResponse.model_validate(SAMPLE_VERDICT)

        ai_signal = verdict.signal("ai_text")

        assert ai_signal is not None
        assert ai_signal.weight == 0.15

    def test_signal_lookup_missing_returns_none(self):
        verdict = VerdictResponse.model_validate(SAMPLE_VERDICT)

        assert verdict.signal("halftone") is None

    def test_device_used_defaults_to_none_when_absent(self):
        verdict = VerdictResponse.model_validate(SAMPLE_VERDICT)

        assert verdict.device_used() is None

    def test_device_used_found_in_signal_extras(self):
        data = dict(SAMPLE_VERDICT)
        data["signals"] = [
            {"name": "ocr", "score": 0.9, "weight": 0.0, "note": "n/a", "extras": {"device_used": "NPU"}}
        ]
        verdict = VerdictResponse.model_validate(data)

        assert verdict.device_used() == "NPU"


class TestPDFReportBuilder:
    def test_build_produces_valid_pdf_bytes(self):
        verdict = VerdictResponse.model_validate(SAMPLE_VERDICT)
        builder = PDFReportBuilder()

        pdf_bytes = builder.build(verdict)

        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 0

    def test_build_handles_verdict_with_no_evidence_or_caveats(self):
        minimal = {"label": "UNVERIFIABLE", "confidence": 0.0}
        verdict = VerdictResponse.model_validate(minimal)
        builder = PDFReportBuilder()

        pdf_bytes = builder.build(verdict)

        assert pdf_bytes[:4] == b"%PDF"
