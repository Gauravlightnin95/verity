"""ui/streamlit_app.py - Member D's demo frontend.

Three input tabs (paste text / paste URL / upload photo). Talks ONLY to
the FastAPI backend's POST /verify - never imports pipeline/agent code
directly, keeping the "baton" pattern intact: the backend owns the
pipeline, the UI only renders what comes back.

Run with: streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import io
import time
from typing import Optional

import httpx
import streamlit as st
from pydantic import BaseModel, Field
from reportlab.lib import colors as pdf_colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.config import settings
from ui.constants import VERDICT_COLORS
from utils.logging_conf import get_logger

logger = get_logger(__name__)


class VerifyRequestError(Exception):
    """Raised when the backend /verify call fails after all retries."""


# Response models mirror core.schemas.Verdict, kept UI-local (rather than
# importing core.schemas directly) so the UI only ever depends on the
# backend's HTTP contract, matching the "baton" pattern - the UI process
# never imports pipeline code.


class PerClaimResult(BaseModel):
    claim_id: str
    label: str
    supporting: int = 0
    refuting: int = 0


class EvidenceCitation(BaseModel):
    evidence_id: str
    claim_id: str
    snippet: str
    url: str
    source_name: str
    source_tier: int
    retrieved_at: str


class SignalInfo(BaseModel):
    name: str
    score: float
    weight: float
    note: str
    extras: Optional[dict] = None


class CheckLog(BaseModel):
    agent_name: str
    status: str  # "ok" | "skipped" | "failed"
    duration_ms: int
    note: Optional[str] = None


class VerdictResponse(BaseModel):
    """The full response shape returned by POST /verify."""

    label: str
    confidence: float
    per_claim: list[PerClaimResult] = Field(default_factory=list)
    evidence_citations: list[EvidenceCitation] = Field(default_factory=list)
    signals: list[SignalInfo] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    checks_performed: list[CheckLog] = Field(default_factory=list)

    def signal(self, name: str) -> Optional[SignalInfo]:
        return next((s for s in self.signals if s.name == name), None)

    def device_used(self) -> Optional[str]:
        """Best-effort lookup of which chip a local model ran on."""
        for signal in self.signals:
            device = (signal.extras or {}).get("device_used")
            if device:
                return device
        return None


class VerifyClient:
    """Thin HTTP client for the backend's POST /verify endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self._base_url = (base_url or settings.backend_base_url).rstrip("/")
        self._timeout_seconds = timeout_seconds or settings.backend_timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    def verify_text(self, text: str) -> VerdictResponse:
        return self._post_verify({"text": text})

    def verify_url(self, url: str) -> VerdictResponse:
        return self._post_verify({"url": url})

    def verify_image(self, image_bytes: bytes, filename: str) -> VerdictResponse:
        return self._post_verify({}, files={"image": (filename, image_bytes)})

    def _post_verify(self, payload: dict, files: Optional[dict] = None) -> VerdictResponse:
        """POST to /verify with retries. The backend's endpoint reads
        text/url as Form(...) fields, so this always sends `data=`, never
        `json=` - a JSON body would silently fail to populate those fields."""
        url = f"{self._base_url}/verify"
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(url, data=payload, files=files)
                response.raise_for_status()
                return VerdictResponse.model_validate(response.json())
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning("backend /verify call failed (attempt %d): %s", attempt, exc)
                if attempt < self._max_retries:
                    time.sleep(min(self._retry_backoff_seconds * (2 ** (attempt - 1)), 10.0))

        raise VerifyRequestError(f"could not reach backend after {self._max_retries} attempts: {last_error}") from last_error


class PDFReportBuilder:
    """Renders a one-page PDF summary of a VerdictResponse using reportlab."""

    def build(self, verdict: VerdictResponse) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("VERITY Fact-Check Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(
            Paragraph(f"<b>Verdict:</b> {verdict.label} (confidence {verdict.confidence:.0%})", styles["Heading2"])
        )
        story.append(Spacer(1, 12))

        if verdict.per_claim:
            story.append(Paragraph("Per-claim results", styles["Heading3"]))
            table_data = [["Claim ID", "Label", "Supporting", "Refuting"]]
            for claim in verdict.per_claim:
                table_data.append([claim.claim_id, claim.label, str(claim.supporting), str(claim.refuting)])
            table = Table(table_data, hAlign="LEFT")
            table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), pdf_colors.HexColor("#333333")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), pdf_colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, pdf_colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ])
            )
            story.append(table)
            story.append(Spacer(1, 12))

        if verdict.evidence_citations:
            story.append(Paragraph("Evidence citations", styles["Heading3"]))
            for citation in verdict.evidence_citations:
                story.append(
                    Paragraph(
                        f"[{citation.claim_id}] {citation.snippet} "
                        f"&mdash; <i>{citation.source_name}</i> "
                        f"(tier {citation.source_tier}) &mdash; {citation.url}",
                        styles["BodyText"],
                    )
                )
            story.append(Spacer(1, 12))

        if verdict.caveats:
            story.append(Paragraph("Caveats", styles["Heading3"]))
            for caveat in verdict.caveats:
                story.append(Paragraph(f"&bull; {caveat}", styles["BodyText"]))

        doc.build(story)
        return buffer.getvalue()


def render_verdict_badge(verdict: VerdictResponse) -> None:
    color = VERDICT_COLORS.get(verdict.label, "#757575")
    st.markdown(
        f"""
        <div style="background-color:{color};padding:16px;border-radius:8px;
                    color:white;font-size:1.4rem;font-weight:600;text-align:center;">
            {verdict.label}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Confidence")
    st.progress(min(max(verdict.confidence, 0.0), 1.0))


def render_per_claim_sections(verdict: VerdictResponse) -> None:
    for claim in verdict.per_claim:
        citations = [c for c in verdict.evidence_citations if c.claim_id == claim.claim_id]
        with st.expander(f"Claim {claim.claim_id} — {claim.label}"):
            st.write(f"Supporting sources: {claim.supporting} | Refuting sources: {claim.refuting}")
            if not citations:
                st.info("No evidence citations available for this claim.")
            for citation in citations:
                st.markdown(
                    f"> {citation.snippet}\n\n"
                    f"**Source:** [{citation.source_name}]({citation.url}) (tier {citation.source_tier})"
                )


def render_photo_forensics_panel(verdict: VerdictResponse) -> None:
    st.subheader("Is this clipping genuine?")

    halftone = verdict.signal("halftone")
    masthead = verdict.signal("masthead_match")
    ela = verdict.signal("ela")

    cols = st.columns(2)
    with cols[0]:
        if halftone:
            st.metric("Halftone (print texture)", f"{halftone.score:.2f}")
            st.caption(halftone.note)
        else:
            st.info("Halftone check not available.")
    with cols[1]:
        if masthead:
            guess = (masthead.extras or {}).get("publication_guess", "unknown")
            st.metric("Masthead match", f"{masthead.score:.2f}")
            st.caption(f"Closest match: {guess}")
        else:
            st.info("Masthead check not available.")

    if ela:
        extras = ela.extras or {}
        heatmap_path = extras.get("heatmap_path") or extras.get("heatmap_url")
        st.caption(ela.note)
        if heatmap_path:
            st.image(heatmap_path, caption="Error Level Analysis heatmap")


def render_transparency_panel(verdict: VerdictResponse) -> None:
    st.subheader("What we checked")
    status_icons = {"ok": "✅", "skipped": "⏭️", "failed": "❌"}
    for check in verdict.checks_performed:
        icon = status_icons.get(check.status, "•")
        line = f"{icon} **{check.agent_name}** — {check.status} ({check.duration_ms} ms)"
        if check.note:
            line += f" — {check.note}"
        st.markdown(line)

    if verdict.caveats:
        st.subheader("Caveats")
        for caveat in verdict.caveats:
            st.warning(caveat)


def render_sidebar(verdict: Optional[VerdictResponse]) -> None:
    st.sidebar.title("VERITY")
    st.sidebar.caption("News & clipping fact-checker")
    if verdict is not None:
        device = verdict.device_used()
        if device:
            st.sidebar.metric("Local model device", device)
    st.sidebar.divider()
    st.sidebar.caption(f"Backend: {settings.backend_base_url}")


def render_result(verdict: VerdictResponse, is_image_input: bool, pdf_builder: PDFReportBuilder) -> None:
    render_verdict_badge(verdict)
    st.divider()
    render_per_claim_sections(verdict)

    if is_image_input:
        st.divider()
        render_photo_forensics_panel(verdict)

    st.divider()
    render_transparency_panel(verdict)

    st.divider()
    pdf_bytes = pdf_builder.build(verdict)
    st.download_button(label="Download report as PDF", data=pdf_bytes, file_name="verity_report.pdf", mime="application/pdf")


_PROGRESS_STAGES_TEXT = ["Reading input…", "Extracting claims…", "Gathering evidence…", "Building verdict…"]
_PROGRESS_STAGES_IMAGE = ["Reading image…", "Running forensics…", "Extracting claims…", "Gathering evidence…", "Building verdict…"]


def _run_with_progress(stages: list[str], call) -> Optional[VerdictResponse]:
    """Shows sequential stage messages while a single blocking backend call
    runs. The call is one request/response - these stages are a UX aid,
    not a stream of real backend events."""
    try:
        with st.status(stages[0], expanded=True) as status:
            for stage_label in stages[:-1]:
                status.update(label=stage_label)
                time.sleep(0.3)
            status.update(label=stages[-1])
            verdict = call()
            status.update(label="Done.", state="complete")
        return verdict
    except VerifyRequestError as exc:
        logger.warning("verification request failed: %s", exc)
        st.error("Couldn't reach the VERITY backend. Please check that the server is running and try again.")
        return None


def render_text_tab(client: VerifyClient, pdf_builder: PDFReportBuilder) -> None:
    text = st.text_area("Paste article text", height=220, key="text_input")
    if st.button("Check", key="check_text") and text.strip():
        verdict = _run_with_progress(_PROGRESS_STAGES_TEXT, lambda: client.verify_text(text))
        if verdict:
            render_result(verdict, is_image_input=False, pdf_builder=pdf_builder)


def render_url_tab(client: VerifyClient, pdf_builder: PDFReportBuilder) -> None:
    url = st.text_input("Paste article URL", key="url_input")
    if st.button("Check", key="check_url") and url.strip():
        verdict = _run_with_progress(_PROGRESS_STAGES_TEXT, lambda: client.verify_url(url))
        if verdict:
            render_result(verdict, is_image_input=False, pdf_builder=pdf_builder)


def render_photo_tab(client: VerifyClient, pdf_builder: PDFReportBuilder) -> None:
    uploaded_file = st.file_uploader("Upload a photo of a newspaper clipping", type=["png", "jpg", "jpeg"], key="photo_input")
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", width=300)
    if st.button("Check", key="check_photo") and uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        verdict = _run_with_progress(_PROGRESS_STAGES_IMAGE, lambda: client.verify_image(image_bytes, uploaded_file.name))
        if verdict:
            render_result(verdict, is_image_input=True, pdf_builder=pdf_builder)


def main() -> None:
    """Streamlit entry point. Run via `streamlit run ui/streamlit_app.py`."""
    st.set_page_config(page_title="VERITY", page_icon="🔎", layout="centered")

    client = VerifyClient()
    pdf_builder = PDFReportBuilder()

    render_sidebar(verdict=None)

    st.title("🔎 VERITY")
    st.caption("Paste text, paste a link, or upload a photo of a newspaper clipping.")

    tab_text, tab_url, tab_photo = st.tabs(["Paste text", "Paste URL", "Upload photo"])
    with tab_text:
        render_text_tab(client, pdf_builder)
    with tab_url:
        render_url_tab(client, pdf_builder)
    with tab_photo:
        render_photo_tab(client, pdf_builder)


if __name__ == "__main__":
    main()
