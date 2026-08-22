"""PDF report rendering.

Purpose: turn a finished Verdict into a one-page, citation-carrying PDF the
user can keep or forward. Lives here rather than in the UI layer because the
web frontend is static - it cannot run ReportLab, so it POSTs the verdict it
already holds back to /report and gets bytes.

Input: a core.schemas.Verdict.
Output: PDF file contents as bytes.

The layout mirrors the transparency rules the rest of the pipeline follows:
the verdict never appears without the evidence it was allowed to rely on, and
the caveats travel with it rather than being dropped on export.
"""

from __future__ import annotations

import io

from reportlab.lib import colors as pdf_colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.schemas import Verdict

# ReportLab's Paragraph parses a small HTML dialect, so any text taken from a
# scraped page has to be escaped or a stray "<" aborts the whole build.
_ESCAPES = (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"))

_TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), pdf_colors.HexColor("#333333")),
    ("TEXTCOLOR", (0, 0), (-1, 0), pdf_colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, pdf_colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
])


def _esc(text: object) -> str:
    out = str(text if text is not None else "")
    for raw, escaped in _ESCAPES:
        out = out.replace(raw, escaped)
    return out


def _table(rows: list[list[str]]) -> Table:
    table = Table(rows, hAlign="LEFT")
    table.setStyle(_TABLE_STYLE)
    return table


def build_pdf(verdict: Verdict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, title="VERITY Fact-Check Report")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("VERITY Fact-Check Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        f"<b>Verdict:</b> {_esc(verdict.label.value)} (confidence {verdict.confidence:.0%})",
        styles["Heading2"],
    ))
    story.append(Spacer(1, 12))

    if verdict.per_claim:
        story.append(Paragraph("Per-claim results", styles["Heading3"]))
        rows = [["Claim ID", "Label", "Supporting", "Refuting"]]
        for claim in verdict.per_claim:
            rows.append([claim.claim_id, claim.label.value, str(claim.supporting), str(claim.refuting)])
        story.append(_table(rows))
        story.append(Spacer(1, 12))

    if verdict.signals:
        story.append(Paragraph("Signals", styles["Heading3"]))
        rows = [["Signal", "Score", "Weight"]]
        for signal in verdict.signals:
            rows.append([signal.name, f"{signal.score:.2f}", f"{signal.weight:.2f}"])
        story.append(_table(rows))
        story.append(Spacer(1, 12))

    if verdict.evidence_citations:
        story.append(Paragraph("Evidence citations", styles["Heading3"]))
        for citation in verdict.evidence_citations:
            story.append(Paragraph(
                f"[{_esc(citation.claim_id)}] {_esc(citation.snippet)} "
                f"&mdash; <i>{_esc(citation.source_name)}</i> "
                f"(tier {citation.source_tier}) &mdash; {_esc(citation.url)}",
                styles["BodyText"],
            ))
        story.append(Spacer(1, 12))

    if verdict.caveats:
        story.append(Paragraph("Caveats", styles["Heading3"]))
        for caveat in verdict.caveats:
            story.append(Paragraph(f"&bull; {_esc(caveat)}", styles["BodyText"]))
        story.append(Spacer(1, 12))

    if verdict.checks_performed:
        story.append(Paragraph("What we checked", styles["Heading3"]))
        for check in verdict.checks_performed:
            note = f" &mdash; {_esc(check.note)}" if check.note else ""
            story.append(Paragraph(
                f"{_esc(check.agent_name)}: {check.status} ({check.duration_ms} ms){note}",
                styles["BodyText"],
            ))

    doc.build(story)
    return buffer.getvalue()
