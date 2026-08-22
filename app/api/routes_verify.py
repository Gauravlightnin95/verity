"""POST /verify: runs the full VERITY pipeline for text, url, or image
input and returns a Verdict. POST /report renders a Verdict the caller
already holds into a PDF. GET /health for a trivial liveness check."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from agents.graph import run
from app.report import build_pdf
from core.schemas import InputPayload, Verdict

router = APIRouter()


@router.post("/verify", response_model=Verdict)
async def verify(
    text: str | None = Form(None),
    url: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> Verdict:
    if image is not None:
        suffix = Path(image.filename or "upload").suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image.read())
            image_path = tmp.name
        payload = InputPayload(input_type="image", image_path=image_path)
    elif url:
        payload = InputPayload(input_type="url", url=url)
    elif text:
        payload = InputPayload(input_type="text", text=text)
    else:
        raise HTTPException(status_code=400, detail="Provide one of: text, url, image")

    return await run(payload)


@router.post("/report")
async def report(verdict: Verdict) -> Response:
    """Render a Verdict the caller already has into a PDF.

    The web UI is a static page, so it cannot run ReportLab itself. It posts
    back the exact verdict it is displaying rather than re-running the
    pipeline - a second run would spend LLM budget and could legitimately
    return a different verdict than the one on screen.
    """
    return Response(
        content=build_pdf(verdict),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="verity_report.pdf"'},
    )
