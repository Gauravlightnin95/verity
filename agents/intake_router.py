"""Intake router (owned by Member B).

Purpose: classify raw InputPayload as text/url/image, detect language, and
politely reject empty or non-news input before it enters the pipeline.
Input: InputPayload (core.schemas). Output: writes `language` and, on
rejection, `reject_reason` onto VerityState.
"""

from pathlib import Path

from langdetect import LangDetectException, detect

from core.schemas import InputPayload

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
_NOT_NEWS = {"hello", "hi", "hey", "test", "ok", "okay", "thanks", "thank you"}
MIN_TEXT_CHARS = 40  # anything shorter cannot be a checkable news claim


def route(payload: InputPayload) -> dict:
    try:
        return _route(payload)
    except Exception as exc:  # never crash - always degrade
        return {"reject_reason": f"router error, input skipped: {exc}"}


def _route(payload: InputPayload) -> dict:
    if payload.input_type == "image":
        path = Path(payload.image_path or "")
        if not path.exists() or path.suffix.lower() not in _IMAGE_EXTS:
            return {"reject_reason": "image file missing or unsupported format"}
        return {"reject_reason": None}

    if payload.input_type == "url":
        return {"reject_reason": None}

    raw = (payload.text or "").strip()
    if not raw:
        return {"reject_reason": "empty input - paste an article, a link, or a photo"}
    if raw.lower() in _NOT_NEWS or len(raw) < MIN_TEXT_CHARS:
        return {"reject_reason": "input too short to contain a checkable news claim"}

    try:
        language = detect(raw)
    except LangDetectException:
        language = "unknown"

    return {"language": language, "reject_reason": None}
