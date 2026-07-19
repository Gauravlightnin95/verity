"""OCR agent (owned by Member C).

Purpose: read text out of a photographed newspaper clipping, via RapidOCR
(ONNX Runtime), after OpenCV preprocessing (perspective correction,
adaptive threshold), and group detected boxes into masthead/dateline/
headline/body regions.
Input: image_path (str). Output: OCRResult (core.schemas). Must still run
on CPU-only machines if no NPU/GPU is present.

Hardware note: this runs on RapidOCR's default onnxruntime CPU execution
provider. Real NPU/iGPU acceleration would need `onnxruntime-openvino`,
which installs a conflicting build of the `onnxruntime` import namespace
that ChromaDB (evidence caching) already depends on - see README's
hardware section for why that's deferred rather than attempted here.
"""

import os
import tempfile
import time

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from core.schemas import OCRRegion, OCRResult

# See forensics/__init__.py - same OpenCL teardown-crash avoidance,
# repeated here so OCR is safe even when imported standalone.
cv2.ocl.setUseOpenCL(False)

_DEVICE_USED = "CPU"  # onnxruntime's CPUExecutionProvider only - see module docstring

_engine: RapidOCR | None = None


def _get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sorts 4 quad points into [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_correction(image, gray):
    """Finds a 4-point page contour and flattens it, like a scanner app."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        return image, gray

    rect = order_points(screen_cnt.reshape(4, 2))
    (tl, tr, br, bl) = rect

    width = max(
        int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))),
        int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))),
    )
    height = max(
        int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))),
        int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))),
    )
    if width <= 0 or height <= 0:
        return image, gray

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (width, height)), cv2.warpPerspective(gray, m, (width, height))


def _quad_to_bbox(box: list[list[float]]) -> tuple[int, int, int, int]:
    """Axis-aligned bbox (xmin,ymin,xmax,ymax) from RapidOCR's 4-point quad."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def group_text_regions(ocr_result: list) -> list[OCRRegion]:
    """ocr_result: list of (box, text, score) from RapidOCR. Groups boxes into
    newspaper regions by vertical position and text height."""
    if not ocr_result:
        return []

    sorted_blocks = sorted(ocr_result, key=lambda b: min(p[1] for p in b[0]))
    full_height = max(max(p[1] for p in b[0]) for b in ocr_result)

    regions = []
    for box, text, score in sorted_blocks:
        ys = [p[1] for p in box]
        min_y = min(ys)
        height = max(ys) - min_y

        region_type = "body"
        if min_y < full_height * 0.1 and height > 30:
            region_type = "masthead"
        elif min_y < full_height * 0.15 and "20" in text:  # naive year-in-dateline check
            region_type = "dateline"
        elif height > 40:
            region_type = "headline"

        regions.append(
            OCRRegion(
                region_type=region_type,
                text=text,
                confidence=float(score),
                bbox=_quad_to_bbox(box),
            )
        )
    return regions


def run_ocr(image_path: str) -> OCRResult:
    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image, gray = perspective_correction(image, gray)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    tmp_fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="verity_ocr_")
    os.close(tmp_fd)
    try:
        cv2.imwrite(temp_path, thresh)
        ocr_res, _elapse = _get_engine()(temp_path)
    finally:
        os.remove(temp_path)

    regions = group_text_regions(ocr_res or [])
    full_text = "\n".join(r.text for r in regions)
    duration_ms = int((time.time() - start_time) * 1000)

    return OCRResult(
        regions=regions,
        full_text=full_text,
        language="en",
        device_used=_DEVICE_USED,
        duration_ms=duration_ms,
    )
