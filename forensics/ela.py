"""Error Level Analysis: detects edited regions via inconsistent JPEG
compression levels. Recompressing at a fixed quality and diffing against
the original makes a pasted-in region - previously compressed a different
number of times than the rest of the image - light up brighter than a
pristine region in the difference heatmap."""

import os
import tempfile

import cv2
import numpy as np

from core.schemas import Signal

# Empirical scale for normalizing the raw mean pixel-diff into 0-1; picked so
# neither the realistic nor the fabricated test fixture saturates at 1.0
# while a clear discrimination gap between them remains (see test_forensics.py).
_SCORE_SCALE = 25.0


def run_ela(image_path: str, quality: int = 90) -> Signal:
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    tmp_fd, temp_jpeg = tempfile.mkstemp(suffix=".jpg", prefix="verity_ela_")
    os.close(tmp_fd)
    try:
        cv2.imwrite(temp_jpeg, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imread(temp_jpeg)
    finally:
        os.remove(temp_jpeg)

    diff = cv2.absdiff(original, compressed)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    max_val = int(np.max(gray_diff)) or 1
    enhanced = np.uint8((gray_diff / max_val) * 255)
    heatmap = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)

    heatmap_fd, heatmap_path = tempfile.mkstemp(suffix=".png", prefix="verity_ela_heatmap_")
    os.close(heatmap_fd)
    cv2.imwrite(heatmap_path, heatmap)

    raw_mean_diff = float(np.mean(gray_diff))
    score = min(1.0, raw_mean_diff / _SCORE_SCALE)

    return Signal(
        name="ela",
        score=score,
        weight=0.0,  # forensics_agent assigns the shared image_forensics category weight
        note=f"mean pixel diff after recompression: {raw_mean_diff:.2f}",
        extras={"heatmap_path": heatmap_path, "max_diff": float(max_val), "raw_mean_diff": raw_mean_diff},
    )
