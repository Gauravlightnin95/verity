"""Masthead/typography forensics: compares the top-of-page masthead crop
against a small registry of known publication logos via perceptual hash +
SSIM. The best match's name also becomes `publication_guess`, which flows
through VerityState to the retrieval agent's archive/e-paper probe."""

import os

import cv2
import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from core.schemas import Signal

# Placeholder reference logos (this repo has no real scraped mastheads yet -
# see the README's hardware/scope notes). Stored as a real data asset, not a
# test fixture, and generated once at import time rather than per-call.
_REFERENCE_DIR = os.path.join("data", "mastheads")
REFERENCE_LOGOS = {
    "The Hindu": os.path.join(_REFERENCE_DIR, "the_hindu.png"),
    "Times of India": os.path.join(_REFERENCE_DIR, "times_of_india.png"),
}


def _ensure_reference_logos() -> None:
    os.makedirs(_REFERENCE_DIR, exist_ok=True)
    for name, path in REFERENCE_LOGOS.items():
        if not os.path.exists(path):
            img = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(img, name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imwrite(path, img)


_ensure_reference_logos()  # runs once per process, not once per run_masthead() call


def run_masthead(image_path: str, masthead_text: str | None = None) -> Signal:
    """Compares the top 15% of the image (a crude masthead crop, since a
    full pipeline would crop using the OCR agent's masthead bbox instead)
    against each reference logo using average-hash + SSIM."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    masthead_crop = img[0 : int(h * 0.15), 0:w]

    pil_img = Image.fromarray(cv2.cvtColor(masthead_crop, cv2.COLOR_BGR2RGB))
    query_hash = imagehash.average_hash(pil_img)
    gray_crop = cv2.cvtColor(masthead_crop, cv2.COLOR_BGR2GRAY)

    masthead_text_lower = (masthead_text or "").lower()
    best_match, best_score = None, 0.0

    for pub_name, ref_path in REFERENCE_LOGOS.items():
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            continue

        ref_hash = imagehash.average_hash(Image.open(ref_path))
        hash_score = max(0.0, 1.0 - (query_hash - ref_hash) / 64.0)  # 64 = max avg-hash diff

        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        ref_gray_resized = cv2.resize(ref_gray, (gray_crop.shape[1], gray_crop.shape[0]))
        ssim_score, _ = ssim(gray_crop, ref_gray_resized, full=True)

        combined_score = (hash_score * 0.4) + (ssim_score * 0.6)
        if pub_name.lower() in masthead_text_lower:
            combined_score = min(1.0, combined_score + 0.3)  # OCR text corroborates the visual match

        if combined_score > best_score:
            best_score = combined_score
            best_match = pub_name

    publication_guess = best_match if best_score > 0.4 else "Unknown"

    return Signal(
        name="masthead_match",
        score=best_score,
        weight=0.0,  # forensics_agent assigns the shared masthead category weight
        note=(
            f"masthead matched '{publication_guess}', similarity {best_score:.2f}"
            if publication_guess != "Unknown"
            else "no matching masthead found"
        ),
        extras={"publication_guess": publication_guess, "ssim_hash_score": float(best_score)},
    )
