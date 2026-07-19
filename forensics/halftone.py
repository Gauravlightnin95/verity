"""Print-artifact (halftone) detection: genuine offset-printed newsprint
shows periodic ink-dot patterns; a digitally fabricated clipping usually
doesn't. Detected via 2D FFT peak strength on background image patches."""

import cv2
import numpy as np

from core.schemas import Signal


def get_background_patches(image, patch_size=128, num_patches=5):
    """Sample low-but-nonzero-variance patches (background, not text/edges)."""
    h, w = image.shape
    patches, variances = [], []

    rng = np.random.RandomState(42)
    for _ in range(50):
        if h <= patch_size or w <= patch_size:
            break
        y = rng.randint(0, h - patch_size)
        x = rng.randint(0, w - patch_size)
        patch = image[y : y + patch_size, x : x + patch_size]
        var = np.var(patch)
        if var > 5:  # skip completely flat regions
            patches.append(patch)
            variances.append(var)

    if not patches:
        y, x = h // 2 - patch_size // 2, w // 2 - patch_size // 2
        if y < 0 or x < 0:
            return [image]
        return [image[y : y + patch_size, x : x + patch_size]]

    order = np.argsort(variances)
    return [patches[i] for i in order[:num_patches]]


def check_halftone_fft(patch):
    """2D FFT peak-to-mean ratio: regular dot grids produce sharp peaks."""
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    magnitude_spectrum[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0  # remove DC component

    mean_mag = np.mean(magnitude_spectrum)
    max_mag = np.max(magnitude_spectrum)
    return max_mag / (mean_mag + 1e-8)


def run_halftone(image_path: str) -> Signal:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    patches = get_background_patches(image)
    ratios = [check_halftone_fft(p) for p in patches]
    avg_ratio = float(np.mean(ratios)) if ratios else 0.0

    # Empirical threshold: real newsprint's dot grid produces a high ratio,
    # smooth digital renders don't.
    has_print_artifacts = avg_ratio > 1.5
    score = min(1.0, avg_ratio / 3.0)

    note = (
        "found strong repeating FFT peaks - consistent with print halftone"
        if has_print_artifacts
        else "no halftone pattern detected - image likely made digitally"
    )

    return Signal(
        name="halftone", score=score, weight=0.0,
        note=note,
        extras={"has_print_artifacts": has_print_artifacts, "avg_fft_peak_ratio": avg_ratio},
    )
