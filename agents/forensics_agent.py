"""Image forensics agent (owned by Member C).

Purpose: run the forensics/ package's four checks (ELA, copy-move,
halftone, masthead match) against a clipping image and return one Signal
per check. Emits Signal.name values "ela", "copymove", "halftone",
"masthead_match" (aliased to fusion categories "image_forensics" and
"masthead" by core.fusion.NAME_TO_CATEGORY) plus a `publication_guess`
string (masthead match) that VerityState exposes to the retrieval agent.
Input: image_path (str), masthead_text (str | None).
Output: list[Signal] (core.schemas). CPU/OpenCV only, no heavy models.
"""

import logging

from app.config import settings
from core.schemas import Signal
from forensics.copymove import run_copymove
from forensics.ela import run_ela
from forensics.halftone import run_halftone
from forensics.masthead import run_masthead

logger = logging.getLogger("verity.forensics")

# Each per-check module returns weight=0.0 (it doesn't know the fusion
# table) - this is the one place that reads app.config.settings and stamps
# the real category weight onto each signal before it re-enters the pipeline.
_IMAGE_FORENSICS_SIGNALS = {"ela", "copymove", "halftone"}
_MASTHEAD_SIGNALS = {"masthead_match"}


def run_forensics(image_path: str, masthead_text: str | None = None) -> list[Signal]:
    signals: list[Signal] = []

    for name, check in (
        ("ela", lambda: run_ela(image_path)),
        ("copymove", lambda: run_copymove(image_path)),
        ("halftone", lambda: run_halftone(image_path)),
        ("masthead", lambda: run_masthead(image_path, masthead_text)),
    ):
        try:
            signal = check()
            if signal.name in _IMAGE_FORENSICS_SIGNALS:
                signal.weight = settings.fusion_weights.image_forensics
            elif signal.name in _MASTHEAD_SIGNALS:
                signal.weight = settings.fusion_weights.masthead
            signals.append(signal)
        except Exception as exc:
            logger.error("%s check failed: %s", name, exc)

    return signals
