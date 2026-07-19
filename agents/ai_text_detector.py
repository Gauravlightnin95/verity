"""AI-text detector (owned by Member D) - "the honesty module".

Estimates whether a piece of OCR'd/scraped article text was likely written
by an AI, using two kinds of clues:

1. Statistical clues: perplexity from a tiny local GPT-2 model (only
   computed when settings.detector_local is True - see the gating note
   below), and burstiness (sentence-length variety, always available, no
   model download needed).
2. A local classifier (optional): injected via the ModelRunner protocol
   (e.g. Member C's ml.openvino_runtime.ModelManager). Only used when
   settings.detector_local is True AND a model runner is supplied.

Score convention: this module's *internal* computations use a "1 = AI-
like" convention throughout (perplexity/burstiness/classifier scores are
all P(AI-generated)-shaped). The returned Signal.score is inverted once,
at the return boundary, to match core.fusion's project-wide convention
that every Signal score means "higher = more true/authentic/trustworthy"
- so a high Signal.score here means the text looks human-written.

Honesty rules (enforced in code, not just prompted):
1. Refuse to score texts under settings.detector_min_words (default 150).
2. This signal's weight is hard-capped at settings.fusion_weights.ai_text_max.
3. Signal.note always carries the disclaimer that this is a stylistic
   signal only, with known false-positive rates, and cannot prove
   authorship.
4. settings.detector_local=False (the default) means GPT-2 perplexity and
   the local classifier are never attempted - zero model downloads on a
   fresh install, matching "keep local inference bursty, lazy-load".
"""

from __future__ import annotations

import re
import statistics
import time
from typing import Optional, Protocol, runtime_checkable

from app.config import settings
from core.schemas import Signal
from utils.logging_conf import get_logger

logger = get_logger(__name__)

DETECTOR_DISCLAIMER = (
    "stylistic signal only - AI-text detectors have known false-positive "
    "rates and cannot prove authorship"
)

# Convenience alias, derived from settings (not a second source of truth) -
# some callers/tests want the ceiling value without constructing a detector.
DETECTOR_WEIGHT_CAP = settings.fusion_weights.ai_text_max


class AITextDetectionError(Exception):
    """Raised for unrecoverable detector failures (not for "text too short",
    which is a normal, expected outcome and returns a Signal rather than
    raising)."""


@runtime_checkable
class ModelRunner(Protocol):
    """Minimal contract for a local model runner (e.g. Member C's
    ml.openvino_runtime.ModelManager) used for the optional classifier
    component. This module never imports that class directly."""

    def predict(self, text: str) -> float:
        """Return P(text is AI-generated) in [0.0, 1.0]."""
        ...


# Perplexity: how "surprised" a language model is by the next word. Lower
# perplexity => more predictable text => more "AI-like". Coarse starting
# points, worth recalibrating against labeled human/AI text later.
_PERPLEXITY_LOW = 15.0
_PERPLEXITY_HIGH = 60.0

# Burstiness: coefficient of variation of sentence lengths. Lower => more
# uniform sentence lengths => more "AI-like".
_BURSTINESS_LOW = 0.15
_BURSTINESS_HIGH = 0.55

_PERPLEXITY_BLEND_WEIGHT = 0.5
_BURSTINESS_BLEND_WEIGHT = 0.5
_CLASSIFIER_BLEND_WEIGHT = 0.6
_STATISTICAL_BLEND_WEIGHT = 0.4

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _linear_score(value: float, low: float, high: float) -> float:
    """Maps `value` to a 0-1 "AI-likelihood" score. `low` saturates at 1.0
    (most AI-like), `high` saturates at 0.0 (most human-like)."""
    if low == high:
        return 0.5
    clamped = max(min(value, max(low, high)), min(low, high))
    fraction = (clamped - low) / (high - low)
    return max(0.0, min(1.0, 1.0 - fraction))


class _StatisticalAnalyzer:
    """Perplexity (tiny local GPT-2, lazy-loaded, only when asked - see
    detect_ai_text's gating) + burstiness (always available, pure stdlib)."""

    def __init__(self, max_retries: int = 2, retry_backoff_seconds: float = 1.5) -> None:
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._model = None
        self._tokenizer = None
        self._load_failed = False

    def _ensure_model_loaded(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_failed:
            return False

        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                # Imported lazily so the statistics-only path (the default)
                # never pays the torch/transformers import cost.
                from transformers import GPT2LMHeadModel, GPT2TokenizerFast

                logger.debug("loading local GPT-2 model for perplexity scoring")
                self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self._model = GPT2LMHeadModel.from_pretrained("gpt2")
                self._model.eval()
                return True
            except Exception as exc:
                last_error = exc
                logger.warning("failed to load local GPT-2 (attempt %d): %s", attempt, exc)
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff_seconds * attempt)

        logger.warning("giving up on GPT-2 perplexity after retries: %s", last_error)
        self._load_failed = True
        return False

    def compute_perplexity(self, text: str) -> Optional[float]:
        if not self._ensure_model_loaded():
            return None

        import torch

        encodings = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings["input_ids"]
        if input_ids.shape[1] < 2:
            return None

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            perplexity = float(torch.exp(outputs.loss))
        return perplexity

    @staticmethod
    def compute_burstiness(text: str) -> float:
        """Coefficient of variation of sentence lengths (words). Higher =
        more variety = more human-like."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        lengths = [len(s.split()) for s in sentences if s.split()]
        if len(lengths) < 2:
            return 0.0
        mean_len = statistics.mean(lengths)
        if mean_len == 0:
            return 0.0
        return statistics.pstdev(lengths) / mean_len


class AITextDetector:
    """Estimates an AI-likelihood `Signal` for a piece of text.

    Parameters
    ----------
    model_runner: Optional ModelRunner for the local classifier component.
        Ignored unless use_local_classifier is True.
    use_local_classifier: Mirrors settings.detector_local. When False (the
        default), GPT-2 perplexity and the classifier are both skipped -
        statistics-only, no model download.
    min_words: Texts with fewer words are refused. Defaults to
        settings.detector_min_words.
    weight_cap: Hard ceiling on the returned signal's weight. Defaults to
        (and is always clamped to) settings.fusion_weights.ai_text_max.
    """

    def __init__(
        self,
        model_runner: Optional[ModelRunner] = None,
        use_local_classifier: Optional[bool] = None,
        min_words: Optional[int] = None,
        weight_cap: Optional[float] = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._model_runner = model_runner
        self._use_local_classifier = (
            settings.detector_local if use_local_classifier is None else use_local_classifier
        )
        self._min_words = settings.detector_min_words if min_words is None else min_words
        cap = settings.fusion_weights.ai_text_max
        self._weight_cap = min(weight_cap, cap) if weight_cap is not None else cap
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._statistical_analyzer = _StatisticalAnalyzer()

    def detect_ai_text(self, text: str) -> Signal:
        word_count = len(text.split()) if text else 0

        if word_count < self._min_words:
            return self._refusal_signal(word_count)

        # Gated behind use_local_classifier: default (False) never attempts
        # a GPT-2 download, matching "keep local inference bursty/lazy".
        perplexity = (
            self._statistical_analyzer.compute_perplexity(text) if self._use_local_classifier else None
        )
        burstiness = self._statistical_analyzer.compute_burstiness(text)

        perplexity_score = (
            _linear_score(perplexity, _PERPLEXITY_LOW, _PERPLEXITY_HIGH) if perplexity is not None else None
        )
        burstiness_score = _linear_score(burstiness, _BURSTINESS_LOW, _BURSTINESS_HIGH)
        statistical_score = self._blend_statistical_scores(perplexity_score, burstiness_score)

        classifier_probability: Optional[float] = None
        if self._use_local_classifier and self._model_runner is not None:
            classifier_probability = self._invoke_classifier_with_retry(text)

        ai_likelihood, method = self._blend_final_score(statistical_score, classifier_probability)
        # Invert once here: internal convention is "1 = AI-like", but
        # core.fusion's project-wide convention is "higher = more
        # trustworthy/human-written" - see module docstring.
        trust_score = 1.0 - ai_likelihood

        note = f"{DETECTOR_DISCLAIMER} (method: {method})"
        extras = {
            "word_count": word_count,
            "perplexity": perplexity,
            "burstiness": round(burstiness, 4),
            "classifier_probability": classifier_probability,
            "ai_likelihood": round(ai_likelihood, 4),
            "method": method,
        }

        logger.info(
            "ai-text detection complete: trust_score=%.4f method=%s words=%d", trust_score, method, word_count
        )

        return Signal(name="ai_text", score=trust_score, weight=self._weight_cap, note=note, extras=extras)

    def _refusal_signal(self, word_count: int) -> Signal:
        logger.info("refusing to score text below minimum word count: %d < %d", word_count, self._min_words)
        note = (
            f"text has {word_count} words, below the minimum of {self._min_words} "
            f"required to score reliably. {DETECTOR_DISCLAIMER}"
        )
        return Signal(
            name="ai_text", score=0.5, weight=0.0, note=note,
            extras={"word_count": word_count, "refused": True},
        )

    @staticmethod
    def _blend_statistical_scores(perplexity_score: Optional[float], burstiness_score: float) -> float:
        if perplexity_score is None:
            return burstiness_score
        total_weight = _PERPLEXITY_BLEND_WEIGHT + _BURSTINESS_BLEND_WEIGHT
        return (
            perplexity_score * _PERPLEXITY_BLEND_WEIGHT + burstiness_score * _BURSTINESS_BLEND_WEIGHT
        ) / total_weight

    @staticmethod
    def _blend_final_score(
        statistical_score: float, classifier_probability: Optional[float]
    ) -> tuple[float, str]:
        if classifier_probability is None:
            return statistical_score, "statistical clues only"
        blended = classifier_probability * _CLASSIFIER_BLEND_WEIGHT + statistical_score * _STATISTICAL_BLEND_WEIGHT
        return max(0.0, min(1.0, blended)), "local classifier + statistical clues"

    def _invoke_classifier_with_retry(self, text: str) -> Optional[float]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                probability = self._model_runner.predict(text)
                return max(0.0, min(1.0, float(probability)))
            except Exception as exc:
                last_error = exc
                logger.warning("local classifier call failed (attempt %d): %s", attempt, exc)
                if attempt < self._max_retries:
                    time.sleep(min(self._retry_backoff_seconds * (2 ** (attempt - 1)), 15.0))

        logger.warning("local classifier failed after retries, using statistics only: %s", last_error)
        return None


_default_detector: Optional[AITextDetector] = None


def detect_ai_text(text: str) -> Signal:
    """Module-level entrypoint agents/graph.py calls: detect_ai_text(text) -> Signal."""
    global _default_detector
    if _default_detector is None:
        _default_detector = AITextDetector()
    return _default_detector.detect_ai_text(text)
