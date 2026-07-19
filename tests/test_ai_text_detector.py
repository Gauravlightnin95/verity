"""
tests/test_ai_text_detector.py
=================================
Tests for `agents/ai_text_detector.py` (D.3), the "honesty module".

GPT-2 perplexity loading is monkeypatched off in every test here so these
tests are deterministic and fully offline regardless of whether the
environment has network access to download model weights — the detector
is designed to gracefully fall back to burstiness-only scoring in exactly
this situation (see `AITextDetector._blend_statistical_scores`), so this
also doubles as a test of that fallback path.
"""

from __future__ import annotations

import pytest

from agents.ai_text_detector import AITextDetector, DETECTOR_DISCLAIMER, DETECTOR_WEIGHT_CAP


LONG_TEXT = (
    "The municipal corporation approved a new metro line on Friday. "
    "Officials said construction would begin next month, pending final environmental clearance. "
    "Residents in the area have long asked for better public transport options. "
    "The project is expected to reduce commute times significantly for thousands of daily riders. "
) * 4  # comfortably over the 150-word minimum

SHORT_TEXT = "This article is far too short to score reliably."


@pytest.fixture(autouse=True)
def _disable_gpt2(monkeypatch):
    """Force every test to skip GPT-2 perplexity loading, so results are
    deterministic and independent of network access in this environment."""
    monkeypatch.setattr(
        "agents.ai_text_detector._StatisticalAnalyzer._ensure_model_loaded",
        lambda self: False,
    )


class TestRefusalRule:
    def test_refuses_text_under_minimum_words(self):
        detector = AITextDetector(min_words=150)
        signal = detector.detect_ai_text(SHORT_TEXT)

        assert signal.weight == 0.0
        assert signal.extras["refused"] is True
        assert "below the minimum of 150" in signal.note

    def test_refusal_still_includes_disclaimer(self):
        detector = AITextDetector(min_words=150)
        signal = detector.detect_ai_text(SHORT_TEXT)

        assert DETECTOR_DISCLAIMER in signal.note

    def test_matches_documented_fixture_shape(self, ai_text_signal_sample_1):
        detector = AITextDetector(min_words=150)
        signal = detector.detect_ai_text(SHORT_TEXT)

        assert signal.name == ai_text_signal_sample_1["name"]
        assert signal.weight == ai_text_signal_sample_1["weight"]
        assert signal.extras["refused"] == ai_text_signal_sample_1["extras"]["refused"]


class TestWeightCap:
    def test_weight_never_exceeds_hard_cap_even_if_requested_higher(self):
        detector = AITextDetector(weight_cap=0.99)  # attempt to override the cap
        signal = detector.detect_ai_text(LONG_TEXT)

        assert signal.weight == DETECTOR_WEIGHT_CAP
        assert signal.weight <= 0.15

    def test_scored_signal_includes_disclaimer(self):
        detector = AITextDetector()
        signal = detector.detect_ai_text(LONG_TEXT)

        assert DETECTOR_DISCLAIMER in signal.note


class TestStatisticalOnlyMode:
    def test_detector_local_false_never_touches_classifier(self):
        calls = []

        class TrackingModelRunner:
            def predict(self, text: str) -> float:
                calls.append(text)
                return 0.9

        detector = AITextDetector(model_runner=TrackingModelRunner(), use_local_classifier=False)
        signal = detector.detect_ai_text(LONG_TEXT)

        assert calls == []  # classifier never invoked
        assert signal.extras["method"] == "statistical clues only"
        assert signal.extras["classifier_probability"] is None

    def test_score_is_valid_probability(self):
        detector = AITextDetector()
        signal = detector.detect_ai_text(LONG_TEXT)

        assert 0.0 <= signal.score <= 1.0


class TestClassifierBlending:
    def test_classifier_result_is_blended_in(self):
        class FakeModelRunner:
            def predict(self, text: str) -> float:
                return 0.9

        detector = AITextDetector(model_runner=FakeModelRunner(), use_local_classifier=True)
        signal = detector.detect_ai_text(LONG_TEXT)

        assert signal.extras["method"] == "local classifier + statistical clues"
        assert signal.extras["classifier_probability"] == 0.9
        assert signal.weight == DETECTOR_WEIGHT_CAP

    def test_matches_documented_fixture_shape(self, ai_text_signal_sample_2):
        class FakeModelRunner:
            def predict(self, text: str) -> float:
                return 0.63

        detector = AITextDetector(model_runner=FakeModelRunner(), use_local_classifier=True)
        signal = detector.detect_ai_text(LONG_TEXT)

        assert signal.name == ai_text_signal_sample_2["name"]
        assert signal.weight == ai_text_signal_sample_2["weight"]
        assert signal.extras["method"] == ai_text_signal_sample_2["extras"]["method"]

    def test_broken_classifier_falls_back_to_statistical_only(self, monkeypatch):
        monkeypatch.setattr("agents.ai_text_detector.time.sleep", lambda _seconds: None)

        class BrokenModelRunner:
            def predict(self, text: str) -> float:
                raise RuntimeError("model not loaded")

        detector = AITextDetector(
            model_runner=BrokenModelRunner(),
            use_local_classifier=True,
            max_retries=2,
            retry_backoff_seconds=0.01,
        )
        signal = detector.detect_ai_text(LONG_TEXT)

        assert signal.extras["classifier_probability"] is None
        assert signal.extras["method"] == "statistical clues only"
        # Detector must still return a usable signal, never raise.
        assert 0.0 <= signal.score <= 1.0
