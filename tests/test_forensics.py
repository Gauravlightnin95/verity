"""Member C forensics tests: fabricated fixture must score worse than the
realistic one, and every emitted Signal.name must be a fusion-recognized
alias (folded in from the old tests/test_mocks.py, now against real code)."""

import os

from core.fusion import NAME_TO_CATEGORY
from forensics.copymove import run_copymove
from forensics.ela import run_ela
from forensics.halftone import run_halftone
from forensics.masthead import run_masthead

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "images")
REALISTIC = os.path.join(_FIXTURES, "realistic_newspaper.png")
FAKE = os.path.join(_FIXTURES, "fake_clipping.png")


def test_fixtures_exist():
    assert os.path.exists(REALISTIC), "realistic fixture missing"
    assert os.path.exists(FAKE), "fake fixture missing"


def test_halftone_discriminates_real_from_fake():
    realistic = run_halftone(REALISTIC)
    fake = run_halftone(FAKE)
    assert realistic.score > fake.score + 0.1, (
        f"realistic halftone ({realistic.score}) should score well above fake ({fake.score})"
    )
    assert realistic.extras["has_print_artifacts"] is True
    assert fake.extras["has_print_artifacts"] is False


def test_ela_discriminates_real_from_fake():
    realistic = run_ela(REALISTIC)
    fake = run_ela(FAKE)
    assert realistic.score > fake.score * 1.5, (
        f"realistic ELA score ({realistic.score}) should score well above fake ({fake.score})"
    )
    assert 0.0 <= realistic.score <= 1.0
    assert 0.0 <= fake.score <= 1.0


def test_copymove_runs_without_crashing_on_both_fixtures():
    for path in (REALISTIC, FAKE):
        signal = run_copymove(path)
        assert 0.0 <= signal.score <= 1.0


def test_masthead_handles_missing_ocr_text_without_crashing():
    # graph.py passes masthead_text=None when OCR found no masthead region.
    signal = run_masthead(REALISTIC, masthead_text=None)
    assert 0.0 <= signal.score <= 1.0
    assert "publication_guess" in signal.extras


def test_forensics_signal_names_match_fusion_aliases():
    from agents.forensics_agent import run_forensics

    signals = run_forensics(REALISTIC, masthead_text="The Daily Example")
    assert signals, "run_forensics produced nothing"
    for sig in signals:
        assert sig.name in NAME_TO_CATEGORY
        assert sig.weight > 0  # forensics_agent must have stamped a real weight
