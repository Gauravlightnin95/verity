"""Shared pytest fixtures. Kept env-independent: the suite must pass with
zero API keys and zero real agents wired in (all use_real_* flags off)."""

import json
from pathlib import Path
from typing import Optional

import pytest

CONTRACTS_DIR = Path(__file__).parent / "fixtures" / "contracts"


def load_contract(name: str):
    with open(CONTRACTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def contract():
    return load_contract


# ---------------------------------------------------------------- Member D
# contract fixtures + lightweight fakes satisfying agents/*.py Protocols


@pytest.fixture
def claims_sample_1() -> list[dict]:
    """A report-genre article with two claims (high/medium checkability)."""
    return load_contract("claims_sample_1.json")


@pytest.fixture
def claims_sample_2() -> list[dict]:
    """An opinion-genre article with three low-checkability claims."""
    return load_contract("claims_sample_2.json")


@pytest.fixture
def stance_results_sample_1() -> list[dict]:
    """Mixed SUPPORTS/REFUTES/NEUTRAL stances across four evidence items."""
    return load_contract("stance_results_sample_1.json")


@pytest.fixture
def stance_results_sample_2() -> list[dict]:
    """All-NEUTRAL stances, including one model-omitted result."""
    return load_contract("stance_results_sample_2.json")


@pytest.fixture
def ai_text_signal_sample_1() -> dict:
    """Refusal-path signal (text under the minimum word count)."""
    return load_contract("ai_text_signal_sample_1.json")


@pytest.fixture
def ai_text_signal_sample_2() -> dict:
    """Fully scored signal (classifier + statistical clues)."""
    return load_contract("ai_text_signal_sample_2.json")


class FakeArticle:
    """Satisfies agents.claim_extractor.ArticleLike."""

    def __init__(
        self,
        title: str = "City Approves New Metro Line",
        body: str = "The municipal corporation on Friday approved a new metro line.",
        author: Optional[str] = "Staff Reporter",
        domain: Optional[str] = "example-news.com",
    ) -> None:
        self.title = title
        self.body = body
        self.author = author
        self.domain = domain


class FakeClaim:
    """Satisfies agents.stance_agent.ClaimLike."""

    def __init__(self, claim_id: str = "c1", text: str = "The metro line was approved on 11 July 2026.") -> None:
        self.claim_id = claim_id
        self.text = text


class FakeEvidence:
    """Satisfies agents.stance_agent.EvidenceLike."""

    def __init__(self, evidence_id: str, snippet: str, source_name: Optional[str] = "Example Fact Check") -> None:
        self.evidence_id = evidence_id
        self.snippet = snippet
        self.source_name = source_name


@pytest.fixture
def fake_article() -> FakeArticle:
    return FakeArticle()


@pytest.fixture
def fake_claim() -> FakeClaim:
    return FakeClaim()


@pytest.fixture
def fake_evidence_list() -> list[FakeEvidence]:
    return [
        FakeEvidence("e1", "The ministry confirmed the metro line approval on Monday."),
        FakeEvidence("e2", "Officials denied any such approval took place."),
        FakeEvidence("e3", "The city council discussed public transport funding generally."),
    ]


@pytest.fixture(autouse=True)
def _dummy_groq_key(monkeypatch):
    # Force settings.groq_api_key to a fake value regardless of what's in
    # the developer's real .env - agents.judge_agent.judge() (exercised for
    # real by test_graph.py, never mocked) passes api_key=settings.
    # groq_api_key explicitly, so a real key in .env would otherwise make
    # test_graph.py hit the real Groq API on every run. The actual call
    # still fails (invalid key), which judge_agent's broad except Exception
    # catches and falls back on, same as a genuinely absent key would.
    from app.config import settings

    monkeypatch.setattr(settings, "groq_api_key", "test-key-not-real")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")


@pytest.fixture(autouse=True)
def _fake_graph_capabilities(monkeypatch):
    """Every real capability is wired directly into agents.graph now (no
    mock/toggle system) - patch them onto graph's module namespace with
    tests/graph_fakes.py so test_graph.py/test_api.py stay fast,
    deterministic, and require zero API keys/network. A no-op for tests
    that don't touch agents.graph."""
    import agents.graph as graph
    from tests import graph_fakes

    for name in (
        "route", "scrape", "retrieve", "score_credibility", "check_temporal",
        "run_ocr", "run_forensics", "extract_claims", "stance_batch", "detect_ai_text",
    ):
        monkeypatch.setattr(graph, name, getattr(graph_fakes, name))


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)
