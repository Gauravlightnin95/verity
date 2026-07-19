"""End-to-end pipeline tests, fully mocked (zero API keys, zero real
agents). Proves the LangGraph wiring itself - fan-out/fan-in, the
mock/real resolver, the async dispatch helper, and rejection
short-circuiting - works correctly."""

from agents.graph import run
from core.schemas import InputPayload, VerdictLabel


async def test_text_path_runs_end_to_end():
    verdict = await run(InputPayload(input_type="text", text="The city approved a new metro line."))

    agent_names = {c.agent_name for c in verdict.checks_performed}
    assert {"router", "text_passthrough", "claim_extractor", "retrieval", "fusion", "judge"} <= agent_names
    assert "ocr" not in agent_names
    assert "forensics" not in agent_names
    assert verdict.per_claim  # mock_claims always yields c1, c2


async def test_url_path_runs_end_to_end():
    verdict = await run(InputPayload(input_type="url", url="https://example.com/article"))

    agent_names = {c.agent_name for c in verdict.checks_performed}
    assert "scraper" in agent_names
    assert "ocr" not in agent_names


async def test_image_path_runs_ocr_and_forensics():
    verdict = await run(InputPayload(input_type="image", image_path="fake/clipping.png"))

    agent_names = {c.agent_name for c in verdict.checks_performed}
    assert {"ocr", "forensics"} <= agent_names
    signal_names = {s.name for s in verdict.signals}
    assert {"halftone", "masthead_match"} <= signal_names


async def test_rejected_input_short_circuits_to_unverifiable():
    verdict = await run(InputPayload(input_type="text", text="   "))

    assert verdict.label == VerdictLabel.UNVERIFIABLE
    assert verdict.caveats
    agent_names = {c.agent_name for c in verdict.checks_performed}
    assert "claim_extractor" not in agent_names  # never reached


async def test_zero_api_keys_still_completes_via_judge_fallback(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "groq_api_key", None)

    verdict = await run(InputPayload(input_type="text", text="The city approved a new metro line."))

    judge_log = next(c for c in verdict.checks_performed if c.agent_name == "judge")
    assert judge_log.status == "failed"  # no real key -> falls back to the fusion verdict
    assert verdict.label is not None


async def test_checks_performed_covers_every_pipeline_stage():
    verdict = await run(InputPayload(input_type="text", text="The city approved a new metro line."))

    agent_names = [c.agent_name for c in verdict.checks_performed]
    for expected in ("stance", "credibility", "temporal", "ai_text", "fusion", "judge"):
        assert expected in agent_names
