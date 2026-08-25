"""LangGraph wiring: router -> (scraper | text passthrough | ocr -> forensics)
-> claim_extractor -> retrieval -> parallel(stance, credibility, temporal,
ai_text) -> fusion -> judge.

Every node is a thin async wrapper: call a capability -> record a CheckLog
-> write the result back into state. A node's own try/except means one
failing capability degrades the pipeline instead of crashing it.

Two nodes can end the run early, both through the same reject_reason ->
END short-circuit: the router (bad input) and claim_extractor (extraction
failed, or nothing survived the check-worthiness filter).

Every node is async, and every capability call goes through one shared
dispatch helper (_timed_call / _call_maybe_async) that awaits async
functions directly and runs sync functions in a worker thread
(asyncio.to_thread) so a slow real call (OCR, an LLM round-trip, a network
request) never blocks the event loop. This uniform rule exists because
retrieve() is async while every other capability (and judge()) is sync -
rather than special-casing which nodes get offloaded, every capability
goes through the same dispatch path.

Two state representations are used deliberately: core.schemas.VerityState
is the framework-agnostic contract everyone else in the project imports;
this file's private _GraphState TypedDict is LangGraph-specific (it adds
Annotated[..., operator.add] reducers on the two fields multiple parallel
nodes write to concurrently - signals and checks_performed) and never
leaks outside this module.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.ai_text_detector import detect_ai_text
from agents.claim_extractor import extract_claims
from agents.credibility_agent import score_credibility
from agents.forensics_agent import run_forensics
from agents.intake_router import route
from agents.judge_agent import judge
from agents.ocr_agent import run_ocr
from agents.retrieval_agent import FACTCHECK_TOOL, retrieve
from agents.scraper_agent import scrape
from agents.swytchcode_client import available as swytchcode_available
from agents.stance_agent import stance_batch
from agents.temporal_agent import check_temporal
from core.fusion import fuse
from core.schemas import (
    ArticleContent,
    CheckLog,
    Claim,
    EvidenceItem,
    InputPayload,
    OCRResult,
    Signal,
    StanceResult,
    Verdict,
    VerdictLabel,
)


# Set as reject_reason when extraction succeeded but nothing survived the
# check-worthiness filter - a legitimate outcome, not a malfunction. run()
# turns any reject_reason into an UNVERIFIABLE verdict carrying it as a
# caveat, so this string is what the user reads.
NO_CHECKWORTHY_CLAIMS_MESSAGE = "no factual claims found in the article"


async def _call_maybe_async(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)


async def _timed_call(agent_name: str, fn, *args, **kwargs):
    t0 = time.monotonic()
    try:
        result = await _call_maybe_async(fn, *args, **kwargs)
        duration_ms = int((time.monotonic() - t0) * 1000)
        return result, CheckLog(agent_name=agent_name, status="ok", duration_ms=duration_ms)
    except Exception as exc:
        duration_ms = int((time.monotonic() - t0) * 1000)
        return None, CheckLog(agent_name=agent_name, status="failed", duration_ms=duration_ms, note=str(exc))


class _GraphState(TypedDict, total=False):
    input_payload: InputPayload
    article: ArticleContent | None
    ocr_result: OCRResult | None
    publication_guess: str | None
    language: str | None
    reject_reason: str | None
    claims: list[Claim]
    evidence: list[EvidenceItem]
    stance_results: list[StanceResult]
    signals: Annotated[list[Signal], operator.add]
    checks_performed: Annotated[list[CheckLog], operator.add]
    verdict: Verdict | None


async def _node_router(state: _GraphState) -> dict:
    result, log = await _timed_call("router", route, state["input_payload"])
    updates: dict = {"checks_performed": [log]}
    if result:
        updates.update(result)
    return updates


async def _node_text_passthrough(state: _GraphState) -> dict:
    t0 = time.monotonic()
    article = ArticleContent(body=state["input_payload"].text or "", domain="user-submitted")
    duration_ms = int((time.monotonic() - t0) * 1000)
    log = CheckLog(agent_name="text_passthrough", status="ok", duration_ms=duration_ms)
    return {"article": article, "checks_performed": [log]}


async def _node_scraper(state: _GraphState) -> dict:
    result, log = await _timed_call("scraper", scrape, state["input_payload"].url)
    updates: dict = {"checks_performed": [log]}
    if result is not None:
        updates["article"] = result
    return updates


async def _node_ocr(state: _GraphState) -> dict:
    result, log = await _timed_call("ocr", run_ocr, state["input_payload"].image_path)
    updates: dict = {"checks_performed": [log]}
    if result is not None:
        updates["ocr_result"] = result
    return updates


async def _node_forensics(state: _GraphState) -> dict:
    ocr_result = state.get("ocr_result")
    masthead_text = None
    if ocr_result:
        masthead = next((r for r in ocr_result.regions if r.region_type == "masthead"), None)
        masthead_text = masthead.text if masthead else None
    result, log = await _timed_call("forensics", run_forensics, state["input_payload"].image_path, masthead_text)
    updates: dict = {"checks_performed": [log]}
    if result is not None:
        updates["signals"] = result
        guess = next(
            (s.extras["publication_guess"] for s in result if s.extras and "publication_guess" in s.extras),
            None,
        )
        if guess:
            updates["publication_guess"] = guess
    return updates


async def _node_claim_extractor(state: _GraphState) -> dict:
    article = state.get("article")
    ocr_result = state.get("ocr_result")
    if article is None and ocr_result is not None:
        body = next((r for r in ocr_result.regions if r.region_type == "body"), None)
        headline = next((r for r in ocr_result.regions if r.region_type == "headline"), None)
        article = ArticleContent(
            title=headline.text if headline else None,
            body=body.text if body else ocr_result.full_text,
            domain=state.get("publication_guess") or "",
        )
    result, log = await _timed_call("claim_extractor", extract_claims, article)
    updates: dict = {"checks_performed": [log], "article": article}
    # _timed_call swallows exceptions into `log` and returns None, so
    # without a reject_reason the run would carry on to retrieval and
    # fusion with an empty claim list. Both terminating cases below reuse
    # the reject_reason -> END short-circuit, but stay distinct: reporting
    # "no factual claims" for a broken classifier would point the user at
    # the article instead of at the install.
    if result is None:
        updates["reject_reason"] = log.note or "claim extraction failed"
    else:
        updates["claims"] = result
        if not result:
            updates["reject_reason"] = NO_CHECKWORTHY_CLAIMS_MESSAGE
    return updates


async def _node_retrieval(state: _GraphState) -> dict:
    result, log = await _timed_call("retrieval", retrieve, state.get("claims", []), state.get("publication_guess"))
    # A second, purely informational row so the UI's transparency panel shows
    # whether the Swytchcode execution kernel was actually available for the
    # providers that route through it - "which checks ran" has to include
    # "and which execution layer ran them" to be honest.
    kernel_ok = swytchcode_available()
    kernel_log = CheckLog(
        agent_name="swytchcode_kernel",
        status="ok" if kernel_ok else "skipped",
        note=(
            f"evidence providers executed via Swytchcode ({FACTCHECK_TOOL})"
            if kernel_ok
            else "kernel unavailable (CLI or .swytchcode/tooling.json missing) - "
                 "those providers skipped, keyless search unaffected"
        ),
    )
    updates: dict = {"checks_performed": [log, kernel_log]}
    if result is not None:
        updates["evidence"] = result
    return updates


async def _node_stance(state: _GraphState) -> dict:
    claims = state.get("claims", [])
    evidence = state.get("evidence", [])
    # One batched LLM call for all claims (not one per claim) - each claim
    # paired with only its own evidence.
    pairs = [(c, [e for e in evidence if e.claim_id == c.claim_id]) for c in claims]

    result, log = await _timed_call("stance", stance_batch, pairs)
    return {"stance_results": result or [], "checks_performed": [log]}


async def _node_credibility(state: _GraphState) -> dict:
    result, log = await _timed_call("credibility", score_credibility, state.get("evidence", []))
    updates: dict = {"checks_performed": [log]}
    if result is not None:
        updates["signals"] = [result]
    return updates


async def _node_temporal(state: _GraphState) -> dict:
    result, log = await _timed_call("temporal", check_temporal, state.get("claims", []), state.get("evidence", []))
    updates: dict = {"checks_performed": [log]}
    if result is not None:
        updates["signals"] = [result]
    return updates


async def _node_ai_text(state: _GraphState) -> dict:
    article = state.get("article")
    ocr_result = state.get("ocr_result")
    text = article.body if article else (ocr_result.full_text if ocr_result else "")
    result, log = await _timed_call("ai_text", detect_ai_text, text)
    updates: dict = {"checks_performed": [log]}
    if result is not None:
        updates["signals"] = [result]
    return updates


async def _node_fusion(state: _GraphState) -> dict:
    t0 = time.monotonic()
    verdict = fuse(
        claims=state.get("claims", []),
        evidence=state.get("evidence", []),
        stance_results=state.get("stance_results", []),
        signals=state.get("signals", []),
        checks_performed=state.get("checks_performed", []),
    )
    log = CheckLog(agent_name="fusion", status="ok", duration_ms=int((time.monotonic() - t0) * 1000))
    return {"verdict": verdict, "checks_performed": [log]}


async def _node_judge(state: _GraphState) -> dict:
    fusion_verdict = state["verdict"]
    final = await asyncio.to_thread(
        judge, fusion_verdict, state.get("evidence", []), state.get("stance_results", [])
    )
    # judge() appends its own CheckLog onto a snapshot of checks_performed
    # taken before fusion's log was merged into graph state - reconcile
    # against state["checks_performed"], the graph's authoritative,
    # fully-reduced log, so the returned Verdict's transparency trail is
    # complete rather than one step behind.
    judge_log = final.checks_performed[-1]
    final.checks_performed = [*state.get("checks_performed", []), judge_log]
    return {"verdict": final, "checks_performed": [judge_log]}


def _route_branch(state: _GraphState) -> str:
    if state.get("reject_reason"):
        return "rejected"
    return state["input_payload"].input_type  # "text" | "url" | "image"


def _route_after_claims(state: _GraphState) -> str:
    if state.get("reject_reason"):
        return "rejected"
    return "retrieval"


def build_graph():
    graph = StateGraph(_GraphState)
    graph.add_node("router", _node_router)
    graph.add_node("text_passthrough", _node_text_passthrough)
    graph.add_node("scraper", _node_scraper)
    graph.add_node("ocr", _node_ocr)
    graph.add_node("forensics", _node_forensics)
    graph.add_node("claim_extractor", _node_claim_extractor)
    graph.add_node("retrieval", _node_retrieval)
    graph.add_node("stance", _node_stance)
    graph.add_node("credibility", _node_credibility)
    graph.add_node("temporal", _node_temporal)
    graph.add_node("ai_text", _node_ai_text)
    graph.add_node("fusion", _node_fusion)
    graph.add_node("judge", _node_judge)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_branch,
        {"rejected": END, "text": "text_passthrough", "url": "scraper", "image": "ocr"},
    )
    graph.add_edge("text_passthrough", "claim_extractor")
    graph.add_edge("scraper", "claim_extractor")
    graph.add_edge("ocr", "forensics")
    graph.add_edge("forensics", "claim_extractor")
    graph.add_conditional_edges(
        "claim_extractor",
        _route_after_claims,
        {"rejected": END, "retrieval": "retrieval"},
    )
    for analysis_node in ("stance", "credibility", "temporal", "ai_text"):
        graph.add_edge("retrieval", analysis_node)
        graph.add_edge(analysis_node, "fusion")
    graph.add_edge("fusion", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run(input_payload: InputPayload) -> Verdict:
    """Single public entrypoint: runs the full pipeline for one input and
    always returns a Verdict, even when the router rejects the input."""
    result = await get_graph().ainvoke(
        {
            "input_payload": input_payload,
            "claims": [],
            "evidence": [],
            "stance_results": [],
            "signals": [],
            "checks_performed": [],
        }
    )
    verdict = result.get("verdict")
    if verdict is not None:
        return verdict
    return Verdict(
        label=VerdictLabel.UNVERIFIABLE,
        confidence=0.0,
        per_claim=[],
        evidence_citations=[],
        signals=[],
        caveats=[result.get("reject_reason") or "input rejected"],
        checks_performed=result.get("checks_performed", []),
    )
