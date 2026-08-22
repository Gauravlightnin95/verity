"""Member B tests (router, scraper, retrieval, credibility, temporal).
All internet calls intercepted by respx -> fast, free, offline."""

import httpx
import pytest
import respx
from swytchcode_runtime import SwytchcodeError

from agents import credibility_agent, intake_router, retrieval_agent, scraper_agent, temporal_agent
from core.schemas import Claim, EvidenceItem, InputPayload


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setattr(retrieval_agent, "_cache_get", lambda t: [])
    monkeypatch.setattr(retrieval_agent, "_cache_put", lambda t, i: None)
    # DuckDuckGo isn't an httpx call respx can intercept, so stub it out -
    # these tests exercise the keyed HTTP providers + dedupe/tier logic.
    async def _no_ddg(claim):
        return []

    monkeypatch.setattr(retrieval_agent, "_duckduckgo", _no_ddg)


CLAIM = Claim(claim_id="c1", text="The municipal corporation approved a new metro line on 11 July 2026",
              event_date="2026-07-11")


# ---------------- B.1 router ----------------
def test_route_accepts_url():
    r = intake_router.route(InputPayload(input_type="url", url="https://thehindu.com/news/article1.html"))
    assert r["reject_reason"] is None


def test_route_rejects_greeting():
    r = intake_router.route(InputPayload(input_type="text", text="hello"))
    assert r["reject_reason"]


def test_route_detects_language_for_accepted_text():
    r = intake_router.route(InputPayload(input_type="text", text=CLAIM.text))
    assert r["reject_reason"] is None
    assert r["language"]


# ---------------- B.2 scraper ----------------
@respx.mock
def test_scrape_happy_path():
    html = "<html><head><title>Metro Approved</title></head><body><article>" + ("City approves metro line. " * 30) + "</article></body></html>"
    respx.get("https://example.com/story").mock(return_value=httpx.Response(200, text=html))
    art = scraper_agent.scrape("https://example.com/story")
    assert art.body and not art.partial


@respx.mock
def test_scrape_never_raises_on_timeout():
    respx.get("https://dead.example/x").mock(side_effect=httpx.ConnectTimeout("boom"))
    art = scraper_agent.scrape("https://dead.example/x")
    assert art.partial is True  # degraded, not crashed


# ---------------- B.3 retrieval ----------------
@pytest.mark.asyncio
@respx.mock
async def test_retrieve_gathers_and_stamps_tiers(monkeypatch):
    monkeypatch.setattr(retrieval_agent, "TAVILY_KEY", "fake")
    monkeypatch.setattr(retrieval_agent, "GNEWS_KEY", "")       # skipped quietly
    monkeypatch.setattr(retrieval_agent, "FACTCHECK_KEY", "")   # skipped quietly
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json={
        "results": [
            {"content": "PIB confirmed the approval on Monday.", "url": "https://pib.gov.in/fact1"},
            {"content": "Duplicate.", "url": "https://pib.gov.in/fact1"},           # dedup target
            {"content": "Random blog take.", "url": "https://someblog.example/p"},  # unknown -> tier 3
        ]}))
    items = await retrieval_agent.retrieve([CLAIM])
    urls = [i.url for i in items]
    assert urls.count("https://pib.gov.in/fact1") == 1                       # deduped
    assert next(i for i in items if "pib.gov.in" in i.url).source_tier == 1  # trust list
    assert next(i for i in items if "someblog" in i.url).source_tier == 3    # unknown


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_survives_provider_failure(monkeypatch):
    monkeypatch.setattr(retrieval_agent, "TAVILY_KEY", "fake")
    monkeypatch.setattr(retrieval_agent, "GNEWS_KEY", "")
    monkeypatch.setattr(retrieval_agent, "FACTCHECK_KEY", "")
    respx.post("https://api.tavily.com/search").mock(side_effect=httpx.ReadTimeout("slow"))
    assert await retrieval_agent.retrieve([CLAIM]) == []  # degraded, no crash


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_drops_social_media_results(monkeypatch):
    # UGC/social platforms are repetition amplifiers, not sources - their
    # results must never become evidence.
    monkeypatch.setattr(retrieval_agent, "TAVILY_KEY", "fake")
    monkeypatch.setattr(retrieval_agent, "GNEWS_KEY", "")
    monkeypatch.setattr(retrieval_agent, "FACTCHECK_KEY", "")
    respx.post("https://api.tavily.com/search").mock(return_value=httpx.Response(200, json={
        "results": [
            {"content": "NASA CONFIRMS aliens!!", "url": "https://www.youtube.com/watch?v=x"},
            {"content": "Shocking discovery repost", "url": "https://www.facebook.com/groups/x/posts/1"},
            {"content": "Official mission update.", "url": "https://www.nasa.gov/news/x"},
        ]}))
    items = await retrieval_agent.retrieve([CLAIM])
    urls = [i.url for i in items]
    assert all("youtube" not in u and "facebook" not in u for u in urls)
    assert any("nasa.gov" in u for u in urls)


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_retries_once_before_giving_up(monkeypatch):
    monkeypatch.setattr(retrieval_agent, "TAVILY_KEY", "fake")
    monkeypatch.setattr(retrieval_agent, "GNEWS_KEY", "")
    monkeypatch.setattr(retrieval_agent, "FACTCHECK_KEY", "")
    route = respx.post("https://api.tavily.com/search")
    route.side_effect = [
        httpx.ReadTimeout("slow"),
        httpx.Response(200, json={"results": [{"content": "ok", "url": "https://pib.gov.in/fact1"}]}),
    ]
    items = await retrieval_agent.retrieve([CLAIM])
    assert route.call_count == 2
    assert len(items) == 1


# ---------------- B.3b fact-check via the Swytchcode kernel ----------------
# _factcheck no longer makes an httpx call, so respx can't intercept it - the
# seam is now agents.swytchcode_client.call. These are the first tests to cover
# this provider at all: every older retrieval test set FACTCHECK_KEY="" and
# short-circuited before the request was ever built.

def _use_factcheck_only(monkeypatch):
    monkeypatch.setattr(retrieval_agent, "FACTCHECK_KEY", "fake")
    monkeypatch.setattr(retrieval_agent, "TAVILY_KEY", "")
    monkeypatch.setattr(retrieval_agent, "GNEWS_KEY", "")


def _fake_kernel(monkeypatch, *, returns=None, raises=None, recorder=None):
    async def _call(canonical_id, request=None, **kwargs):
        if recorder is not None:
            recorder.append((canonical_id, request))
        if raises is not None:
            raise raises
        return returns

    monkeypatch.setattr(retrieval_agent.swytchcode_client, "call", _call)


_FACTCHECK_PAYLOAD = {
    "claims": [{
        "text": "A new metro line was approved on 11 July 2026",
        "claimReview": [{
            "url": "https://factly.in/metro-check",
            "textualRating": "False",
            "reviewDate": "2026-07-13",
            "publisher": {"name": "Factly"},
        }],
    }]
}


@pytest.mark.asyncio
async def test_factcheck_maps_kernel_payload_into_evidence(monkeypatch):
    _use_factcheck_only(monkeypatch)
    calls = []
    _fake_kernel(monkeypatch, returns=_FACTCHECK_PAYLOAD, recorder=calls)

    items = await retrieval_agent.retrieve([CLAIM])

    assert calls[0][0] == retrieval_agent.FACTCHECK_TOOL
    assert calls[0][1]["params"]["query"] == CLAIM.text
    item = next(i for i in items if "factly.in" in i.url)
    assert "rated 'False'" in item.snippet          # rating carried into the snippet
    assert item.source_name == "Factly"
    assert item.source_tier == 1                     # trust list still applied
    assert item.published_date == "2026-07-13"


@pytest.mark.asyncio
async def test_factcheck_degrades_when_kernel_unavailable(monkeypatch):
    # call() returns None when the CLI/tooling.json is missing. That must skip
    # the provider, not crash the run - the zero-setup demo path.
    _use_factcheck_only(monkeypatch)
    _fake_kernel(monkeypatch, returns=None)
    assert await retrieval_agent.retrieve([CLAIM]) == []


@pytest.mark.asyncio
async def test_unretryable_kernel_error_is_not_retried(monkeypatch):
    _use_factcheck_only(monkeypatch)
    calls = []
    _fake_kernel(
        monkeypatch,
        raises=SwytchcodeError("invalid API key", 3, {"category": "auth", "retryable": False}),
        recorder=calls,
    )
    assert await retrieval_agent.retrieve([CLAIM]) == []
    assert len(calls) == 1  # kernel said don't bother - no wasted second call


@pytest.mark.asyncio
async def test_provider_failure_is_logged_without_leaking_the_api_key(monkeypatch, caplog):
    # The kernel echoes the outbound request (API key and all) into stderr, which
    # becomes the exception message. retrieval must never write that to the log.
    _use_factcheck_only(monkeypatch)
    _fake_kernel(monkeypatch, raises=SwytchcodeError(
        '[swytchcode exec] request tool=x {"params":{"key":"SUPERSECRET123"}}\n'
        '{"error":"tool not configured","category":"not_found"}', 2, None,
    ))
    with caplog.at_level("WARNING", logger="verity.retrieval"):
        await retrieval_agent.retrieve([CLAIM])
    assert "SUPERSECRET123" not in caplog.text
    assert "tool not configured" in caplog.text


@pytest.mark.asyncio
async def test_retryable_kernel_error_is_retried_once(monkeypatch):
    _use_factcheck_only(monkeypatch)
    calls = []
    _fake_kernel(
        monkeypatch,
        raises=SwytchcodeError("upstream 503", 4, {"category": "transport", "retryable": True}),
        recorder=calls,
    )
    assert await retrieval_agent.retrieve([CLAIM]) == []
    assert len(calls) == 2  # one attempt + the single retry the constitution allows


# ---------------- B.4 pure functions ----------------
def _ev(tier, date="2026-07-13"):
    return EvidenceItem(evidence_id="e", claim_id="c1", snippet="s", url=f"https://t{tier}.example",
                        source_name="src", source_tier=tier, published_date=date,
                        retrieved_at="2026-07-14T10:22:00Z")


def test_credibility_weighted_by_tier():
    high, low = credibility_agent.score_credibility([_ev(1), _ev(1)]), credibility_agent.score_credibility([_ev(3), _ev(3)])
    assert high.score > low.score and high.weight == 0.10


def test_temporal_flags_recycled_story():
    stale = temporal_agent.check_temporal([CLAIM], [_ev(1, date="2024-01-05")])
    fresh = temporal_agent.check_temporal([CLAIM], [_ev(1, date="2026-07-13")])
    assert stale.score < fresh.score and "recycled" in stale.note
