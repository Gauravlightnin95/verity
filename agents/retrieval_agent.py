"""Evidence retrieval (owned by Member B).

Purpose: for each Claim, fan out to DuckDuckGo (keyless, primary), plus
Tavily, Google Fact Check Tools, GNews, and an archive/e-paper probe (using
`publication_hint` from VerityState) when their keys are set; dedupe by URL,
tier by data/publications.json, and cache to a local JSON file.
Input: claims (list[Claim]), publication_hint (str | None).
Output: list[EvidenceItem] (core.schemas). Every provider call must degrade
(skip + log) rather than raise on a missing key or timeout.

Google Fact Check Tools is executed through the Swytchcode kernel
(agents/swytchcode_client.py) instead of a hand-written httpx call, so schema
validation, request assembly and typed error classification belong to the
execution layer rather than to this module. Tavily, GNews and the Wayback probe
remain on raw httpx: they are not in the Swytchcode registry (verified with
`swytchcode discover`), and registering a custom OpenAPI spec is a
platform-account step rather than a CLI one.

DuckDuckGo needs no API key, so retrieval works out-of-the-box - the other
providers are pure bonus. The cache is a plain JSON-file-per-claim (keyed by
claim-text hash): instant, offline-safe, and - unlike the old ChromaDB
approach - needs no embedding model download (we only ever look up by hash,
never by vector similarity, so embeddings were wasted work).
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlparse

import httpx

from agents import swytchcode_client
from app.config import settings
from core.schemas import Claim, EvidenceItem

log = logging.getLogger("verity.retrieval")

# Swytchcode canonical tool id for Google Fact Check Tools' claims:search.
# Confirmed present in the Swytchcode registry via `swytchcode discover`; the
# bundle is fetched with `swytchcode get factchecktools` and enabled with
# `swytchcode add <id>`, which records it in .swytchcode/tooling.json.
FACTCHECK_TOOL = "factchecktools.v1alpha1.claimssearch.list"

# Sourced from app.config.settings (loaded from .env), not os.getenv - .env
# is only ever parsed by pydantic-settings into `settings`, never injected
# into the real process environment, so os.getenv(...) here would always
# read empty regardless of what's in .env.
TAVILY_KEY = settings.tavily_api_key or ""
FACTCHECK_KEY = settings.google_factcheck_api_key or ""
GNEWS_KEY = settings.gnews_api_key or ""
TIMEOUT = httpx.Timeout(10.0)
RETRIES = 1  # one retry per provider call, per project constitution

# ---------------------------------------------------------------- trust list
_PUB_FILE = Path(__file__).resolve().parent.parent / "data" / "publications.json"
try:
    _PUBLICATIONS = {p["domain"]: p for p in json.loads(_PUB_FILE.read_text(encoding="utf-8"))}
except Exception as exc:  # degrade: everything becomes tier 3
    log.warning("could not load publications.json (%s) - all sources tier 3", exc)
    _PUBLICATIONS = {}


# User-generated / social platforms are repetition amplifiers, not news
# sources - a hoax "supported" by YouTube videos, Facebook posts, and
# TikToks repeating it is exactly how misinformation propagates. Their
# results are dropped from evidence entirely.
_UGC_DOMAINS = {
    "youtube.com", "youtu.be", "facebook.com", "instagram.com", "tiktok.com",
    "reddit.com", "quora.com", "x.com", "twitter.com", "pinterest.com",
    "threads.net", "medium.com", "blogspot.com", "wordpress.com", "tumblr.com",
}


def _netloc(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.").lower()


def _is_ugc(url: str) -> bool:
    netloc = _netloc(url)
    return any(netloc == d or netloc.endswith("." + d) for d in _UGC_DOMAINS)


def _tier_for(url: str) -> int:
    netloc = _netloc(url)
    if netloc in _PUBLICATIONS:
        return _PUBLICATIONS[netloc]["tier"]
    # Subdomain inherits its parent's tier (e.g. science.nasa.gov -> nasa.gov).
    for domain, pub in _PUBLICATIONS.items():
        if netloc.endswith("." + domain):
            return pub["tier"]
    return 3  # unknown domain = tier 3


def _domain_for_publication(name: str) -> str | None:
    for pub in _PUBLICATIONS.values():
        if pub["name"].lower() == name.lower():
            return pub["domain"]
    return None


# ---------------------------------------------------------------- cache
# One JSON file per claim (keyed by claim-text hash) under data/cache/ (which
# .gitignore already ignores). Check it BEFORE the internet: rehearse a demo
# once -> every repeat query answers instantly and offline. No DB, no model
# download - we only ever look up by exact hash.
_CACHE_DIR = Path("data") / "cache" / "evidence"


def _cache_key(claim_text: str) -> str:
    return hashlib.sha1(claim_text.lower().strip().encode()).hexdigest()


def _cache_get(claim_text: str) -> list[EvidenceItem]:
    path = _CACHE_DIR / f"{_cache_key(claim_text)}.json"
    if not path.exists():
        return []
    try:
        return [EvidenceItem(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
    except Exception as exc:
        log.info("cache read failed: %s", exc)
        return []


def _cache_put(claim_text: str, items: list[EvidenceItem]) -> None:
    if not items:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / f"{_cache_key(claim_text)}.json"
        path.write_text(
            json.dumps([item.model_dump(mode="json") for item in items]), encoding="utf-8"
        )
    except Exception as exc:
        log.info("cache write failed: %s", exc)


async def _with_retry(fn: Callable[..., Awaitable[list[dict]]], *args) -> list[dict]:
    """One attempt + one retry per provider call. A 4xx (bad key, quota, bad
    request) won't fix on an immediate retry, so those raise straight away
    instead of wasting a second call and more latency.

    For Swytchcode-executed providers we don't have to infer that from a status
    code: the kernel classifies each failure and says whether it is retryable.
    Prefer its verdict, and fall back to the status heuristic for the providers
    still on raw httpx."""
    last_exc: Exception | None = None
    for attempt in range(1 + RETRIES):
        try:
            return await fn(*args)
        except Exception as exc:
            retryable = swytchcode_client.is_retryable(exc)
            if retryable is None and isinstance(exc, httpx.HTTPStatusError):
                retryable = not (400 <= exc.response.status_code < 500)
            if retryable is False:
                raise  # kernel (or 4xx) says a retry is pointless
            last_exc = exc
            log.info(
                "attempt %d failed for %s: %s",
                attempt + 1, fn.__name__, swytchcode_client.describe_error(exc),
            )
    raise last_exc  # caught by asyncio.gather(..., return_exceptions=True)


# ---------------------------------------------------------------- providers
# Each provider returns [] on missing key / no results. HTTP providers raise
# on real failure so _with_retry can decide; DuckDuckGo swallows its own
# errors (rate limits shouldn't retry). All failures end up skipped + logged.

async def _duckduckgo(claim: Claim) -> list[dict]:
    """Keyless web search (the ddgs library). Runs its synchronous client in
    a worker thread so it doesn't block the event loop."""
    def _search() -> list[dict]:
        from ddgs import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(claim.text, max_results=5))

    try:
        rows = await asyncio.to_thread(_search)
    except Exception as exc:
        log.info("DuckDuckGo search failed (rate limit / network): %s", exc)
        return []

    out: list[dict] = []
    for r in rows:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        out.append({
            "snippet": (r.get("body") or r.get("title") or "")[:300],
            "url": url,
            "source_name": urlparse(url).netloc,
        })
    return out


async def _tavily(client: httpx.AsyncClient, claim: Claim) -> list[dict]:
    if not TAVILY_KEY:
        log.info("Tavily skipped: no API key")
        return []
    resp = await client.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_KEY, "query": claim.text, "max_results": 5},
    )
    resp.raise_for_status()
    return [
        {"snippet": r.get("content", "")[:300], "url": r["url"], "source_name": urlparse(r["url"]).netloc}
        for r in resp.json().get("results", [])
    ]


async def _factcheck(client: httpx.AsyncClient, claim: Claim) -> list[dict]:
    """Google Fact Check Tools, executed through the Swytchcode kernel rather
    than a hand-rolled httpx call. Swytchcode owns request assembly, schema
    validation and error classification for this provider; VERITY still owns the
    credential (from .env) and still owns what counts as evidence.

    `client` is unused here - kept in the signature so every provider stays
    uniform under the asyncio.gather fan-out in _retrieve_one."""
    if not FACTCHECK_KEY:
        log.info("Google Fact Check skipped: no API key")
        return []
    payload = await swytchcode_client.call(
        FACTCHECK_TOOL,
        {"params": {"query": claim.text, "key": FACTCHECK_KEY, "pageSize": 5}},
    )
    if payload is None:  # kernel unavailable - degrade, exactly like a missing key
        return []
    out = []
    for c in payload.get("claims", []):
        for review in c.get("claimReview", []):
            out.append({
                "snippet": f"{c.get('text', '')} - rated '{review.get('textualRating', '?')}'"[:300],
                "url": review.get("url", ""),
                "source_name": review.get("publisher", {}).get("name", "fact-checker"),
                "published_date": review.get("reviewDate"),
            })
    return out


async def _gnews(client: httpx.AsyncClient, claim: Claim) -> list[dict]:
    if not GNEWS_KEY:
        log.info("GNews skipped: no API key")
        return []
    resp = await client.get(
        "https://gnews.io/api/v4/search",
        params={"q": claim.text, "token": GNEWS_KEY, "max": 5, "lang": "en"},
    )
    resp.raise_for_status()
    return [
        {"snippet": (a.get("description") or a.get("title", ""))[:300],
         "url": a["url"],
         "source_name": a.get("source", {}).get("name", urlparse(a["url"]).netloc),
         "published_date": a.get("publishedAt")}
        for a in resp.json().get("articles", [])
    ]


async def _archive_probe(client: httpx.AsyncClient, claim: Claim, publication_hint: str | None) -> list[dict]:
    """If Member C guessed the publication (via the shared pipeline state - we
    never import their code), search that paper's own site + web.archive.org.
    Finding the headline there proves the clipping really came from that paper."""
    if not publication_hint:
        return []
    domain = _domain_for_publication(publication_hint)
    if not domain:
        return []

    out: list[dict] = []

    # 1. Site-restricted search on the paper's own website (via Tavily, if keyed)
    if TAVILY_KEY:
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": TAVILY_KEY, "query": f'site:{domain} "{claim.text[:80]}"', "max_results": 3},
            )
            resp.raise_for_status()
            out += [
                {"snippet": r.get("content", "")[:300], "url": r["url"],
                 "source_name": f"{publication_hint} (own site)"}
                for r in resp.json().get("results", [])
            ]
        except Exception as exc:
            log.info("archive probe (own site) failed: %s", exc)

    # 2. Wayback Machine availability check (no API key needed)
    try:
        target = out[0]["url"] if out else f"https://{domain}"
        resp = await client.get("https://archive.org/wayback/available", params={"url": target})
        resp.raise_for_status()
        snap = resp.json().get("archived_snapshots", {}).get("closest")
        if snap and snap.get("available"):
            out.append({
                "snippet": f"Archived copy found for {target} (snapshot {snap.get('timestamp', '?')})",
                "url": snap["url"],
                "source_name": "Wayback Machine",
            })
    except Exception as exc:
        log.info("archive probe (wayback) failed: %s", exc)

    return out


# ---------------------------------------------------------------- main entry
async def retrieve(claims: list[Claim], publication_hint: str | None = None) -> list[EvidenceItem]:
    all_items: list[EvidenceItem] = []
    try:
        for claim in claims[:5]:
            cached = _cache_get(claim.text)
            if cached:
                log.info("cache hit for claim %s (%d items)", claim.claim_id, len(cached))
                all_items += cached
                continue
            fresh = await _retrieve_one(claim, publication_hint)
            _cache_put(claim.text, fresh)
            all_items += fresh
    except Exception as exc:  # the one unforgivable bug is a crash - so: don't
        log.error("retrieval degraded: %s", exc)
    return _dedupe(all_items)


async def _retrieve_one(claim: Claim, publication_hint: str | None) -> list[EvidenceItem]:
    # The whole async trick: fire all 4 searches at the same time, wait once.
    # Total time ~ the slowest single call, instead of the sum of all four.
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        results = await asyncio.gather(
            _with_retry(_duckduckgo, claim),
            _with_retry(_tavily, client, claim),
            _with_retry(_factcheck, client, claim),
            _with_retry(_gnews, client, claim),
            _with_retry(_archive_probe, client, claim, publication_hint),
            return_exceptions=True,  # one dead provider must not kill the rest
        )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    items: list[EvidenceItem] = []
    for provider, result in zip(("duckduckgo", "tavily", "factcheck", "gnews", "archive"), results):
        if isinstance(result, Exception):
            # describe_error, not the raw exception: a kernel failure carries the
            # whole of the CLI's stderr, which echoes the outbound request - API
            # key included - and spans many lines. Logging it verbatim would
            # write provider secrets into logs/verity.log.
            log.warning(
                "%s failed for claim %s: %s - skipped",
                provider, claim.claim_id, swytchcode_client.describe_error(result),
            )
            continue
        for i, raw in enumerate(result):
            if not raw.get("url"):
                continue
            if _is_ugc(raw["url"]):
                log.info("dropping UGC/social result for claim %s: %s", claim.claim_id, raw["url"])
                continue
            items.append(EvidenceItem(
                evidence_id=f"{claim.claim_id}-{provider}-{i}",
                claim_id=claim.claim_id,
                snippet=raw["snippet"],
                url=raw["url"],
                source_name=raw["source_name"],
                source_tier=_tier_for(raw["url"]),
                published_date=raw.get("published_date"),
                retrieved_at=now,
            ))
    return items


def _dedupe(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for item in items:
        key = (item.claim_id, item.url.rstrip("/"))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
