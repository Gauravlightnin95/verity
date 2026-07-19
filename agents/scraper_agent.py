"""URL scraper (owned by Member B).

Purpose: fetch an article URL and extract its readable content.
Input: url (str). Output: ArticleContent (core.schemas), with partial=True
on paywalled/broken fetches instead of raising.
"""

import logging
from urllib.parse import urlparse

import httpx
import trafilatura

from core.schemas import ArticleContent

log = logging.getLogger("verity.scraper")

TIMEOUT = 10.0
RETRIES = 1
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; VERITY-hackathon/1.0)"}
MIN_BODY_CHARS = 300  # anything shorter is treated as a partial extraction


def scrape(url: str) -> ArticleContent:
    domain = urlparse(url).netloc.removeprefix("www.")

    html = _download(url)
    if html is None:
        log.warning("scrape failed for %s - returning partial empty article", url)
        return ArticleContent(title="", body="", domain=domain, partial=True)

    title, body, author, publish_date = "", "", None, None
    try:
        body = trafilatura.extract(html, url=url, include_comments=False) or ""
        meta = trafilatura.extract_metadata(html)
        if meta is not None:
            title = getattr(meta, "title", None) or ""
            author = getattr(meta, "author", None)
            publish_date = getattr(meta, "date", None)
    except Exception as exc:
        log.warning("trafilatura failed for %s: %s", url, exc)

    return ArticleContent(
        title=title,
        body=body,
        author=author,
        publish_date=publish_date,
        domain=domain,
        partial=len(body) < MIN_BODY_CHARS,  # paywall / broken page / thin content
    )


def _download(url: str) -> str | None:
    """One attempt + one retry, 10s timeout each. None on total failure."""
    for attempt in range(1 + RETRIES):
        try:
            resp = httpx.get(url, timeout=TIMEOUT, headers=HEADERS, follow_redirects=True)
            if resp.status_code < 400 and resp.text:
                return resp.text
            log.info("attempt %d: HTTP %d for %s", attempt + 1, resp.status_code, url)
        except httpx.HTTPError as exc:
            log.info("attempt %d failed for %s: %s", attempt + 1, url, exc)
    return None
