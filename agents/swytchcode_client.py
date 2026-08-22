"""Swytchcode execution-kernel adapter (shared infrastructure, owned by Member B).

Purpose: the single place VERITY talks to the Swytchcode CLI. Swytchcode is an
execution kernel for agent tool calls - it owns schema validation, auth
assembly, request building and typed error classification, so an agent stops
hand-rolling an HTTP client (and hand-rolling its failure modes) per provider.

Input:  a Swytchcode canonical tool id (e.g.
        "factchecktools.v1alpha1.claimssearch.list") plus the kernel's args
        object: {"body": ..., "params": ..., "Authorization": ..., "headers": ...}.
Output: the normalized JSON payload the kernel returns, as a dict. Never raises
        for an unavailable kernel - callers get `None` and a logged skip.

Why this matters for VERITY specifically. The hand-written providers this
replaces unpacked raw JSON positionally (`a["url"]`, `resp.json()["articles"]`).
When a provider renames a field, that is a KeyError, swallowed by
retrieval_agent's broad `except Exception`, and the evidence for a claim
*silently disappears* - dragging the verdict toward UNVERIFIABLE with nothing in
`checks_performed` to say why. For a system whose entire thesis is "show your
work", silent evidence loss is the worst failure mode available. The kernel
turns that same drift into a typed, categorised error that this module surfaces
as a real skip note.

Degradation contract (project rule 7): if the CLI binary or the project's
tooling.json is missing, every call logs once and returns None. VERITY still
completes a run on the keyless DuckDuckGo provider, exactly as it already does
when an API key is absent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from swytchcode_runtime import SwytchcodeError
from swytchcode_runtime import exec as swytchcode_exec

from app.config import settings

log = logging.getLogger("verity.swytchcode")

# Where `swytchcode init` writes the trusted-tool registry / execution policy.
# Its presence is what makes `swytchcode exec` offline-capable: the kernel reads
# only local files (tooling.json + integration bundles) and never calls the
# registry, so a demo machine needs no network for the kernel itself.
TOOLING_FILE = Path(__file__).resolve().parent.parent / ".swytchcode" / "tooling.json"

# Per-call subprocess timeout. Rule 7 wants a timeout on every network call;
# the kernel performs the HTTP request inside this subprocess, so bounding the
# subprocess bounds the network call.
TIMEOUT_SECONDS = 15.0

# Kernel error categories that cannot succeed on an immediate retry. Used only
# when the classified record omits an explicit `retryable` flag.
_TERMINAL_CATEGORIES = frozenset({"not_found", "auth", "validation", "policy"})


def _cli_available() -> bool:
    """True when both halves of the kernel are present: the binary and this
    project's tooling.json. Resolved per call rather than cached at import so
    that running `swytchcode init` / `add` does not require an app restart."""
    if shutil.which("swytchcode") is None:
        return False
    return TOOLING_FILE.is_file()


def available() -> bool:
    """Public probe, for callers that want to log a skip with their own wording."""
    return _cli_available()


async def call(
    canonical_id: str,
    request: dict[str, Any] | None = None,
    *,
    dry_run: bool | None = None,
) -> dict[str, Any] | None:
    """Execute one Swytchcode tool and return its normalized payload.

    Returns None (never raises) when the kernel is unavailable, so a missing CLI
    degrades a provider the same way a missing API key already does. A kernel
    error that *did* execute raises SwytchcodeError, so the caller's retry logic
    can consult `details["retryable"]`.

    `swytchcode_exec` is a synchronous subprocess call, so it runs in a worker
    thread - the same treatment retrieval_agent already gives the synchronous
    `ddgs` client and graph.py gives every sync capability. Blocking the event
    loop here would serialise the whole parallel provider fan-out.
    """
    if not _cli_available():
        log.info("%s skipped: swytchcode kernel unavailable (CLI or tooling.json missing)", canonical_id)
        return None

    effective_dry_run = settings.swytchcode_dry_run if dry_run is None else dry_run

    env: dict[str, str] = {}
    if settings.swytchcode_token:
        env["SWYTCHCODE_TOKEN"] = settings.swytchcode_token

    result = await asyncio.to_thread(
        swytchcode_exec,
        canonical_id,
        request,
        cwd=str(TOOLING_FILE.parent.parent),
        env=env or None,
        dry_run=effective_dry_run,
        timeout=TIMEOUT_SECONDS,
    )

    if not isinstance(result, dict):
        return None

    # The kernel's --json envelope is {"data": ..., "request": ..., "status_code": N}.
    # `status_code` is load-bearing: the CLI exits 0 for API-level failures
    # (4xx/5xx are "the call ran and the server answered"), so an auth failure
    # arrives here as a perfectly successful exec whose payload happens to be an
    # error document. Unwrapping `data` and discarding the status is how a 400
    # turns into {} turns into "no evidence for this claim" - the exact silent
    # evidence loss this whole module exists to prevent. Raise instead, so the
    # caller degrades loudly and the skip is logged.
    status_code = result.get("status_code")
    if isinstance(status_code, int) and not 200 <= status_code < 300:
        raise SwytchcodeError(
            f"provider returned HTTP {status_code}: {_provider_message(result.get('data'))}",
            status_code,
            {
                # 429 and 5xx are worth one retry; other 4xx are terminal.
                "category": "provider_http_error",
                "retryable": status_code == 429 or status_code >= 500,
            },
        )

    if isinstance(result.get("data"), dict):
        return result["data"]
    return result


def _provider_message(data: Any) -> str:
    """Best-effort one-line reason from a provider's error document.

    Deliberately reads only the error body: the envelope's sibling `request`
    field holds the fully-built URL with the API key in the query string, so it
    must never reach a log line or a CheckLog note.
    """
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"][:200]
        if isinstance(error, str):
            return error[:200]
        if isinstance(data.get("message"), str):
            return data["message"][:200]
    return "no error message in response body"


def _classify(exc: BaseException) -> dict[str, Any] | None:
    """Recover the kernel's classified error record {error, category, retryable,
    suggested_action, docs_url} from a failed call.

    swytchcode_runtime parses that record by running json.loads over the whole
    of stderr. In practice the CLI prints progress spinners, telemetry notices
    and MCP-registration warnings to stderr *before* the JSON, so that parse
    fails: `exc.details` arrives as None and `exc.message` is the entire noisy
    blob, spinner frames included. Left alone that blob would be what lands in a
    CheckLog note and renders in the UI's transparency panel.

    So: prefer the runtime's own parse when it worked, and otherwise scan
    stderr backwards for the last line that is a JSON object carrying "error".
    """
    if isinstance(exc, SwytchcodeError) and isinstance(exc.details, dict):
        if any(v is not None for v in exc.details.values()):
            return exc.details
    if not isinstance(exc, SwytchcodeError):
        return None
    for line in reversed((exc.message or "").splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("error"), str):
            return obj
    return None


def is_retryable(exc: BaseException) -> bool | None:
    """The kernel's own verdict on whether a failed call is worth retrying.

    Returns True/False when Swytchcode classified the error, or None when it
    could not (spawn failures, older CLI builds) so the caller can fall back to
    its own heuristic. This replaces guessing retryability from an HTTP status
    code with the execution layer's classification of what actually went wrong.
    """
    classified = _classify(exc)
    if classified is None:
        return None
    if isinstance(classified.get("retryable"), bool):
        return classified["retryable"]
    # Not every classified record carries an explicit `retryable` flag, but some
    # categories are self-evidently terminal: a tool that isn't in tooling.json,
    # a rejected credential or a schema-invalid request will fail identically on
    # a second attempt. Spending the retry on those is pure added latency.
    if classified.get("category") in _TERMINAL_CATEGORIES:
        return False
    return None


def describe_error(exc: BaseException) -> str:
    """One-line, human-readable rendering of a kernel failure for CheckLog notes
    and the UI transparency panel - carrying the category and suggested action
    the kernel supplies, instead of a bare stringified exception.

    Always one line: the raw stderr blob is multi-line and contains spinner
    frames, which would corrupt both the log stream and the UI panel.
    """
    classified = _classify(exc)
    if classified is None:
        return " ".join(str(exc).split())[:300]
    # Two shapes reach here: the runtime's own `details` dict (which carries the
    # metadata but *not* the message - that stays on exc.message), and the raw
    # record we salvaged from stderr (which does carry "error"). Handle both.
    message = classified.get("error") or getattr(exc, "message", "") or str(exc)
    parts = [message]
    if classified.get("category"):
        parts.append(f"category={classified['category']}")
    if classified.get("suggested_action"):
        parts.append(f"suggested: {classified['suggested_action']}")
    return " ".join(" | ".join(parts).split())[:300]
