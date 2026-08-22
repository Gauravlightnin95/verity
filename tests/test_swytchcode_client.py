"""Tests for agents/swytchcode_client.py - the Swytchcode execution-kernel adapter.

No test may spawn the real CLI: `swytchcode_exec` is monkeypatched at the module
seam, and kernel availability is forced explicitly in both directions so the
suite behaves identically on a machine that has the binary installed and one
that does not.
"""

import asyncio

import pytest
from swytchcode_runtime import SwytchcodeError

from agents import swytchcode_client


@pytest.fixture
def kernel_up(monkeypatch):
    monkeypatch.setattr(swytchcode_client, "_cli_available", lambda: True)


@pytest.fixture
def kernel_down(monkeypatch):
    monkeypatch.setattr(swytchcode_client, "_cli_available", lambda: False)


def _fake_exec(returns=None, raises=None, recorder=None):
    def _run(canonical_id, request=None, **kwargs):
        if recorder is not None:
            recorder.append({"canonical_id": canonical_id, "request": request, **kwargs})
        if raises is not None:
            raise raises
        return returns

    return _run


# ---------------- degradation (project rule 7) ----------------
@pytest.mark.asyncio
async def test_call_returns_none_when_kernel_unavailable(kernel_down, monkeypatch):
    # Must not even attempt to spawn: a missing CLI degrades a provider exactly
    # the way a missing API key already does, rather than crashing the pipeline.
    def _boom(*a, **k):
        raise AssertionError("exec must not be called when the kernel is unavailable")

    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _boom)
    assert await swytchcode_client.call("some.tool", {"params": {}}) is None


@pytest.mark.asyncio
async def test_available_reflects_cli_probe(kernel_down):
    assert swytchcode_client.available() is False


# ---------------- response unwrapping ----------------
@pytest.mark.asyncio
async def test_call_unwraps_data_envelope(kernel_up, monkeypatch):
    monkeypatch.setattr(
        swytchcode_client,
        "swytchcode_exec",
        _fake_exec(returns={"data": {"claims": [{"text": "x"}]}, "summary": "ok"}),
    )
    assert await swytchcode_client.call("t", {}) == {"claims": [{"text": "x"}]}


@pytest.mark.asyncio
async def test_call_passes_through_unwrapped_payload(kernel_up, monkeypatch):
    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _fake_exec(returns={"claims": []}))
    assert await swytchcode_client.call("t", {}) == {"claims": []}


# The kernel exits 0 for API-level failures - a 4xx means "the call ran and the
# server answered", not "the call failed". So an auth error arrives as a
# successful exec whose payload is an error document. Unwrapping `data` and
# ignoring `status_code` silently turns that into "no evidence for this claim".
_HTTP_400_ENVELOPE = {
    "data": {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                       "status": "INVALID_ARGUMENT"}},
    "request": {"method": "GET",
                "url": "https://factchecktools.googleapis.com/v1alpha1/claims:search?key=SUPERSECRET123"},
    "status_code": 400,
}


@pytest.mark.asyncio
async def test_api_level_error_raises_instead_of_returning_empty(kernel_up, monkeypatch):
    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _fake_exec(returns=_HTTP_400_ENVELOPE))
    with pytest.raises(SwytchcodeError) as excinfo:
        await swytchcode_client.call("t", {})
    assert "400" in str(excinfo.value)
    assert "API key not valid" in str(excinfo.value)


@pytest.mark.asyncio
async def test_api_level_error_never_echoes_the_request_url(kernel_up, monkeypatch):
    # The envelope's `request.url` carries the fully-built URL, API key included.
    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _fake_exec(returns=_HTTP_400_ENVELOPE))
    with pytest.raises(SwytchcodeError) as excinfo:
        await swytchcode_client.call("t", {})
    assert "SUPERSECRET123" not in str(excinfo.value)
    assert "SUPERSECRET123" not in swytchcode_client.describe_error(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,retryable",
    [(400, False), (401, False), (404, False), (429, True), (500, True), (503, True)],
)
async def test_provider_http_errors_get_sensible_retryability(kernel_up, monkeypatch, status, retryable):
    envelope = {"data": {"error": {"message": "nope"}}, "status_code": status}
    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _fake_exec(returns=envelope))
    with pytest.raises(SwytchcodeError) as excinfo:
        await swytchcode_client.call("t", {})
    assert swytchcode_client.is_retryable(excinfo.value) is retryable


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 201, 204])
async def test_success_statuses_unwrap_normally(kernel_up, monkeypatch, status):
    envelope = {"data": {"claims": [{"text": "x"}]}, "status_code": status}
    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _fake_exec(returns=envelope))
    assert await swytchcode_client.call("t", {}) == {"claims": [{"text": "x"}]}


@pytest.mark.asyncio
async def test_call_returns_none_for_non_dict_payload(kernel_up, monkeypatch):
    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _fake_exec(returns=None))
    assert await swytchcode_client.call("t", {}) is None


# ---------------- request plumbing ----------------
@pytest.mark.asyncio
async def test_call_forwards_request_and_bounds_the_subprocess(kernel_up, monkeypatch):
    calls = []
    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _fake_exec(returns={}, recorder=calls))
    await swytchcode_client.call("my.tool", {"params": {"query": "q"}})

    assert calls[0]["canonical_id"] == "my.tool"
    assert calls[0]["request"] == {"params": {"query": "q"}}
    # Rule 7 wants a timeout on every network call; the kernel performs the HTTP
    # request inside this subprocess, so the subprocess must be bounded.
    assert calls[0]["timeout"] == swytchcode_client.TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_dry_run_defaults_to_settings_but_is_overridable(kernel_up, monkeypatch):
    from app.config import settings

    calls = []
    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _fake_exec(returns={}, recorder=calls))

    monkeypatch.setattr(settings, "swytchcode_dry_run", True)
    await swytchcode_client.call("t", {})
    assert calls[-1]["dry_run"] is True

    await swytchcode_client.call("t", {}, dry_run=False)
    assert calls[-1]["dry_run"] is False


@pytest.mark.asyncio
async def test_kernel_errors_propagate_for_the_callers_retry_logic(kernel_up, monkeypatch):
    # Unlike an unavailable kernel, a call that actually executed and failed must
    # raise, so retrieval_agent._with_retry can consult the retryable flag.
    monkeypatch.setattr(
        swytchcode_client, "swytchcode_exec", _fake_exec(raises=SwytchcodeError("upstream 503"))
    )
    with pytest.raises(SwytchcodeError):
        await swytchcode_client.call("t", {})


@pytest.mark.asyncio
async def test_call_does_not_block_the_event_loop(kernel_up, monkeypatch):
    """The fan-out in _retrieve_one is concurrent; a synchronous subprocess run
    on the event loop thread would serialise it. Two concurrent calls against a
    blocking exec must overlap rather than queue."""
    import threading

    barrier = threading.Barrier(2, timeout=5)

    def _blocking(canonical_id, request=None, **kwargs):
        barrier.wait()  # only clears if both calls are genuinely in flight
        return {"ok": True}

    monkeypatch.setattr(swytchcode_client, "swytchcode_exec", _blocking)
    results = await asyncio.gather(swytchcode_client.call("a", {}), swytchcode_client.call("b", {}))
    assert results == [{"ok": True}, {"ok": True}]


# ---------------- error classification ----------------
def test_is_retryable_reads_the_kernels_classification():
    yes = SwytchcodeError("rate limited", 1, {"category": "rate_limit", "retryable": True})
    no = SwytchcodeError("bad key", 3, {"category": "auth", "retryable": False})
    assert swytchcode_client.is_retryable(yes) is True
    assert swytchcode_client.is_retryable(no) is False


# The real stderr the CLI produces: telemetry notice, an MCP warning, a spinner
# line, a request log, and only then the classified JSON. swytchcode_runtime
# json.loads() the whole blob, so it fails to parse and hands us details=None
# with the entire blob as the message. Captured verbatim from a live run.
_NOISY_STDERR = (
    "Telemetry is disabled. Run `swytchcode login` or set SWYTCHCODE_TOKEN to enable usage tracking.\n"
    "MCP server swytchcode already exists in local config\n"
    "⚠ could not register MCP server with Claude Code - configure it manually if needed\n"
    "⠋ Fetching factchecktools...⠙ Fetching factchecktools...\n"
    "2026/08/22 12:56:24 [swytchcode exec] request tool=factchecktools.v1alpha1.claimssearch.list\n"
    '{"error":"upstream timed out","category":"transport","retryable":true,'
    '"suggested_action":"retry the request","docs_url":"https://docs.swytchcode.com/cli/tools/"}'
)


def test_classification_survives_the_clis_noisy_stderr():
    exc = SwytchcodeError(_NOISY_STDERR, 4, None)  # details=None, as the runtime leaves it
    assert swytchcode_client.is_retryable(exc) is True

    note = swytchcode_client.describe_error(exc)
    assert note.startswith("upstream timed out")
    assert "category=transport" in note
    # The spinner frames and telemetry banner must never reach a CheckLog note.
    assert "Fetching" not in note
    assert "Telemetry" not in note
    assert "\n" not in note


def test_describe_error_stays_single_line_and_bounded():
    # Unclassifiable failures still must not dump a multi-line blob into the UI.
    exc = SwytchcodeError("line one\nline two\n" + "x" * 500)
    note = swytchcode_client.describe_error(exc)
    assert "\n" not in note
    assert len(note) <= 300


@pytest.mark.parametrize("category", ["not_found", "auth", "validation", "policy"])
def test_terminal_categories_are_not_retried_even_without_a_flag(category):
    # Real CLI records frequently omit `retryable`. A tool missing from
    # tooling.json or a rejected credential fails identically on a second
    # attempt, so the retry is pure added latency on every claim.
    exc = SwytchcodeError(f'{{"error":"nope","category":"{category}"}}', 2, None)
    assert swytchcode_client.is_retryable(exc) is False


def test_unknown_category_without_flag_defers_to_the_caller():
    exc = SwytchcodeError('{"error":"odd","category":"something_new"}', 2, None)
    assert swytchcode_client.is_retryable(exc) is None


def test_describe_error_does_not_leak_the_echoed_request():
    # The CLI echoes the outbound request to stderr, API key included. That blob
    # becomes exc.message, so anything logging it verbatim writes provider
    # secrets into logs/verity.log.
    stderr = (
        '2026/08/22 [swytchcode exec] request tool=x {"params":{"key":"SUPERSECRET123"}}\n'
        '{"error":"tool not configured","category":"not_found"}'
    )
    note = swytchcode_client.describe_error(SwytchcodeError(stderr, 2, None))
    assert "SUPERSECRET123" not in note
    assert "tool not configured" in note


def test_is_retryable_returns_none_when_unclassified():
    # Spawn failures and older CLI builds carry no details - the caller must be
    # able to tell "not retryable" from "kernel had no opinion" and fall back.
    assert swytchcode_client.is_retryable(SwytchcodeError("spawn failed")) is None
    assert swytchcode_client.is_retryable(ValueError("unrelated")) is None


def test_describe_error_surfaces_category_and_action():
    exc = SwytchcodeError(
        "quota exceeded", 1,
        {"category": "rate_limit", "retryable": True, "suggested_action": "back off and retry"},
    )
    note = swytchcode_client.describe_error(exc)
    assert "quota exceeded" in note
    assert "rate_limit" in note
    assert "back off and retry" in note


def test_describe_error_falls_back_to_str_for_plain_exceptions():
    assert swytchcode_client.describe_error(ValueError("boom")) == "boom"
