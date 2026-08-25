"""
tests/fakes.py
================
Test doubles for the Member D test suite.

`FakeStructuredChatModel` stands in for `langchain_core.language_models.chat_models.BaseChatModel`
wherever agent code calls `.with_structured_output(schema).invoke(messages)`.
This keeps claim/stance extraction tests fully offline and deterministic —
no real LLM API call is ever made, and no `respx`/HTTP-level mocking is
needed for these two agents since they never touch `httpx` directly (that
plumbing lives inside the `groq`/`langchain-groq` SDKs).

`FakeClaimClassifier` stands in for the check-worthiness classifier
`ClaimExtractor` runs over every sentence the LLM returns. The real one is
a Hugging Face pipeline that downloads a model on first use, which would
break the "fully offline" contract above (and cost ~30s a run), so every
claim-extraction test injects this instead.

`VerifyClient` (Member D's own HTTP client to Member A's backend) *does*
talk to httpx directly, so its tests use `respx` instead — see
`test_streamlit_app.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FakeStructuredChatModel:
    """A minimal stand-in for a LangChain `BaseChatModel` configured with
    `.with_structured_output(...)`.

    Parameters
    ----------
    responses:
        A list of values to return on successive `.invoke()` calls. Each
        entry may be:
          - a structured-output instance/dict to return normally, or
          - an `Exception` instance, which is raised instead (used to
            simulate transient failures for retry tests).
        If more calls are made than there are entries, the last entry is
        reused indefinitely.
    on_invoke:
        Optional callback invoked with the messages passed to `.invoke()`,
        for asserting what was actually sent to "Claude" (e.g. verifying
        the honesty rules are present in the prompt).
    """

    responses: list[Any] = field(default_factory=list)
    on_invoke: Optional[Callable[[list[Any]], None]] = None
    call_count: int = field(default=0, init=False)

    def with_structured_output(self, schema: Any) -> "FakeStructuredChatModel":
        # Real BaseChatModel.with_structured_output returns a new runnable;
        # here we just return self since this fake only ever produces
        # pre-programmed structured responses.
        return self

    def invoke(self, messages: list[Any]) -> Any:
        if self.on_invoke is not None:
            self.on_invoke(messages)

        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        result = self.responses[index]

        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class FakeClaimClassifier:
    """A stand-in for `agents.claim_extractor`'s check-worthiness
    classifier: called once per claim, returns one label per call.

    Parameters
    ----------
    labels:
        Labels returned on successive calls. An `Exception` entry is
        raised instead of returned, simulating a per-sentence failure. If
        more calls are made than there are entries, the last entry is
        reused indefinitely — so a single-entry list labels every claim
        the same way.
    by_text:
        Optional exact-text -> label overrides, consulted before `labels`.
        Use it when a test needs a specific mix of kept and dropped
        sentences rather than a positional sequence.
    """

    labels: list[Any] = field(default_factory=lambda: ["Check-worthy Factual"])
    by_text: dict[str, str] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list, init=False)

    @property
    def call_count(self) -> int:
        return len(self.seen)

    def __call__(self, text: str) -> str:
        self.seen.append(text)

        if text in self.by_text:
            return self.by_text[text]
        if not self.labels:
            raise AssertionError(f"FakeClaimClassifier has no label for {text!r}")

        index = min(len(self.seen) - 1, len(self.labels) - 1)
        label = self.labels[index]
        if isinstance(label, Exception):
            raise label
        return label
