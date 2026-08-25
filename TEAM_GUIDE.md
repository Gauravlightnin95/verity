# VERITY — Team Guide

One place for: how the project is put together now that all four
workstreams are merged, what the input/output actually looks like, and
what to study so you can talk about this system confidently in front of
judges. For setup/run instructions, see the root `README.md` — this file
is the deeper technical/study companion.

---

## 1. Status: fully merged

All four workstreams (Member A's core/glue, B's internet work, C's image
work, D's AI brain + screen) are merged into one tree and wired for real
— there is no mock/toggle system anymore. `agents/graph.py` imports every
real capability directly; `tests/graph_fakes.py` + an autouse
`conftest.py` fixture provide the equivalent fast/offline fakes, but only
inside the test suite, not as shipped production code.

```
uv sync
copy .env.example .env
uv run pytest                          # 142 tests, all green, zero API keys needed
uv run uvicorn app.main:app --reload   # backend + web UI on :8000 -> open http://localhost:8000
uv run streamlit run ui/streamlit_app.py   # legacy Streamlit UI on :8501 (being retired)
```

```
curl -X POST http://localhost:8000/verify -F text="The city approved a new metro line."
curl -X POST http://localhost:8000/verify -F url="https://example.com/some-article"
curl -X POST http://localhost:8000/verify -F image=@clipping.png
curl http://localhost:8000/health
```

With zero API keys, every capability still degrades gracefully (skip +
log, never crash) — you get back a real, schema-valid `Verdict`, usually
`UNVERIFIABLE` with `caveats` explaining exactly what was skipped and why.

### Where everything lives now

| Capability | File | Originally |
|---|---|---|
| Route input | `agents/intake_router.py` | B |
| Scrape a URL | `agents/scraper_agent.py` | B |
| Retrieve evidence | `agents/retrieval_agent.py` (async) | B |
| Score source credibility | `agents/credibility_agent.py` | B |
| Check temporal consistency | `agents/temporal_agent.py` | B |
| OCR a photo | `agents/ocr_agent.py` | C |
| Image forensics | `agents/forensics_agent.py` + `forensics/*.py` | C |
| Local model runtime | `ml/openvino_runtime.py` | C |
| Extract claims | `agents/claim_extractor.py` | D |
| Stance detection | `agents/stance_agent.py` | D |
| AI-text detection | `agents/ai_text_detector.py` | D |
| Demo UI | `ui/web/` (Streamlit version retired) | D |
| Schemas/fusion/judge/wiring | `core/`, `agents/graph.py`, `agents/judge_agent.py` | A |

If you need to touch one of these: it's real code now, no flag to flip —
edit it, run `uv run pytest`, and (if it's on the hot path) smoke-test
live via `uvicorn` + `curl` per §1.

---

## 2. Input & output format

### 2.1 Input — the web UI, or the API directly

The UI (`ui/web/`, served at `http://localhost:8000`) is a thin HTTP client with three tabs
(paste text / paste URL / upload photo) that calls the same endpoint
below — nothing to choose between, they're the same contract. Calling
the API directly: `multipart/form-data` with **exactly one** of:

| Field | Type | Example |
|---|---|---|
| `text` | form field | `-F text="some claim"` |
| `url` | form field | `-F url="https://..."` |
| `image` | file upload | `-F image=@clipping.png` |

### 2.2 Output — always a `Verdict` JSON

Every call to `/verify` returns one JSON object matching
`core/schemas.py`'s `Verdict` model, synchronously, in the HTTP response
body (not streamed — the UI's stage-by-stage progress messages are a UX
aid around one blocking call, not a real event stream).

```json
{
  "label": "MIXED",
  "confidence": 0.11,
  "per_claim": [
    {"claim_id": "c1", "label": "TRUE", "supporting": 1, "refuting": 0},
    {"claim_id": "c2", "label": "FALSE", "supporting": 0, "refuting": 1}
  ],
  "evidence_citations": [
    {"evidence_id": "e1", "claim_id": "c1", "snippet": "...", "url": "...",
     "source_name": "PIB Fact Check", "source_tier": 1,
     "retrieved_at": "2026-07-14T10:22:00Z"}
  ],
  "signals": [
    {"name": "source_credibility", "score": 0.8, "weight": 0.10, "note": "..."}
  ],
  "caveats": ["..."],
  "checks_performed": [
    {"agent_name": "retrieval", "status": "ok", "duration_ms": 12, "note": ""}
  ]
}
```

| Field | What it means |
|---|---|
| `label` | One of `TRUE / MOSTLY_TRUE / MIXED / MISLEADING / FALSE / UNVERIFIABLE / SATIRE_OPINION` |
| `confidence` | 0–1, how strongly the evidence agrees (not "how true") |
| `per_claim` | Each extracted claim's own verdict + supporting/refuting counts |
| `evidence_citations` | Every source the verdict is allowed to rely on — empty only if `label` is `UNVERIFIABLE`/`SATIRE_OPINION` |
| `signals` | The individual weighted clues that went into fusion (credibility, forensics, ai_text, ...) |
| `caveats` | Human-readable warnings: skipped checks, low-confidence signals, unverified claims |
| `checks_performed` | The full transparency trail — which agent ran, how long, ok/skipped/failed. This is what the UI's "what we checked" panel renders from. |

`GET /health` just returns `{"status": "ok"}` — use it for a liveness
check / to confirm the server is up before demoing.

---

## 3. Tech stack to study

Grouped by how deep you need to go. **Must know** = core glue code you
should be able to explain line by line. **Should know** = understand the
concept and where it plugs in. **Nice to know** = good for a confident
one-liner if a judge asks.

### Must know (the core/glue surface area)

| Tech | What it is | Where it lives here |
|---|---|---|
| **Pydantic v2** | Python data-validation library; a schema is a class, invalid data raises immediately | `core/schemas.py` — every model, `model_validator` for the citation/UNVERIFIABLE rule |
| **FastAPI** | Async Python web framework | `app/main.py`, `app/api/routes_verify.py` — routing, `Form`/`File` uploads, auto docs at `/docs` |
| **LangGraph** | Graph-based agent orchestration on top of LangChain | `agents/graph.py` — this is the centerpiece, see 3.1 below |
| **langchain-groq** | LangChain's wrapper around the Groq API, adds structured output | `agents/judge_agent.py`, `agents/claim_extractor.py`, `agents/stance_agent.py` — `ChatGroq(...).with_structured_output(...)` |
| **Groq API** | Free-tier LLM host (fast inference, Llama 3.3) doing claim extraction, stance judging, and final verdict write-up. Stance is **batched** into one call for all claims, so a verify makes only ~4 LLM calls total — well under the free rate limit | `core/prompts.py` (every prompt), the three agent files above |
| **pytest + monkeypatch** | Testing framework; `monkeypatch` swaps out functions/attrs for the test's duration | every `tests/test_*.py` — e.g. `tests/graph_fakes.py` patched onto `agents.graph` instead of hitting the real API/network |
| **uv** | Fast, modern Python package manager (replaces pip/poetry/venv) | `pyproject.toml`, `uv.lock` |

#### 3.1 LangGraph — the concept you must be able to explain

- A **graph** is a set of **nodes** (plain Python functions) and
  **edges** (which node runs next). State flows through as one shared
  object.
- **Conditional edges**: `agents/graph.py`'s `_route_branch` picks
  `"text"` / `"url"` / `"image"` / `"rejected"` based on the input —
  this is how one input type takes a different path through the pipeline
  than another.
- **Fan-out / fan-in**: `retrieval` has edges to four nodes at once
  (`stance`, `credibility`, `temporal`, `ai_text`); LangGraph runs them
  as one "superstep" and only proceeds to `fusion` once **all four**
  finish. This is the "parallel Analysis Agents" from the architecture
  diagram, for real.
- **Reducers**: when multiple parallel nodes write to the *same* state
  field (here: `signals` and `checks_performed`), a plain overwrite
  would silently drop the other branches' results. `agents/graph.py`
  fixes this with `Annotated[list[Signal], operator.add]` — LangGraph
  automatically concatenates instead of overwriting. This is the single
  trickiest LangGraph detail in the codebase; understand it, it's a
  great thing to explain to a judge.
- **Async nodes + a uniform dispatch helper**: every node is `async def`;
  `_timed_call`/`_call_maybe_async` in `agents/graph.py` `await`s real
  async capabilities (evidence retrieval) directly and runs sync ones
  (OCR, forensics, the LLM agents) via `asyncio.to_thread`, so nothing
  blocks the event loop regardless of which capability is slow.
- **Compile once, invoke many times**: `build_graph()` returns a
  compiled graph; `get_graph()` caches it so every `/verify` call
  reuses the same compiled pipeline (`await graph.ainvoke(initial_state_dict)`).

### Should know (originally separate workstreams, now one tree)

| Tech | Used for | Originally |
|---|---|---|
| **httpx**, **trafilatura** | Async HTTP fetch + article extraction | B |
| **DuckDuckGo (`ddgs`)** | Keyless web search — the primary evidence source, so retrieval works with no API keys | B |
| **Tavily / Google Fact Check Tools / GNews APIs** | Optional bonus evidence providers (used when their keys are set) | B |
| **JSON file cache** | One file per claim under `data/cache/` — caches retrieved evidence so repeat queries are instant/offline (replaced the old ChromaDB vector store, which needed an embedding-model download for zero benefit) | B |
| **respx** | Mocks `httpx` calls in tests so they run without network | B (and anyone testing HTTP code) |
| **OpenCV** | Image preprocessing: perspective correction, adaptive threshold; also the ELA/copy-move/halftone math | C |
| **RapidOCR (onnxruntime)** | The actual OCR engine | C |
| **OpenVINO** | Intel's inference toolkit — `ml/openvino_runtime.py` selects NPU→GPU→CPU with automatic fallback; not yet in OCR's hot path (see README's hardware section for why) | C |
| **FFT (Fast Fourier Transform)** | The math behind halftone/print-dot detection — periodic patterns show up as sharp frequency peaks | C |
| **ELA (Error Level Analysis), copy-move detection** | Classic image-forensics techniques for spotting edits | C |
| **imagehash / SSIM** | Perceptual image similarity — used for masthead matching | C |
| **Hugging Face Transformers** | Core dependency: the check-worthiness classifier that filters extracted claims, plus optional local GPT-2 for perplexity (`DETECTOR_LOCAL=true`) | D |
| **Streamlit** | Turns a Python script into a web UI with no HTML/CSS/JS | D |

### Nice to know (concepts, for confident Q&A)

- **RAG (Retrieval-Augmented Generation)** — why the judge prompt forces
  "only cite what's in the snippets": this is the standard technique for
  stopping an LLM from hallucinating facts, applied to fact-checking.
- **Ensemble / weighted signal fusion** — `core/fusion.py` is a classic
  "committee of weak, specialized signals beats one black-box model"
  design. If asked "why not just ask an LLM if it's fake," this is your
  answer.
- **Perplexity & burstiness** — statistical fingerprints of AI-generated
  text (predictable word choices, uniform sentence length).
- **Quantization (INT8)** — shrinking a model's numeric precision for
  ~4x smaller/faster inference, small accuracy cost. Why local models can
  run on a laptop's NPU at all (not yet exercised here — see README).
- **Mock/toggle scaffolding as a build strategy** — until the merge, every
  capability had a hard-coded mock behind a settings flag, so all four
  workstreams could build in parallel against one frozen schema
  (`core/schemas.py`) without blocking on each other. That scaffolding is
  gone now that everything is real and merged; the same idea survives
  inside the test suite as `tests/graph_fakes.py`, so the end-to-end tests
  stay fast and offline without shipping mocks as production code.

---

## 4. Anticipated judge questions (cheat sheet)

**"Why not just ask an LLM if this is fake?"**
Because a single model can't tell truth from fluent writing, and it will
hallucinate confidently. We decompose into narrow signals (evidence
match, source credibility, forensics, AI-text style) with fixed weights,
and the LLM only writes up an explanation it's *forbidden* to contradict.

**"What happens with zero internet / zero API keys?"**
The whole pipeline still returns a valid verdict — every external call is
wrapped, failures degrade to a skipped/failed signal (never a crash), and
with no evidence at all the verdict is forced to `UNVERIFIABLE` rather
than guessing. This isn't theoretical: `judge` genuinely falls back to
the deterministic fusion verdict when there's no valid Groq key, and
it's visible in `checks_performed` — try it yourself, it's the real code
path, not a demo mode. (Web search itself needs no key — DuckDuckGo is
keyless — so even a fresh checkout gathers real evidence.)

**"How do you handle a real photo of a real newspaper vs. a fake
clipping?"**
Two separate questions, scored separately: *is this clipping a genuine
printed artifact* (halftone dot pattern via FFT, masthead match) vs. *are
the claims inside it true* (evidence retrieval). A real clipping can
carry a false story; a fake clipping can carry a true one — we report
both.

**"Why LangGraph instead of just calling functions in order?"**
Parallel branches (stance/credibility/temporal/ai_text all run at once
and are waited on together — not four sequential calls), a visual graph
that doubles as an architecture diagram, and per-node failure isolation
that keeps the pipeline alive when one agent fails.

**"How confident are you in the AI-text detector?"**
Deliberately not very — it's capped at 15% of the final weight, refuses
to score under 150 words, and every note carries a disclaimer. We treat
it as a stylistic hint, never a verdict on its own, because detectors
like this have known false-positive rates (OpenAI discontinued its own
detector for exactly this reason).
