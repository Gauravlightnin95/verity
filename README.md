# VERITY

**Verification Engine for Real-time Integrity of Textual & Visual news** —
an agentic news-verification system. Give it pasted text, a URL, or a
photo of a print newspaper clipping; it returns a graded, evidence-cited
verdict — never a bare TRUE/FALSE.

```
USER INPUT (text / url / photo)
  -> Intake Router          classify input, detect language, reject non-news
  -> Ingestion               URL -> scraper | photo -> OCR + image forensics
  -> Claim Extraction        LLM: decompose into <=5 atomic checkable claims
  -> Evidence Retrieval      parallel fan-out: DuckDuckGo (keyless), Tavily, Google Fact Check (via Swytchcode), GNews, archive probe
  -> Analysis (parallel)     stance detection, source credibility, AI-text, temporal
  -> Judge / Fusion          weighted signal fusion -> graded verdict + confidence + citations
  -> Report Renderer         web verdict card + PDF export
```

No single model decides truth. Narrow agents each emit a weighted
`Signal`; `core/fusion.py` combines them deterministically into a
`Verdict`, and an LLM judge writes the human-readable explanation without
being allowed to contradict it.

## Quick start

```
uv sync                                    # install deps (~2-3GB: torch ships by default now)
copy .env.example .env                     # your local config (gitignored)
uv run pytest                              # 154 tests, all green, zero API keys required
uv run uvicorn app.main:app --reload       # backend + web UI on :8000
```

Then open **http://localhost:8000** — the UI is served by the same
process, so there is no second command and no second port.

<details>
<summary>Legacy Streamlit UI (being retired)</summary>

```
uv run streamlit run ui/streamlit_app.py   # opens on :8501, talks to the backend above
```
</details>

The only key you really need is `GROQ_API_KEY` (free tier at
[console.groq.com](https://console.groq.com/keys)) — it powers claim
extraction, stance judging, and the verdict write-up. **Web search works
with no key at all** via DuckDuckGo, so evidence retrieval runs
out-of-the-box; `TAVILY_API_KEY` / `GNEWS_API_KEY` / `GOOGLE_FACTCHECK_API_KEY`
are optional bonus providers. Every capability degrades gracefully without
its key — it never crashes the pipeline. With zero keys the system still
returns a schema-valid verdict (typically `UNVERIFIABLE`, with `caveats`
explaining exactly what was skipped).

## How to check a real news claim

**Web UI (recommended)** — open `http://localhost:8000`, pick a tab:

- **Paste text**: drop in an article or a claim, click Check.
- **Paste URL**: drop in a link, click Check.
- **Upload photo**: upload a photo of a newspaper clipping (PNG/JPG), click Check.

Each returns a color-coded verdict badge, a per-claim breakdown with
evidence links, a "what we checked" transparency panel, and (for photos)
an "is this clipping genuine?" forensics panel — plus a PDF download.

**API directly** (what the UI calls under the hood):

```
curl -X POST http://localhost:8000/verify -F text="The city approved a new metro line on 11 July 2026."
curl -X POST http://localhost:8000/verify -F url="https://example.com/some-article"
curl -X POST http://localhost:8000/verify -F image=@clipping.png
curl -X POST http://localhost:8000/report -H 'Content-Type: application/json' -d @verdict.json --output report.pdf
curl http://localhost:8000/health
```

Every response is one `Verdict` JSON object:

| Field | Meaning |
|---|---|
| `label` | `TRUE / MOSTLY_TRUE / MIXED / MISLEADING / FALSE / UNVERIFIABLE / SATIRE_OPINION` |
| `confidence` | 0-1, how strongly the evidence agrees (not "how true") |
| `per_claim` | Each extracted claim's own label + supporting/refuting counts |
| `evidence_citations` | Every source the verdict is allowed to rely on |
| `signals` | The individual weighted clues fusion combined (credibility, forensics, ai_text, ...) |
| `caveats` | Skipped checks, low-confidence signals, unverified claims |
| `checks_performed` | Full transparency trail: which agent ran, how long, ok/skipped/failed |

## Architecture

Wired as a **LangGraph** `StateGraph` (`agents/graph.py`) over a shared
state object — each node reads what it needs, calls one capability
function, and writes the result back, wrapped in try/except so one
failing capability degrades the verdict instead of crashing the pipeline.
Every node is async; a shared dispatch helper awaits real async calls
(evidence retrieval) directly and runs sync calls (OCR, forensics, the
LLM agents) in a worker thread, so nothing blocks the event loop.

```
app/            FastAPI entrypoint, config, POST /verify + GET /health
agents/         one module per pipeline stage + graph.py (LangGraph wiring)
core/           schemas.py (frozen Pydantic contracts), fusion.py (verdict math), prompts.py (every LLM prompt)
ml/             openvino_runtime.py (NPU/GPU/CPU model loader)
forensics/      ela.py, copymove.py, halftone.py, masthead.py - pure OpenCV
data/           publications.json (25-outlet trust registry)
ui/web/         index.html + styles.css + app.js (the frontend, served at :8000)
ui/             streamlit_app.py (legacy frontend, being retired)
tests/          test_*.py + fixtures/ (sample clippings, contract JSON)
utils/          logging_conf.py (structured logging)
```

## Swytchcode execution kernel

Evidence providers are API calls, and a fact-checker that loses evidence
silently is worse than one that fails loudly. The hand-written providers
unpacked raw JSON positionally (`a["url"]`); when a provider renames a
field that is a `KeyError`, swallowed by retrieval's broad `except`, and
the evidence for a claim **silently disappears** — dragging the verdict
toward `UNVERIFIABLE` with nothing in `checks_performed` to say why.

[Swytchcode](https://www.swytchcode.com/) is an execution kernel for
agent tool calls: it owns request assembly, schema validation, auth and
typed error classification, so that same drift surfaces as a categorised
error instead of an empty list. Google Fact Check Tools runs through it
via [agents/swytchcode_client.py](agents/swytchcode_client.py) — the only
place VERITY touches the kernel.

```
npm install -g swytchcode
swytchcode login                                          # device-flow OAuth
swytchcode get "Google Other"                             # fetch the bundle
swytchcode add factchecktools.v1alpha1.claimssearch.list  # enable it in tooling.json
```

`get` takes a **project**, not a library. `factchecktools` is a library
inside the `Google Other` project, so `swytchcode get factchecktools`
fails with "may not have published bundles yet" — misleading, but it just
means the name isn't a project. `swytchcode search` lists project names;
`swytchcode discover "<intent>"` finds the method ids inside them. The
bundle we execute is committed; the other 210 Google libraries `get`
pulls down are gitignored as a local cache.

`.swytchcode/tooling.json` is the trusted-tool registry: a tool absent
from it cannot be executed, no matter what any agent asks for. That
enforces mechanically what the project constitution otherwise only
asserts in prose. `swytchcode exec` reads only local files and never
calls the registry, so once the bundle is committed the kernel runs
offline; set `SWYTCHCODE_DRY_RUN=true` to have every kernel call report
the request it *would* make without issuing it.

**It degrades like everything else here.** No CLI installed, no
`tooling.json`, or a tool not yet enabled — the provider is skipped and
logged, and the run completes on keyless DuckDuckGo. The transparency
panel gains a `swytchcode_kernel` row stating which it was.

One sharp edge worth knowing: the CLI **exits 0 for API-level failures**
(a 4xx means "the call ran and the server answered"), so an auth error
arrives as a successful exec whose payload happens to be an error
document. Unwrapping `data` and ignoring the envelope's `status_code` is
how a `400 API_KEY_INVALID` becomes `{}` becomes "no evidence for this
claim" — the exact silent loss described above, reintroduced one layer
down. `swytchcode_client.call()` checks `status_code` and raises, so a
bad key shows up as a logged skip rather than a quietly thinner verdict.

Tavily, GNews and the Wayback probe stay on raw `httpx`: they are not in
the Swytchcode registry (checked with `swytchcode discover`), and
registering a custom OpenAPI spec is a platform-account step rather than
a CLI one.

## Fusion weights

| Signal | Weight |
|---|---|
| Evidence stance | 45% |
| Archive/e-paper match | 15% |
| Source credibility | 10% |
| Image forensics (ELA/copy-move/halftone) | 10% |
| Masthead match | 5% |
| AI-text detector | 15% max — hard-capped, never decisive alone |

**AI-text detection is deliberately weak-signaled.** It refuses to score
under 150 words, is capped at 15% fusion weight, and always carries the
disclaimer: *"stylistic signal only — AI-text detectors have known
false-positive rates and cannot prove authorship."*

**`UNVERIFIABLE` is a first-class, neutral outcome.** Offline/regional
journalism is systematically under-indexed — absence of evidence is never
scored as `FALSE`.

## Hardware (Intel Core Ultra 7 258V / Arc 140V iGPU / NPU / 32GB LPDDR5X)

- `ml/openvino_runtime.py`'s `ModelManager` selects devices in priority
  order **NPU → GPU → CPU**, catching unavailable-device errors and
  falling through — correct as shipped for this chip, and safe on any
  CPU-only machine too.
- OpenCV's own OpenCL backend is explicitly disabled
  (`cv2.ocl.setUseOpenCL(False)`, in `forensics/__init__.py` and
  `agents/ocr_agent.py`) — forensics/OCR run on modest single images, not
  video, so OpenCL bought nothing here and had a reproducible teardown
  crash on this machine's iGPU driver stack once `openvino` was also
  installed. Fixing it was a one-line, zero-behavior-change call.
- **OCR runs on RapidOCR's default CPU execution provider, not
  OpenVINO, by design for now.** Real NPU/iGPU-accelerated OCR needs the
  `onnxruntime-openvino` package, which installs a conflicting build of
  the `onnxruntime` import namespace RapidOCR already depends on.
  Getting both to coexist means isolating OCR into its own environment,
  or converting RapidOCR's ONNX models to OpenVINO IR and running them
  through `ModelManager` directly instead of RapidOCR's own session —
  real engineering work, deliberately out of scope for this pass. OCR is
  still fast (~1-4s/image) on CPU.
- **`transformers` + `torch` (~2-3GB) are core dependencies**, installed
  by a plain `uv sync`. They used to sit in an optional `local-ai-text`
  extra, but the claim extractor's check-worthiness filter runs a local
  Hugging Face classifier over every extracted sentence, so without them
  every verify now fails at claim extraction rather than degrading. The
  extra name is kept as a no-op so the older command still works.
- **The local AI-text classifier stays opt-in at runtime.** That is a
  separate switch from the install: `DETECTOR_LOCAL=true` in `.env` turns
  on GPT-2 perplexity, and the default remains statistics-only
  (burstiness) with no weights download and no local inference cost.
- Converting any model to a quantized OpenVINO IR (`ml/convert/`) hasn't
  been done yet — nothing in the pipeline depends on it; it's genuinely
  new work, not integration, and is the natural next step if you want a
  real "runs on the NPU" demo moment beyond the architecture already
  being in place for it.

## Tech stack

Python 3.11+ • FastAPI + Uvicorn • Pydantic v2 • LangGraph + LangChain
(`langchain-groq`) • Groq API (Llama 3.3) • DuckDuckGo (`ddgs`, keyless
search) + Tavily / Google Fact Check Tools / GNews (retrieval) • httpx +
trafilatura (scraping) • Swytchcode CLI + `swytchcode-runtime` (tool-call execution
kernel) • local JSON evidence cache • RapidOCR (onnxruntime) •
OpenCV / scikit-image / imagehash (forensics) • OpenVINO (local model
runtime) • Streamlit + ReportLab (UI/PDF) • pytest + pytest-asyncio + respx
(tests) • uv (dependency management).

## Testing

```
uv run pytest                 # everything, ~9s, zero API keys/network
uv run pytest tests/ -k "fusion or forensics"   # a single area
```

Real network/LLM calls are never made in tests: HTTP is intercepted with
`respx`, LLM calls are faked at the `_build_chain`/`with_structured_output`
seam, the Swytchcode kernel is faked at `swytchcode_client.call` (no test
ever spawns the real CLI), and `tests/graph_fakes.py` + an autouse `conftest.py` fixture stand
in for every pipeline capability so `test_graph.py`/`test_api.py` stay
fast and deterministic while still exercising the real LangGraph wiring.

## Limitations (state these honestly, always)

1. VERITY assists, it does not adjudicate — outputs are evidence
   summaries with confidence, for human judgment.
2. AI-text detection is a weak signal, capped at 15% weight, worse on
   short/translated/regional text.
3. Offline/local journalism is systematically under-indexed —
   `UNVERIFIABLE` is neutral, never treated as `FALSE`.
4. Retrieval bias: search engines over-represent English and large
   outlets.
5. Adversarial fragility: a determined forger photographing a
   laser-printed fake on real newsprint can pass the halftone check.
6. Uploaded images are processed locally for forensics; only extracted
   text is sent to cloud APIs.

## Project status: prototype

VERITY is a working **hackathon prototype**, not a finished product. It
runs end-to-end today — real web search, real image forensics, real LLM
judging, graded and cited verdicts — but it was built by a four-person
team on free-tier APIs and a single laptop. Accuracy, coverage, and speed
will improve as the project gets more resources: paid API tiers remove
the daily LLM token budget, better search coverage widens the evidence
pool, and the outlet trust registry (currently 74 hand-curated
publications) grows with community review.

## Roadmap — where this goes next

**Verification memory (planned).** Today every check starts from zero.
The next major feature is a persistent claim memory: every verified claim,
its evidence, and its verdict get stored (claim text embeddings + a local
vector store), so that
- a hoax that was debunked once is recognized **instantly** the next time
  anyone submits a paraphrase of it — no re-retrieval, no LLM cost;
- verdicts can be **updated over time** as new evidence appears, with a
  visible history ("this claim was UNVERIFIABLE in July, confirmed FALSE
  in August");
- repeated-misinformation patterns (the same fake resurfacing every few
  months) become detectable and reportable.

**Other planned directions**, roughly in order:

- **NPU-accelerated local models** — the OpenVINO runtime (NPU → GPU →
  CPU) is already in place; converting the OCR and AI-text models to
  quantized IR moves them onto the Intel NPU for faster, cooler inference.
- **Regional-language support** — Hindi/Devanagari OCR and translation
  before retrieval, so print journalism that only exists in regional
  languages can be verified first-class.
- **Richer evidence sources** — dedicated fact-check aggregators, news
  archives, and e-paper portals alongside the current web search.
- **WhatsApp/messaging ingress** — meet misinformation where it actually
  spreads; forward a message to a bot, get a cited verdict back.
- **Community verification queue** — `UNVERIFIABLE` items get a human
  review lane, and reviewer verdicts feed back into the claim memory.
- **Deepfake detection for news photos** — extend the forensics suite
  beyond print clippings to manipulated photographs themselves.
