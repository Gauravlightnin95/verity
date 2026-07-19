---
title: "Project VERITY — Agentic AI News Verification System"
subtitle: "Hackathon Research, Architecture & Build Blueprint (with Claude Code prompts)"
author: "Team Lead Research Document"
date: "July 2026"
geometry: margin=2.2cm
fontsize: 10pt
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
toccolor: NavyBlue
toc: true
toc-depth: 2
numbersections: true
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{\small Project VERITY — News Verification System}
  - \fancyhead[R]{\small Hackathon Blueprint}
  - \fancyfoot[C]{\thepage}
  - \usepackage[dvipsnames]{xcolor}
---

\newpage

# Executive Summary

**Project VERITY** (Verification Engine for Real-time Integrity of Textual & Visual news) is an agentic AI system that takes a piece of news — as typed text, a URL, or **a photo of an offline/print-only newspaper** — and produces a graded, evidence-backed verdict on its authenticity, with citations and a confidence score.

The core insight of this design: **no single model can decide truth**. Instead, we orchestrate a team of specialized agents — a Claim Extractor, an Evidence Hunter, a Source Auditor, a Forensics Analyst (for images/OCR), an AI-Text Detector, and a Judge — each producing signals that a final aggregation agent fuses into a transparent verdict. This is what makes the system *agentic* rather than a single classifier, and it's what makes the output *trustable*: every verdict ships with its evidence trail.

**The offline-news problem** (news printed in a local paper but never published online) is solved with a layered fallback pipeline: OCR + layout analysis → digital e-paper archive cross-check → masthead/typography forensics → print-artifact detection (halftone patterns) → AI-generated-text detection → linguistic-style comparison against the claimed publication → and finally an honest "Unverifiable — here is what we checked" verdict when evidence is genuinely absent. The system never fakes certainty.

**Hardware reality**: the target build machine is a modern Intel Core Ultra "AI PC" laptop with integrated Arc graphics, an on-chip NPU, and shared system memory — a capable but thermally constrained thin-and-light platform. The blueprint therefore uses a **hybrid compute strategy**: small quantized models run locally on the NPU/iGPU via OpenVINO (cool, low-power), while heavy reasoning is delegated to cloud LLM APIs. Local inference is bursty, never sustained, keeping the chassis cool.

This document contains: (1) full problem analysis, (2) system architecture, (3) the offline-news verification playbook, (4) hardware-aware optimization plan, (5) recommended tech stack with alternatives, (6) a clean, labelled project file structure, (7) a complete sequence of ready-to-paste **Claude Code prompts** to build the project end-to-end, (8) a hackathon execution timeline, and (9) limitations you must state honestly in your demo.

\newpage

# Problem Statement & Challenge Analysis

## What "fake news detection" actually requires

The naive framing — "train a classifier that outputs TRUE/FALSE" — fails at hackathons and in production for the same reasons:

1. **Truth is not a property of text style.** A perfectly written article can be false; a badly written one can be true. Style-based classifiers learn publication quirks, not truth.
2. **Truth is time-dependent.** "X resigned today" may be false now and true tomorrow. Verification requires *live retrieval*, not a frozen model.
3. **Truth is claim-level, not article-level.** One article can contain five claims: three true, one exaggerated, one fabricated. The system must decompose.
4. **Absence of evidence ≠ evidence of absence.** Local and offline news often simply isn't indexed. A system that marks everything un-indexed as "fake" is dangerous and untrustworthy — especially in India where enormous amounts of legitimate journalism exist only in regional print.

## The hurdle matrix

| # | Hurdle | Why it's hard | Our mitigation |
|---|--------|---------------|----------------|
| H1 | News published only offline (print) | No digital record to cross-check | Photo upload + OCR + forensics pipeline (Section 4) |
| H2 | AI-generated fake articles | LLM text is fluent; detectors are imperfect | Detector as *one weighted signal*, never sole verdict |
| H3 | Manipulated/fabricated newspaper images | Photoshopped clippings look real | Image forensics: ELA, copy-move, halftone analysis, font consistency |
| H4 | Regional languages (Hindi, etc.) | OCR + retrieval quality drops | PaddleOCR multilingual + translation step before retrieval |
| H5 | Breaking news with no coverage yet | Legit news can be "unverifiable" | Graded verdict scale incl. "Unverified — too recent" |
| H6 | Source credibility gaming | Fake sites mimic real mastheads | Domain-age, WHOIS, known-source registry, IFCN signatory list |
| H7 | Hardware limits (thin laptop, thermal) | Can't run big models locally | NPU/iGPU offload + cloud hybrid (Section 5) |
| H8 | Hallucinating verifier | The LLM judge itself can invent evidence | Strict citation-required prompting; verdict must quote retrieved snippets |
| H9 | Satire / opinion pieces | Not "fake" but not factual | Genre classifier stage: satire/opinion/report before fact-check |
| H10 | Old news recirculated as new | True event, false timing | Date extraction + timeline check against archives |

## Design principles (write these on the demo slide)

- **Graded verdicts, never binary.** Output scale: `TRUE · MOSTLY TRUE · MIXED · MISLEADING · FALSE · UNVERIFIABLE · SATIRE/OPINION`.
- **Every verdict cites evidence.** No citation → capped at UNVERIFIABLE.
- **Signals, not oracles.** AI-text detectors, image forensics, and credibility scores are fused probabilistically; no single signal decides.
- **Fail honest.** When the pipeline can't verify, it says exactly which checks ran and what they found.
- **Cool and quiet.** Local compute is scheduled on the NPU first, iGPU second, CPU last, in short bursts.

\newpage

# System Architecture — The Agent Team

## High-level pipeline

```
 USER INPUT ──► [1. Intake Router]
   (text / URL / photo)      │
                             ▼
                  [2. Ingestion Agents]
                   ├─ URL → article scraper + metadata
                   ├─ Text → passthrough + language detect
                   └─ Photo → OCR Agent + Image Forensics Agent
                             │
                             ▼
                  [3. Claim Extraction Agent]
                   → atomic, checkable claims + entities + dates
                             │
                             ▼
                  [4. Evidence Retrieval Agent]  (parallel fan-out)
                   ├─ Web search (Tavily/Serper/Brave)
                   ├─ Google Fact Check Tools API
                   ├─ News APIs (GNews / NewsData.io)
                   ├─ E-paper & archive lookup (for print claims)
                   └─ Local knowledge cache (ChromaDB)
                             │
                             ▼
                  [5. Analysis Agents]  (parallel)
                   ├─ Stance Detection (does evidence support/refute?)
                   ├─ Source Credibility Auditor
                   ├─ AI-Text Detector (local, NPU)          ← photo/text path
                   ├─ Image Forensics (local, iGPU)          ← photo path
                   └─ Temporal Consistency Checker
                             │
                             ▼
                  [6. Judge / Aggregator Agent]
                   → weighted fusion → graded verdict
                   → confidence score + full evidence trail
                             │
                             ▼
                  [7. Report Renderer]  → UI card + shareable PDF
```

## Agent responsibilities (contract-style, so Claude Code can implement cleanly)

**1. Intake Router** — Classifies input type; rejects non-news inputs politely; detects language; routes to the correct ingestion path. *Runs locally (regex + small classifier).*

**2a. Scraper Agent (URL path)** — Fetches article with `trafilatura`/`newspaper4k`; extracts headline, body, byline, publish date, canonical domain. Captures `robots.txt` compliance.

**2b. OCR Agent (photo path)** — PaddleOCR (or RapidOCR ONNX on OpenVINO) with layout analysis: separates masthead, headline, body columns, captions, date line. Outputs structured text + per-region confidence. *Runs on NPU/iGPU.*

**2c. Image Forensics Agent (photo path)** — Runs Error Level Analysis, copy-move detection, noise-pattern (PRNU-style) consistency, EXIF audit, and **halftone-dot detection** (a genuine photo of a printed newspaper shows rosette/halftone patterns from offset printing; a screenshot of a digitally fabricated "clipping" usually does not). *Runs on iGPU, bursty.*

**3. Claim Extraction Agent** — LLM call (cloud). Decomposes article into ≤5 atomic claims: `{claim, entities, location, date, checkability: high/med/low}`. Filters opinion/satire via genre tag.

**4. Evidence Retrieval Agent** — Fan-out queries per claim; deduplicates; ranks by source tier; stores snippets with URL + retrieval timestamp into ChromaDB for caching and for the citation trail.

**5a. Stance Detector** — For each (claim, evidence snippet) pair: SUPPORTS / REFUTES / NEUTRAL. Two options: a small NLI model (DeBERTa-v3-small, quantized, local NPU) for cheap pre-filtering, then cloud LLM for the final stance on top-k snippets.

**5b. Source Credibility Auditor** — Scores each evidence domain: known-outlet registry (hand-curated JSON of ~200 outlets incl. Indian nationals + PIB Fact Check + IFCN signatories), domain age heuristic, HTTPS, about-page presence. Output 0–1 tier score.

**5c. AI-Text Detector** — Local quantized transformer detector + perplexity/burstiness statistics. **Crucially: output is a probability with an honesty disclaimer, weighted at most ~15% in fusion** (Section 4.5 explains why).

**5d. Temporal Checker** — Compares claim dates vs. evidence dates; flags "old news as new" and "future-dated" anomalies.

**6. Judge Agent** — Cloud LLM with a strict system prompt: must produce JSON `{verdict, confidence, per_claim: [...], evidence_citations: [...], caveats}`; is *forbidden* from asserting facts not present in retrieved snippets; must down-grade to UNVERIFIABLE when evidence is thin.

**7. Report Renderer** — Streamlit/React card: verdict badge, confidence bar, claim-by-claim table, clickable citations, "what we checked" transparency panel, and a one-click PDF export (great for judges).

## Orchestration choice

Use **LangGraph** (graph of nodes = agents above, with parallel fan-out for retrieval/analysis) or, for maximum hackathon simplicity, **plain async Python with Claude tool-use**. Recommendation: **LangGraph** — its visual graph is itself a demo asset, checkpointing gives you resilience when a web call fails mid-demo, and parallel branches map exactly to the architecture diagram. CrewAI is a fine alternative if the team prefers role-based ergonomics; avoid AutoGen for this (heavier than needed).
\newpage

# The Offline-News Problem — Deep-Dive Playbook

This is your differentiator. Most hackathon fact-checkers die the moment a claim isn't on Google. VERITY runs a **seven-layer fallback chain** for a photo of a print-only article:

## Layer 1 — High-fidelity OCR with layout understanding

- **PaddleOCR PP-OCRv4** (multilingual: English, Hindi/Devanagari, and more) exported to ONNX and executed with **OpenVINO Execution Provider** so it runs on the integrated Arc GPU or NPU.
- Layout parsing separates: **masthead** (publication name/logo), **dateline**, **headline**, **byline**, **body columns**, **captions**. Each region gets its own confidence score.
- Pre-processing: perspective correction (OpenCV `getPerspectiveTransform`), adaptive thresholding, deskew — newspaper photos are almost never flat.

## Layer 2 — Digital-archive cross-check (the cheap win)

"Offline-only" is often "offline-first": many Indian papers publish **e-paper editions** (page-image PDFs) even for content never posted as web articles. Pipeline:

1. From the masthead + dateline, identify publication and date.
2. Query: publication e-paper portal, Google News archive, `site:` searches on the publication's domain, and the Internet Archive.
3. If the same headline/date is found in the e-paper → strong authenticity signal for the *artifact* (the clipping is real), separate from the *claims* (which still go through normal fact-checking).

**Key conceptual move for your demo**: VERITY verifies two distinct things and reports them separately — **(a) Is this clipping a genuine artifact of the claimed publication?** and **(b) Are the claims inside it true?** A real clipping can carry a false story; a fabricated clipping can carry a true one.

## Layer 3 — Masthead & typography forensics

- Maintain a small local registry (`data/publications.json`) of ~50 known publications: masthead reference images, canonical fonts, column widths, date formats.
- Compare the uploaded masthead against the reference via perceptual hash + SSIM. Font-consistency check: fake clippings frequently mix fonts or use fonts the paper never uses.
- Date-format check: e.g., a paper that always prints "New Delhi | 9 July 2026" but the clipping shows "07/09/26" is a red flag.

## Layer 4 — Print-artifact (halftone) detection

Genuine offset-printed newspapers show **halftone dot patterns / rosettes** visible when a photo is taken at normal resolution; newsprint has characteristic texture and ink bleed. A "clipping" that is actually a digitally rendered image (crisp anti-aliased text, no dot structure, uniform background) is likely fabricated. Implementation: FFT-based periodic-pattern detection on background regions + local variance texture statistics. Cheap (OpenCV, CPU/iGPU), fast, and a fantastic demo moment.

## Layer 5 — AI-generated-text detection (handle with honesty)

The user's suggested approach — detect whether the OCR'd text is AI-generated vs human-written — is included, but engineered responsibly:

- **Models (local, quantized, OpenVINO/ONNX):** a RoBERTa/DeBERTa-based AI-text classifier fine-tuned on human-vs-LLM news corpora, plus statistical features (perplexity under a small local LM, burstiness, token-distribution flatness).
- **Hard truth you must state in the demo:** AI-text detectors have significant false-positive rates, degrade on short texts, on translated/regional-language text, and on lightly paraphrased LLM output. OpenAI withdrew its own detector for low accuracy. Therefore VERITY (a) requires ≥150 words before scoring, (b) caps this signal's fusion weight at ~15%, (c) reports it as "stylistic signal" not proof, and (d) never lets it alone flip a verdict.
- This *honest framing* scores better with judges than overclaiming.

## Layer 6 — Linguistic-style consistency

Embed the OCR'd body text (local MiniLM embedding model on NPU) and compare against a cached style profile of the claimed publication (built from ~30 of its public articles). Large style distance = another soft red flag. Fully local, fast.

## Layer 7 — Honest UNVERIFIABLE + human-loop hook

If layers 2–6 produce weak/neutral signals, the verdict is `UNVERIFIABLE (offline source)` with the full checklist of what was tested. Optional stretch feature: a "community verification" queue where the claim is logged for human review — mention it as roadmap, don't build it in the hackathon.

## Fusion weights for the photo path (starting point, tune later)

| Signal | Weight | Direction |
|---|---|---|
| Claims corroborated/refuted by retrieved evidence | 45% | dominant |
| Archive/e-paper artifact match | 15% | authenticity of clipping |
| Source credibility of claimed publication | 10% | prior |
| Image forensics (ELA, copy-move, halftone) | 10% | artifact authenticity |
| Masthead/typography consistency | 5% | artifact authenticity |
| AI-text detection | 15% max | stylistic signal only |

\newpage

# Hardware-Aware Engineering — Intel Core Ultra "AI PC" Laptops

## What you actually have (and why it's better than it sounds)

A current-generation Intel Core Ultra mobile SoC is a low-power platform with three compute engines you can use *independently*:

- **CPU**: a mix of P-cores and E-cores — fine for orchestration, I/O, OpenCV.
- **iGPU**: integrated Arc graphics, tens of TOPS class — good for OCR, vision models, small-LLM inference.
- **NPU**: tens of TOPS, extremely power-efficient — ideal for embeddings, classifiers, detectors. **Running work on the NPU is the single best thermal trick on this class of machine**: it delivers inference at a fraction of the wattage of CPU/GPU.
- **Unified system memory**, shared across CPU/GPU/NPU — no discrete-VRAM ceiling; on a typical 16–32 GB configuration, a 7B Q4 model (~4.5 GB) plus OCR plus embeddings all fit comfortably.

## The hybrid compute strategy (thermal-first)

| Workload | Where it runs | Why |
|---|---|---|
| Agent reasoning, claim extraction, judging | **Cloud LLM API** (Claude / other) | Zero local heat; best quality; a hackathon Wi-Fi call is cheaper than a 30 s local 8B generation |
| OCR (PaddleOCR/RapidOCR via OpenVINO) | **NPU → iGPU fallback** | Vision workloads map perfectly to NPU; ~1–2 s per image |
| AI-text detector (DeBERTa-small INT8) | **NPU** | Tiny model, milliseconds, near-zero power |
| Embeddings (all-MiniLM-L6-v2 INT8) | **NPU** | Same |
| Image forensics (OpenCV, FFT) | **CPU/iGPU** | Short bursts, negligible heat |
| Optional offline-demo LLM (Llama-3.1-8B / Qwen2.5-7B, Q4) | **iGPU via llama.cpp SYCL/Vulkan or IPEX-LLM/OpenVINO GenAI** | Only as a fallback mode if venue Wi-Fi dies — cap at short outputs |

## Concrete optimization checklist

1. **OpenVINO everywhere local.** Convert every local model to OpenVINO IR, quantize to **INT8 with NNCF** (or INT4 for the fallback LLM). Use `device="NPU"` with `"GPU"` fallback: `core.compile_model(model, "NPU")`.
2. **ONNX Runtime + OpenVINO EP** for anything already in ONNX (RapidOCR ships ONNX models — easiest path).
3. **Lazy loading + model unloading.** Load the forensics models only when a photo arrives; free them after. Keeps RAM headroom and idle temps low.
4. **Async fan-out for network calls.** `asyncio.gather` on all retrieval calls: total latency ≈ slowest call, and the CPU sleeps while waiting (cool).
5. **Batch = 1, streaming UI.** Show results per-stage as they complete (OCR done → claims extracted → evidence found → verdict). Perceived speed without sustained load.
6. **Thermal guards in code**: a `utils/thermal.py` that reads package temperature (via `pyspectator`/WMI on Windows or `psutil` sensors on Linux) and, above ~85 °C, delays local inference jobs by a few seconds and forces NPU-only mode. Gimmicky? Slightly. Judge-impressing systems-awareness? Absolutely.
7. **Windows power mode**: set "Best power efficiency" while demoing on battery; the NPU path barely notices.
8. **Cache aggressively.** ChromaDB persists retrieved evidence; repeated demo runs are instant and offline-safe.
9. **If local LLM fallback is enabled**: `llama.cpp` built with SYCL for Arc, context ≤4k, `n_gpu_layers=-1`, Q4_K_M, max_tokens ≤512. Expect roughly ~10–20 tok/s class performance on this iGPU — usable for a fallback, not your primary path.

\newpage

# Recommended Tech Stack (with alternatives)

## Primary recommendation ("boring on the edges, novel in the middle")

| Layer | Choice | Alternatives | Rationale |
|---|---|---|---|
| Language | Python 3.11+ | — | Ecosystem for ML + agents |
| Agent orchestration | **LangGraph** | CrewAI, plain Claude tool-use loop | Parallel branches, checkpointing, graph = demo visual |
| Reasoning LLM | **Claude API (claude-sonnet-4-6)** | GPT-4o, Gemini, Groq-hosted Llama for free speed | Strong citation-following & JSON reliability |
| Web search | **Tavily API** | Serper.dev, Brave Search API | Purpose-built for RAG, generous free tier |
| Fact-check lookup | **Google Fact Check Tools API** | ClaimBuster API | Free, aggregates IFCN fact-checkers |
| News retrieval | **GNews / NewsData.io** | NewsAPI.org | Free tiers, Indian coverage |
| OCR | **PaddleOCR → ONNX → OpenVINO** | RapidOCR (ONNX-native), Tesseract 5 | Accuracy on newsprint + NPU offload |
| Local inference runtime | **OpenVINO 2025.x + NNCF** | ONNX Runtime (OpenVINO EP), IPEX-LLM | First-party Intel NPU/iGPU support |
| AI-text detection | Fine-tuned **DeBERTa-v3-small** INT8 + perplexity stats | RoBERTa-base OpenAI-detector lineage, Binoculars method | Small, NPU-friendly |
| Embeddings | **all-MiniLM-L6-v2** INT8 | bge-small-en-v1.5 | 80 MB, milliseconds on NPU |
| Vector store | **ChromaDB (persistent, local)** | FAISS + SQLite | Zero-ops |
| Image forensics | **OpenCV + NumPy/SciPy** (ELA, copy-move, FFT halftone) | forensics libs (noiseprint) | Transparent, explainable, light |
| Backend API | **FastAPI** | Flask | Async-native for fan-out |
| Frontend | **Streamlit** (hackathon) | Next.js + Tailwind (if a frontend dev is free) | UI in hours, not days |
| Fallback local LLM | Qwen2.5-7B-Instruct Q4 via llama.cpp-SYCL | Llama-3.1-8B, Phi-3.5-mini | Offline demo insurance |
| Packaging | `uv` for env, `.env` for keys | pip/poetry | Fast, reproducible |

## IDE & AI-assistant tooling for the team

- **VS Code + Claude Code (terminal or VS Code extension)** — primary build driver; the prompt sequence in Section 8 is written for it.
- **Cursor** — fine alternative if a teammate prefers inline-edit ergonomics; the same prompts work.
- **Jupyter / marimo notebooks** inside `notebooks/` for model-conversion experiments (OpenVINO quantization) so experiments never pollute app code.
- **GitHub + branch-per-agent** workflow: `feat/ocr-agent`, `feat/retrieval-agent` — agents are decoupled by contract, so the team parallelizes cleanly.
- **Postman/Bruno** collection for the FastAPI endpoints — lets the frontend person work against mocks from hour one.

## Project file structure (clean & labelled — enforce this in every Claude Code prompt)

```
verity/
├── README.md                  # setup, architecture diagram, demo script
├── pyproject.toml             # deps managed by uv
├── .env.example               # ANTHROPIC_API_KEY, TAVILY_API_KEY, ...
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── config.py              # settings, model paths, fusion weights
│   └── api/
│       └── routes_verify.py   # POST /verify (text|url|image)
├── agents/
│   ├── graph.py               # LangGraph wiring of the whole pipeline
│   ├── intake_router.py
│   ├── scraper_agent.py
│   ├── ocr_agent.py
│   ├── forensics_agent.py
│   ├── claim_extractor.py
│   ├── retrieval_agent.py
│   ├── stance_agent.py
│   ├── credibility_agent.py
│   ├── ai_text_detector.py
│   ├── temporal_agent.py
│   └── judge_agent.py
├── core/
│   ├── schemas.py             # Pydantic models: Claim, Evidence, Verdict...
│   ├── fusion.py              # weighted signal fusion → graded verdict
│   └── prompts.py             # every LLM prompt, versioned constants
├── ml/
│   ├── openvino_runtime.py    # device selection NPU→GPU→CPU, lazy load
│   ├── models/                # .xml/.bin OpenVINO IR files (gitignored)
│   └── convert/               # one-off conversion+quantization scripts
├── forensics/
│   ├── ela.py                 # error level analysis
│   ├── copymove.py
│   ├── halftone.py            # FFT print-artifact detector
│   └── masthead.py            # pHash/SSIM vs publications registry
├── data/
│   ├── publications.json      # known outlets registry + credibility tiers
│   └── cache/                 # ChromaDB persistence (gitignored)
├── ui/
│   └── streamlit_app.py       # demo frontend
├── tests/
│   ├── test_fusion.py
│   ├── test_ocr_agent.py
│   └── fixtures/              # sample clippings: real, fabricated, AI-written
├── utils/
│   ├── thermal.py             # temperature guard
│   └── logging_conf.py        # structured logs per agent (great for demo)
└── notebooks/
    └── 01_model_conversion.ipynb
```
\newpage

# Claude Code Prompt Sequence — Build the Project Step-by-Step

Paste these into Claude Code **in order**, one at a time, reviewing output between steps. Each prompt is self-contained, restates the file-structure contract (so generated code stays clean and labelled), and ends with a verification command. Start Claude Code inside an empty `verity/` folder with `git init` done.

> **Tip:** first run `/init` in Claude Code after Prompt 1 so it generates a `CLAUDE.md`; then paste the "Project constitution" below into that `CLAUDE.md` — Claude Code will obey it in every later prompt.

## Prompt 0 — Project constitution (put this in CLAUDE.md)

```text
This repo is VERITY, an agentic news-verification system. Rules you must
always follow:
1. Respect the existing folder layout (app/, agents/, core/, ml/,
   forensics/, data/, ui/, tests/, utils/). Never dump code in root.
2. Every module starts with a docstring: purpose, inputs, outputs.
3. All inter-agent data uses Pydantic models from core/schemas.py —
   never raw dicts across module boundaries.
4. All LLM prompts live ONLY in core/prompts.py as named constants.
5. All config (API keys, model paths, fusion weights, device order
   NPU>GPU>CPU) lives in app/config.py, loaded from .env.
6. Local ML inference must go through ml/openvino_runtime.py, which
   lazy-loads models and selects device NPU→GPU→CPU. Target hardware:
   an Intel Core Ultra "AI PC" laptop with integrated Arc GPU, NPU,
   and shared memory. Keep local inference bursty; unload models
   after use.
7. Every network call is async with a timeout and one retry.
8. Verdicts must be one of: TRUE, MOSTLY_TRUE, MIXED, MISLEADING,
   FALSE, UNVERIFIABLE, SATIRE_OPINION. Never output a verdict without
   an evidence_citations list; if empty, verdict must be UNVERIFIABLE.
9. Write/extend a pytest for every module you create.
10. Use uv for dependency management; update pyproject.toml, never
    pip-install ad hoc.
```

## Prompt 1 — Scaffold

```text
Create the initial scaffold for VERITY, an agentic fake-news
verification system, using exactly this structure: [paste the file
tree from the blueprint]. Initialize pyproject.toml with uv, deps:
fastapi, uvicorn, pydantic, pydantic-settings, langgraph, langchain-
anthropic, httpx, trafilatura, chromadb, streamlit, opencv-python,
numpy, scipy, pillow, pytest, python-dotenv. Create .env.example with
ANTHROPIC_API_KEY, TAVILY_API_KEY, GNEWS_API_KEY, GOOGLE_FACTCHECK_
API_KEY. Create core/schemas.py with Pydantic models: InputPayload,
OCRResult (regions: masthead/dateline/headline/body with confidences),
Claim, EvidenceItem (snippet,url,source_tier,retrieved_at), Signal
(name,score,weight,note), Verdict (label enum, confidence, per_claim
list, evidence_citations, caveats, checks_performed). Create app/
config.py with a FusionWeights section defaulting to: evidence .45,
archive_match .15, source_credibility .10, image_forensics .10,
masthead .05, ai_text .15. Stub every agent file in agents/ with its
docstring contract and a NotImplementedError. Write README.md with the
architecture ASCII diagram. Verify: `uv run pytest` collects 0 errors
and `uv run python -c "from core.schemas import Verdict"` passes.
```

## Prompt 2 — Ingestion: router + scraper

```text
Implement agents/intake_router.py and agents/scraper_agent.py.
Router: given InputPayload (text|url|image_path), detect type,
detect language (langdetect), reject empty/non-news input with a
friendly message. Scraper: async fetch with httpx (10s timeout,
1 retry), extract with trafilatura: title, body, author, publish
date, domain; return a normalized ArticleContent schema (add it to
core/schemas.py). Handle paywalls/failures gracefully by returning
partial content with a flag. Add tests with mocked HTTP using
respx. Verify: uv run pytest tests/ -k "router or scraper".
```

## Prompt 3 — OpenVINO runtime + OCR agent

```text
Implement ml/openvino_runtime.py: a ModelManager class that lazy-
loads OpenVINO IR or ONNX models, compiles with device priority
["NPU","GPU","CPU"] (catch unavailable-device errors and fall
through), exposes infer(), and unload(model_name) to free memory.
Add openvino and onnxruntime-openvino to deps. Then implement
agents/ocr_agent.py using RapidOCR's ONNX models routed through
onnxruntime with the OpenVINO execution provider (fallback: CPU EP).
Pipeline: load image → OpenCV preprocessing (grayscale, deskew via
minAreaRect, adaptive threshold, perspective correction if 4-point
document contour found) → OCR → group boxes into regions by position
and font size: masthead (top, largest), dateline (near masthead,
contains a date pattern), headline, body columns → return OCRResult.
If NPU/GPU are absent on the dev machine, everything must still run
on CPU. Include tests/fixtures/sample_clipping.png generation script
(render a fake newspaper clipping with PIL for testing). Verify:
uv run pytest tests/test_ocr_agent.py.
```

## Prompt 4 — Image forensics suite

```text
Implement the forensics/ package. ela.py: recompress JPEG at q=90,
absolute diff, normalize, return mean/max error and a heatmap array.
copymove.py: ORB keypoints + BFMatcher, flag clusters of self-matches
at consistent offsets. halftone.py: take background/whitespace patches,
2D FFT, detect periodic peaks in 60–150 lpi band typical of newsprint;
also compute local texture variance; return has_print_artifacts bool +
score. masthead.py: perceptual hash (imagehash) + SSIM against
reference images listed in data/publications.json (create it with 10
example outlets, fields: name, domain, credibility_tier 1-3, masthead_
ref, typical_date_format). Then implement agents/forensics_agent.py
that runs all four, times each, and returns a list[Signal]. Everything
CPU/OpenCV — no heavy models. Tests with synthetic images (pristine vs
pasted-region) proving copymove and ELA discriminate. Verify: uv run
pytest tests/ -k forensics.
```

## Prompt 5 — AI-text detector (honest mode)

```text
Implement agents/ai_text_detector.py with two signal sources:
(1) statistical: token-level perplexity + burstiness using a small
local GPT-2 (via transformers, exported to OpenVINO INT8 in
ml/convert/convert_detector.py — write that script too, using optimum-
intel), and (2) classifier: load a HuggingFace AI-text-detection
checkpoint (e.g. a RoBERTa-base openai-detector class model),
export+quantize likewise. Rules: refuse to score texts under 150
words (return Signal with note "too short — not scored"); output
probability with wide uncertainty band; hard-cap this Signal's
weight at config value (0.15); the note field must always include:
"stylistic signal only — AI-text detectors have known false-positive
rates and cannot prove authorship". Add config flag DETECTOR_LOCAL=
false fallback that uses simple statistics only, so the pipeline
works before models are converted. Tests: human Wikipedia paragraph
vs obviously templated LLM text → scores must differ in the right
direction. Verify: uv run pytest tests/ -k detector.
```

## Prompt 6 — Retrieval + fact-check APIs

```text
Implement agents/retrieval_agent.py. For each Claim, run in parallel
with asyncio.gather: (a) Tavily search (top 5), (b) Google Fact Check
Tools claims:search, (c) GNews query, (d) archive probe: build
queries "<headline>" + publication name + site:<publication domain>,
plus a web.archive.org availability check on the publication URL if
known. Normalize everything to EvidenceItem, deduplicate by URL,
tier sources using data/publications.json (unknown domains = tier 3),
persist to ChromaDB with all-MiniLM embeddings (add sentence-
transformers or use chromadb default embedder for now), and check the
cache first so repeated demo queries are instant/offline. Every call:
5s timeout, 1 retry, and a MissingAPIKey graceful skip so the system
degrades instead of crashing. Tests with respx mocks. Verify: uv run
pytest tests/ -k retrieval.
```

## Prompt 7 — LLM agents: claims, stance, judge

```text
Implement claim_extractor.py, stance_agent.py, temporal_agent.py and
judge_agent.py using langchain-anthropic (model from config, default
claude-sonnet-4-6) with structured output into the Pydantic schemas.
Put ALL prompts in core/prompts.py:
- CLAIM_EXTRACTION_PROMPT: decompose article into max 5 atomic
  checkable claims with entities/dates; classify genre
  (report/opinion/satire); if opinion or satire, mark claims
  checkability=low.
- STANCE_PROMPT: given one claim + up to 8 evidence snippets (each
  with an index), label each snippet SUPPORTS/REFUTES/NEUTRAL and
  cite indices; forbidden from using outside knowledge.
- JUDGE_PROMPT: given all Signals, per-claim stances, and fusion
  weights, output Verdict JSON. Hard rules in the prompt: no
  citation → UNVERIFIABLE; evidence weight dominates; ai_text signal
  may not exceed its weight cap; caveats must list every check that
  was skipped or low-confidence.
temporal_agent.py: pure Python — parse dates from claims and
evidence (dateutil), flag old-news-as-new (evidence for identical
event >90 days older than claim date) and future dates.
Also implement core/fusion.py: weighted average of Signals into a
confidence, mapping table confidence+stance-balance → verdict label,
unit-tested against 8 hand-written scenarios including "no evidence
found" → UNVERIFIABLE. Verify: uv run pytest tests/test_fusion.py.
```

## Prompt 8 — LangGraph wiring + API

```text
Implement agents/graph.py: LangGraph StateGraph with state =
VerityState (add to schemas). Nodes: router → (scraper | ocr →
forensics) → claim_extractor → retrieval → parallel(stance,
credibility, ai_text?, temporal) → fusion → judge → report.
Photo path also runs ai_text on OCR body text. Add per-node timing
into state.checks_performed. Then app/api/routes_verify.py: POST
/verify accepting JSON {text|url} or multipart image; returns the
Verdict; add GET /health. Wire app/main.py with CORS for localhost.
Also utils/logging_conf.py: structured per-agent logs (agent name,
duration, outcome) and utils/thermal.py: read CPU package temp via
psutil if available; expose guard() that async-sleeps 3s when temp
> 85C and sets ml device preference to NPU-only; no-op if sensors
unavailable. Call guard() before every local inference. Verify:
uvicorn app.main:app then curl POST /verify with a text claim
returns a Verdict JSON end-to-end (mock keys OK).
```

## Prompt 9 — Streamlit demo UI

```text
Implement ui/streamlit_app.py: three input tabs (Paste text / URL /
Upload newspaper photo). On submit, call the FastAPI endpoint and
stream stage-by-stage status (OCR done → N claims extracted → M
evidence items → verdict). Result card: big color-coded verdict
badge (green→red gradient by label), confidence bar, claim-by-claim
expander with stance per evidence item and clickable source links,
an "Artifact authenticity" panel for the photo path (masthead match,
halftone, ELA thumbnail heatmap), a transparency panel listing
checks_performed with timings, and a "Download report (PDF)" button
(reportlab: verdict, claims, citations). Add a sidebar showing
compute placement (which device each local model used) — pulls from
ml runtime — this is our hardware story for judges. Dark, clean,
minimal. Verify: streamlit run ui/streamlit_app.py works against the
running API with a sample text claim.
```

## Prompt 10 — Fixtures, E2E test, demo script

```text
Create tests/fixtures: (a) generate_fake_clipping.py producing three
PNGs — real-style clipping with halftone-textured background, a
digitally fabricated crisp clipping, and a clipping whose body is
LLM-boilerplate text; (b) tests/test_e2e.py running the full graph on
each with all external APIs mocked, asserting the fabricated one gets
lower artifact-authenticity signals. Update README.md with: setup
steps (uv sync, .env), how to convert models (ml/convert), how to run
API+UI, a 3-minute demo script (one true recent claim, one known
false claim, one photo upload), and the limitations section verbatim
from docs. Add a Makefile: make dev, make test, make demo. Verify:
uv run pytest -q all green.
```

## Prompt 11 — (Optional) offline fallback LLM

```text
Add an optional offline mode: integrate llama-cpp-python (SYCL/Vulkan
build note in README for Intel Arc) loading Qwen2.5-7B-Instruct
Q4_K_M from ml/models/. app/config.py flag LLM_BACKEND=cloud|local.
When local: claim extraction and judging use the local model with
max_tokens 512 and the same prompts; print a UI banner "Offline mode
— reduced quality". Keep default cloud. Verify config switch works
with the model file absent (clear error message, no crash).
```

\newpage

# Hackathon Execution Timeline (36-hour template)

| Hours | Milestone | Owner split (4-person team) |
|---|---|---|
| 0–2 | Prompts 0–1: scaffold, keys, CLAUDE.md; UI mock starts | All → then split |
| 2–8 | Prompt 2, 6, 7: text/URL happy path end-to-end (no photo yet) | Dev A: retrieval, Dev B: LLM agents |
| 8–14 | Prompts 3–4: OCR + forensics | Dev C (owns OpenVINO), Dev A assists |
| 14–18 | Prompt 5: AI-text detector + model conversion notebook | Dev C |
| 18–24 | Prompt 8: graph wiring, API hardening, thermal guard | Dev B |
| 24–30 | Prompt 9: UI polish, PDF report, compute-placement sidebar | Dev D (frontend) |
| 30–34 | Prompt 10: fixtures, E2E, rehearse demo twice, cache demo queries | All |
| 34–36 | Buffer: sleep for one of you, slides for another | — |

**Demo script (3 minutes):** (1) paste a fresh true headline → TRUE with citations in ~10 s; (2) paste a known viral hoax → FALSE with fact-check citations; (3) upload the fabricated clipping fixture → artifact panel lights up red (no halftone, masthead mismatch) while honestly reporting claims as UNVERIFIABLE; (4) point at the compute sidebar: "OCR ran on the NPU of this laptop at ~2 W."

# Trust, Ethics & Stated Limitations (put this slide in your deck)

1. **VERITY assists, it does not adjudicate.** Outputs are evidence summaries with confidence, meant for human judgment.
2. **AI-text detection is a weak signal.** Known false-positive risk, worse on short, translated, or regional-language text; capped at 15% weight and labeled "stylistic signal."
3. **Offline/local journalism is systematically under-indexed** — UNVERIFIABLE is a neutral outcome, never treated as FALSE.
4. **Retrieval bias**: search engines over-represent English and large outlets; we tier but cannot eliminate this.
5. **Adversarial fragility**: a determined forger can photograph a laser-printed fake on real newsprint; halftone checks then pass. Layered signals reduce but don't remove this.
6. **Privacy**: uploaded images are processed locally for forensics; only extracted text is sent to cloud APIs. Say this out loud — judges love it, and the on-device NPU makes it true.
7. **No training on user uploads**; cache is local ChromaDB, wipeable with one command.

# Stretch Ideas (mention as roadmap, build only if ahead of schedule)

- **WhatsApp bot ingress** (where Indian misinformation actually spreads) via Twilio sandbox.
- **Deepfake image/photo check** for news *photos* (not just clippings) using a small ViT detector on the NPU.
- **Regional language first-class**: Hindi OCR + IndicTrans2 translation before retrieval.
- **Browser extension** that sends the current article URL to `/verify`.
- **Community verification queue** for UNVERIFIABLE offline items.

# Summary — Why This Wins

VERITY is not "an LLM that says fake or real." It is a **transparent team of narrow agents** — retrieval, forensics, stylometry, credibility — fused with explicit weights into graded, cited verdicts, engineered specifically for the hard case everyone else ignores (**print-only news**) and tuned for the silicon in a modern AI-PC laptop (**NPU-first, cool-running**). The honesty engineering — UNVERIFIABLE as a first-class verdict, capped detector weight, a visible "what we checked" panel — is precisely what makes the output *trustable*, and it is the story your demo should tell.
