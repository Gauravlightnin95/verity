---
title: "Project VERITY — Team Work Split (Student-Friendly Edition)"
subtitle: "Who builds what, explained in plain language — learn the concepts as you build"
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
  - \fancyhead[L]{\small Project VERITY — Team Work Split}
  - \fancyhead[R]{\small Student-Friendly Edition}
  - \fancyfoot[C]{\thepage}
  - \usepackage[dvipsnames]{xcolor}
---

\newpage

# Read This First — The Big Idea (5 minutes, all four of you)

We are four equal teammates building one project: **VERITY**, a system that checks whether a piece of news is real or fake. You give it text, a link, or a photo of a newspaper, and it answers with a verdict like TRUE / FALSE / UNVERIFIABLE, along with proof.

The project has four natural areas, so each of us takes one:

| Member | Area | One-line job description |
|---|---|---|
| **A** | Core & Glue | Defines the common data shapes everyone uses, and connects all parts into one pipeline |
| **B** | Internet Work | Fetches articles, searches the web for evidence, rates how trustworthy sources are |
| **C** | Image Work | Reads text out of newspaper photos and checks if the photo is genuine or edited |
| **D** | AI Brain & Screen | Talks to the AI model (Claude) for reasoning, and builds the app screen users see |

No boss here — Member A's job is "connecting", not "commanding". Merging code is a task like any other task.

## How can four people build separately without breaking each other's code?

This is the most important concept of the whole document, so let's learn it properly.

**The problem:** If I write my code assuming your function returns a list, and you actually return a dictionary, our code explodes when we join it on the last day. Classic hackathon death.

**The solution: agree on the "shape" of data FIRST, then build separately.**

Think of it like a food delivery system. The restaurant (Member C) and the delivery app (Member D) never talk to each other directly. They only agree on one thing: *the box format* — every order comes in a standard box with a label saying what's inside. As long as everyone packs and reads the standard box, the restaurant can change its kitchen and the app can change its screens, and nothing breaks.

In our project, the "standard boxes" are defined in **one Python file: `core/schemas.py`**, written by Member A in the first 3 hours, reviewed by all four of us together once, and then **frozen** (nobody changes it without a group vote).

> **New word — Schema:** a schema is just a written-down description of what a piece of data must look like. Example: "an Evidence item must have a `snippet` (text), a `url` (text), and a `source_tier` (number 1, 2 or 3)." That's it. Nothing scary.

> **New word — Pydantic:** a popular Python library that turns a schema into a Python class. Its superpower: if anyone creates an Evidence item with a missing url or a source_tier of "banana", Pydantic instantly throws a clear error. So wrong-shaped data gets caught the second it is created, not on demo day. You write it like a normal class:
>
> ```python
> from pydantic import BaseModel
>
> class EvidenceItem(BaseModel):
>     snippet: str
>     url: str
>     source_tier: int
> ```

## How do I build my part when your part doesn't exist yet?

With **mocks**.

> **New word — Mock:** a fake, hard-coded version of a function. It doesn't do any real work — it just instantly returns realistic-looking example data in the correct shape. Example: the real `run_ocr()` will take a photo and extract text using an ML model (slow, complex). The mock `run_ocr()` ignores the photo and returns the same hard-coded text every time (instant, dumb, perfect for testing).

Member A writes a mock for *every* function in the project on day one. Then each of us replaces only *our own* mock with real code, while testing against everyone else's mocks. On integration day, swapping a mock for the real thing is a one-line change — because both return the exact same data shape.

## What each member must hand over (same format for everyone)

1. **Your code**, only inside your assigned folders, with one clearly-named main function others will call.
2. **Sample output files** — at least 2 real example outputs of your function, saved as `.json` files in `tests/fixtures/contracts/`. *(JSON is just a text format for structured data — the examples later in this doc are JSON. These files prove your output matches the agreed shape, and teammates use them as test data.)*
3. **Tests** — small `pytest` scripts that automatically check your functions work. *(pytest is a tool: you write tiny functions starting with `test_`, run `pytest` in the terminal, and it tells you pass/fail. Internet calls are faked in tests so tests run offline — you'll see how below.)*
4. **A 10-line `PART_README.md`** in your folder: how to run your part, what API keys it needs, what it can't do yet.

## How we share code: Git in 60 seconds

We use one shared GitHub repository. Each member works on their own **branch** (a parallel copy of the code — like editing your own copy of a shared Google Doc, then merging your changes in when ready). Branch names: `part-a-core`, `part-b-internet`, `part-c-image`, `part-d-brain-ui`. When your piece works, you open a **Pull Request (PR)** — a "please merge my changes" request that the others can see and comment on. We merge into a shared `dev` branch at the checkpoints listed at the end of this document.

\newpage

# Member A — Core & Glue (schemas, pipeline, final verdict math)

**Your learning topics this weekend:** Pydantic, what an API is, LangGraph basics, weighted averages.

**Your folders:** `core/`, `agents/graph.py`, `agents/judge_agent.py`, `app/`, `mocks/`, `tests/`.

## A.1 — Write `core/schemas.py` (do this FIRST, hours 0–3)

Define every data shape the team uses, as Pydantic classes. Keep them boring and obvious:

- `InputPayload` — what the user sent us: `input_type` ("text" / "url" / "image") plus the text, url, or image path.
- `ArticleContent` — a scraped article: title, body, author, publish date, website domain.
- `OCRRegion` / `OCRResult` — text read from a photo, split into parts of a newspaper page: masthead (the paper's name at the top), dateline (the printed date), headline, body. Each part has the text, a confidence number 0–1 ("how sure the reader is"), and a bbox — the rectangle's pixel coordinates on the image.
- `Claim` — one single checkable statement pulled out of the article, e.g. *"The metro line was approved on 11 July."* An article becomes a list of up to 5 claims.
- `EvidenceItem` — one piece of proof found on the internet: a snippet (short quote-like summary), its url, the source name, and `source_tier` (1 = very trustworthy like a government fact-checker, 2 = normal news site, 3 = unknown random website).
- `StanceResult` — for one claim and one evidence item: does the evidence SUPPORT it, REFUTE it (contradict it), or is it NEUTRAL (related but doesn't say)?
- `Signal` — a generic "clue" with a score 0–1 and a weight (how much this clue matters). Example: the halftone check from Member C produces a Signal.
- `Verdict` — the final answer: a label (TRUE / MOSTLY_TRUE / MIXED / MISLEADING / FALSE / UNVERIFIABLE / SATIRE_OPINION), a confidence 0–1, per-claim results, the evidence citations, and `caveats` (honest warnings, e.g. "we found no coverage for claim 3").

After writing it, get everyone on a 15-minute call, walk through it, fix anything confusing, then freeze it.

## A.2 — Write `mocks/` (hour 3, right after schemas)

One fake function per teammate deliverable, returning realistic hard-coded objects. This unblocks the whole team — Member D can build the entire app screen against your mocks before anyone writes real logic.

## A.3 — Write the verdict math: `core/fusion.py`

This is honestly just a **weighted average**, like calculating a course grade where the final exam is worth 45% and homework is worth 10%.

Every check in our system produces a Signal (score 0–1). Fusion multiplies each score by its weight and adds them up:

| Clue (Signal) | Weight | Meaning |
|---|---|---|
| Evidence stance (do trusted sources support or refute the claims?) | 45% | The main thing |
| Archive match (did we find this exact article in the paper's official archive?) | 15% | Proves the clipping is real |
| Source credibility | 10% | Is the claimed publication reputable? |
| Image forensics (edited photo?) | 10% | From Member C |
| Masthead match | 5% | From Member C |
| AI-text detector | 15% **maximum** | From Member D — capped on purpose, see D.3 |

Extra rules (write them as simple `if` statements): if we found **zero** evidence, the label can never be better than UNVERIFIABLE. If the article is satire or opinion, label is SATIRE_OPINION. Write 8 small tests: invent 8 situations by hand ("all evidence supports" → TRUE, "no evidence at all" → UNVERIFIABLE, ...) and assert your function returns the right label for each.

## A.4 — The Judge: `agents/judge_agent.py`

One call to the Claude API that takes the fusion numbers + the evidence snippets and writes the human-readable explanation. The prompt must contain this rule, word for word: *"You may only state facts that appear in the provided snippets. If you cannot cite a snippet, the verdict is UNVERIFIABLE."* This stops the AI from making things up (called **hallucination**).

All AI prompts for the whole team live in one file, `core/prompts.py`, that you maintain — teammates send you their prompt text and you paste it in. One file = easy to review, easy to improve.

## A.5 — The pipeline and the API: `agents/graph.py` + `app/`

> **New word — API (Application Programming Interface):** a way for two programs to talk over the network. Our backend runs a small web server; Member D's screen sends it a request like *"POST /verify with this text"* and gets the Verdict back as JSON. **FastAPI** is the Python library that makes building this server easy — one function + one decorator line per endpoint.

> **New word — LangGraph:** a library for connecting steps into a flowchart that runs itself. Each step ("node") is just a Python function. You declare the arrows: router → scraper → claims → retrieval → ... → verdict. LangGraph handles running steps in order, passing the shared data along, and running independent steps **in parallel** (at the same time — e.g. the credibility check and the stance check don't depend on each other, so they run together and save time).

Each node is a thin wrapper: take what you need from the shared state → call a teammate's function (mock at first, real later) → put the result back in state. Wrap every call in try/except so that **if one part fails, the pipeline logs it and continues with a weaker verdict instead of crashing**. That log entry (`CheckLog`: which agent, ok/skipped/failed, how long it took) also powers the "what we checked" panel in the app — our transparency feature.

## Your handover outputs

- Frozen `core/schemas.py` + `mocks/` by hour 3 ← your most important deliverable
- A running server by hour 8 that returns a complete (mock-data) Verdict — so Member D has something live to build against
- Sample file `verdict_sample.json` — the final output shape of the whole project:

```json
{
  "label": "MISLEADING",
  "confidence": 0.71,
  "per_claim": [
    {"claim_id": "c1", "label": "TRUE",  "supporting": 3, "refuting": 0},
    {"claim_id": "c2", "label": "FALSE", "supporting": 0, "refuting": 2}
  ],
  "evidence_citations": [
    {"evidence_id": "e1", "claim_id": "c1",
     "snippet": "The ministry confirmed on Monday that...",
     "url": "https://pib.gov.in/...", "source_name": "PIB Fact Check",
     "source_tier": 1, "retrieved_at": "2026-07-14T10:22:00Z"}
  ],
  "signals": [
    {"name": "evidence_stance", "score": 0.62, "weight": 0.45,
     "note": "3 snippets support, 2 refute across claims"},
    {"name": "ai_text", "score": 0.58, "weight": 0.15,
     "note": "stylistic signal only - detectors can be wrong"}
  ],
  "caveats": ["Image checks skipped: text input",
              "Claim c3 unverifiable: no coverage found"],
  "checks_performed": [
    {"agent_name": "retrieval", "status": "ok", "duration_ms": 2140,
     "note": "12 evidence items, 4 from cache"}
  ]
}
```

*(Reading tip: "MISLEADING with 0.71 confidence, because claim 1 is true but claim 2 is false — here are the sources, here's every clue with its weight, and here's what we honestly couldn't check.")*

\newpage

# Member B — Internet Work (fetching, searching, source trust)

**Your learning topics:** HTTP requests, async programming, web scraping, caching.

**Your folders:** `agents/intake_router.py`, `agents/scraper_agent.py`, `agents/retrieval_agent.py`, `agents/credibility_agent.py`, `agents/temporal_agent.py`, `data/`.

**Your one commandment: never crash — always degrade.** The internet WILL fail during the demo: a site will time out, an API key will hit its limit. Your code must treat every failure as "okay, skip that source, note it in the log, return what we have." An exception that kills the pipeline is the only unforgivable bug in your area.

## B.1 — Intake router: `route(payload) -> RoutedInput`

Figure out what the user gave us: plain text, a URL, or an image. Detect the language (use the `langdetect` library — one function call). If the input is empty or obviously not news ("hello"), politely reject it with a reason instead of wasting the pipeline's time.

## B.2 — Scraper: `scrape(url) -> ArticleContent`

> **New word — Scraping:** downloading a web page and extracting the useful part (the article) while throwing away menus, ads, and cookie banners. We don't parse HTML by hand — the library **trafilatura** does it in one call: give it the page, get back title, body text, author, and date.

Download with **httpx** (a modern requests-like library) with a 10-second timeout and one retry. If the page is paywalled or broken, return whatever partial content you got with `partial=true` — never raise an error.

## B.3 — Evidence retrieval: `retrieve(claims, publication_hint) -> list[EvidenceItem]` (your centerpiece)

For each claim, ask several sources at once:

1. **Tavily** — a search API built for AI apps: send a query, get clean text results (top 5).
2. **Google Fact Check Tools API** — a free Google database of articles by professional fact-checkers. If someone already debunked this claim, this finds it instantly.
3. **GNews API** — recent news articles matching the claim.
4. **Archive probe** — if Member C's masthead check guessed the publication (you receive that guess as `publication_hint` through the shared pipeline state — you never import their code), search that paper's own website and web.archive.org for the headline. Finding it there proves the clipping really came from that paper.

> **New word — async / `asyncio.gather`:** normally Python does one thing at a time, so 4 searches × 2 seconds = 8 seconds of waiting. `async` lets Python fire all 4 requests simultaneously and wait for them together → total time ≈ the slowest single call (~2 s). You write `await asyncio.gather(search1(), search2(), search3(), search4())`. That's the whole trick.

Then clean up: remove duplicate URLs, and stamp each item's `source_tier` by looking its domain up in `data/publications.json` (below); unknown domains = tier 3.

> **New word — Caching:** saving results so repeats are instant. Store every evidence item in **ChromaDB** (a small local database, runs as a folder on disk — no server setup). Before searching the internet, check the cache first. Practical demo benefit: rehearse the demo once at home → all demo queries are cached → on stage the answers appear instantly, even if the venue Wi-Fi is dying.

Missing API key? Skip that provider quietly and log it. Your part must run (in degraded form) with **zero** keys.

## B.4 — Two small pure functions

- `score_credibility(evidence) -> Signal` — average trustworthiness of the sources found, weighted by tier. Pure math, no internet.
- `check_temporal(claims, evidence) -> Signal` — compare dates (use the `dateutil` library to parse messy date strings). Catches a classic fake-news trick: a REAL old story reshared as if it happened today. If evidence for the same event is much older (90+ days) than the claim pretends, flag it.

## B.5 — `data/publications.json` — the trust list

A hand-made list of at least 20 outlets: name, website domain, credibility tier (1–3), typical printed date format, and a filename of the paper's masthead reference image (Member C uses that image — agree on filenames with them once). Include Indian national papers, PIB Fact Check, some regional outlets, and 2–3 known misinformation sites as tier-3 examples.

## Your handover outputs

Sample `.json` files for each function's output, plus tests. **Your tests must pass with no internet:** use the **respx** library, which intercepts your httpx calls during tests and returns fake responses you define — so tests are fast, free, and work offline.

```json
[
  {"evidence_id": "e1", "claim_id": "c1",
   "snippet": "The ministry confirmed on Monday that...",
   "url": "https://pib.gov.in/...", "source_name": "PIB Fact Check",
   "source_tier": 1, "published_date": "2026-07-13",
   "retrieved_at": "2026-07-14T10:22:00Z"},
  {"evidence_id": "e2", "claim_id": "c1",
   "snippet": "Contrary to viral posts, officials denied...",
   "url": "https://factly.in/...", "source_name": "Factly",
   "source_tier": 1, "published_date": "2026-07-13",
   "retrieved_at": "2026-07-14T10:22:03Z"}
]
```

\newpage

# Member C — Image Work (reading photos, spotting fakes)

**Your learning topics:** OCR, OpenCV basics, what an FFT tells you, running ML models locally.

**Your folders:** `agents/ocr_agent.py`, `agents/forensics_agent.py`, `forensics/`, `ml/`, `tests/fixtures/images/`.

Your area is the project's star feature — checking a **photo of a printed newspaper**, which most fact-check tools can't handle at all.

## C.1 — Local model runner: `ml/openvino_runtime.py`

> **New word — OpenVINO:** Intel's free toolkit for running ML models efficiently on ordinary laptops. Modern Intel laptops have three chips that can run models: the CPU (normal processor), the iGPU (built-in graphics), and the **NPU** (Neural Processing Unit — a small chip designed only for AI math, which runs models using very little power, so the laptop stays cool and quiet).

Write a small `ModelManager` class that: loads a model file only when first needed (**lazy loading** — don't fill up RAM at startup), tries devices in order NPU → GPU → CPU (if a device doesn't exist, catch the error and try the next — **this matters because teammates without an NPU must still be able to run your code on plain CPU**), runs predictions, can unload a model to free memory, and remembers which device it used (the app shows this — it's our "runs on-device" story for the judges). Member D will also load their model through your class, so share its usage in your PART_README by hour 6.

## C.2 — OCR: `run_ocr(image_path) -> OCRResult`

> **New word — OCR (Optical Character Recognition):** turning a photo of text into actual text characters. We use **RapidOCR**, a ready-made open-source OCR model — you don't train anything, you just run it.

Photos of newspapers are tilted, shadowed, and curved, so clean the image first with **OpenCV** (the standard Python image library — every operation below is a couple of lines):

1. Convert to grayscale.
2. **Deskew** — detect the tilt angle and rotate the page straight.
3. **Adaptive threshold** — make text crisply black-on-white even with uneven lighting.
4. **Perspective correction** — if the photo was taken at an angle, find the page's four corners and mathematically "flatten" it, like a scanner app does.

Then run OCR and group the detected text boxes into newspaper regions by their position and size: the **masthead** (paper's name — top of page, biggest text), the **dateline** (contains a date pattern, near the top), the **headline**, and the **body** columns. Return an `OCRResult` with each region's text, confidence, and box coordinates.

## C.3 — Forensics: `run_forensics(image_path, masthead_text) -> list[Signal]`

Four independent checks, each producing one Signal (score 0–1). Explanations included — you can repeat these to the judges:

- **ELA — Error Level Analysis** (`forensics/ela.py`). When you save a JPEG, every part of the image gets compressed equally. If someone later pastes a fake headline onto the image and re-saves it, the pasted part has been compressed a *different number of times* than the rest. Trick to reveal it: re-save the image at 90% quality and subtract it from the original — edited regions light up brighter in the difference image. Save that difference as a small heatmap PNG; the app displays it.
- **Copy-move detection** (`forensics/copymove.py`). Forgers often clone one part of an image onto another (e.g., copying background to hide something). Use ORB (an OpenCV feature detector) to find distinctive points, then check if groups of points inside the image match *each other* at a consistent offset — that means a region was duplicated.
- **Halftone check** (`forensics/halftone.py`) — the cleverest one. Real printed newspapers are made of thousands of tiny regularly-spaced ink dots (look at any newspaper photo with a magnifier — that dot pattern is called **halftone**). A genuine *photo of print* shows this periodic dot texture; a fake "clipping" made in an image editor is perfectly smooth. Detection: take patches of the background and apply an **FFT** (Fast Fourier Transform — a mathematical lens that reveals repeating patterns in an image as sharp peaks; regular dot grids produce unmistakable peaks, smooth digital images don't). Peaks in the typical newsprint range → real print. No peaks → suspicious.
- **Masthead match** (`forensics/masthead.py`). Compare the masthead region against the reference logo images in Member B's `publications.json`, using a perceptual hash plus SSIM (two standard "how visually similar are these images" scores, both one-liners from libraries). Also output your best `publication_guess` — that string travels through the pipeline to Member B, who uses it to search that paper's archive.

## C.4 — Test images: `tests/fixtures/images/generate_fixtures.py`

A script (using PIL, the Python image library) that draws three fake newspaper clippings: (a) realistic — with a noisy halftone-style textured background, (b) fabricated — perfectly crisp digital text, (c) one whose body text is generic AI-sounding filler (Member D tests their detector on it). **These three images ARE our demo assets. Treat them as seriously as your code.**

## Your handover outputs

Sample outputs + the three PNGs + a test that proves discrimination: the fabricated image must score clearly worse on halftone and ELA than the realistic one.

```json
{
  "regions": [
    {"region_type": "masthead", "text": "The Daily Example",
     "confidence": 0.96, "bbox": [40, 22, 980, 110]},
    {"region_type": "dateline", "text": "New Delhi | 12 July 2026",
     "confidence": 0.91, "bbox": [40, 118, 420, 150]},
    {"region_type": "headline", "text": "City Approves New Metro Line",
     "confidence": 0.94, "bbox": [40, 160, 980, 230]},
    {"region_type": "body", "text": "The municipal corporation on Friday...",
     "confidence": 0.88, "bbox": [40, 240, 500, 900]}
  ],
  "full_text": "...", "language": "en",
  "device_used": "NPU", "duration_ms": 1840
}
```

```json
[
  {"name": "halftone", "score": 0.18, "weight": 0.10,
   "note": "no print dot pattern found - image likely made digitally",
   "extras": {"has_print_artifacts": false}},
  {"name": "masthead_match", "score": 0.35, "weight": 0.05,
   "note": "closest match 'The Daily Example', similarity 0.35 - below 0.7 threshold",
   "extras": {"publication_guess": "The Daily Example"}}
]
```

\newpage

# Member D — AI Brain & Screen (LLM calls, AI-text detector, the app UI)

**Your learning topics:** prompting an LLM through an API, structured outputs, why AI-text detectors are unreliable, Streamlit.

**Your folders:** `agents/claim_extractor.py`, `agents/stance_agent.py`, `agents/ai_text_detector.py`, `ml/convert/`, `ui/`.

## D.1 — Claim extractor: `extract_claims(article) -> list[Claim]`

One call to the Claude API. The prompt asks: break this article into at most 5 short, individually checkable factual claims, list the entities (people/places/organizations) and dates in each, and classify the article's genre — news report, opinion, or satire (opinion and satire get `checkability: low`, because you can't fact-check "the government is doing a great job").

> **New word — Structured output:** normally an LLM replies with free-flowing text, which is painful to parse. Structured output means telling the API "your answer MUST be JSON matching this exact schema" — the library (`langchain-anthropic`) enforces it, and you get back ready-made `Claim` objects instead of a paragraph you'd have to dissect.

## D.2 — Stance agent: `stance(claim, evidence) -> list[StanceResult]`

For one claim, show the model up to 8 evidence snippets, each with a number. Ask: for each snippet, does it SUPPORT, REFUTE, or stay NEUTRAL about the claim — and quote the snippet number in a one-line reason. The critical rule to write in the prompt: *"Judge ONLY from the snippets. Do not use your own knowledge."* Otherwise the model answers from memory instead of from our evidence — which defeats the whole system.

## D.3 — AI-text detector: `detect_ai_text(text) -> Signal` (the honesty module)

Can we tell if the OCR'd article text was written by an AI? *Partially*, using two kinds of clues:

- **Statistical clues.** Perplexity = how "surprised" a language model is by each next word. AI text tends to pick predictable words → low surprise. Burstiness = variety in sentence length; humans mix long and short sentences, AI is more uniform. Compute both by running the text through a tiny local GPT-2 model.
- **A classifier.** A small pre-trained model (RoBERTa-type) that was fine-tuned on examples of human vs AI writing and outputs a probability. Download from Hugging Face; the script in `ml/convert/` shrinks it (quantization = storing the model's numbers in 8-bit instead of 32-bit — 4x smaller and faster, barely less accurate) and Member C's ModelManager runs it locally.

**Now the honesty part — this is a feature, not a weakness, and we will say it on stage:** these detectors are known to be unreliable. They wrongly flag human writing (especially non-native English and translated text), and fail on short texts. Even OpenAI shut down its own detector for poor accuracy. So we enforce three rules in code: (1) refuse to score texts under 150 words, (2) this Signal's weight is hard-capped at 15% so it can never decide a verdict alone, (3) its note field always carries the disclaimer: *"stylistic signal only — AI-text detectors have known false-positive rates and cannot prove authorship."* Also add a config switch `DETECTOR_LOCAL=false` that uses only the statistical clues — so your part works even before the model conversion is done.

## D.4 — The app screen: `ui/streamlit_app.py`

> **New word — Streamlit:** a Python library that turns a script into a web app with zero HTML/CSS/JavaScript. `st.button("Check")`, `st.file_uploader(...)` — each line is a UI element. Perfect for hackathons.

Your app talks ONLY to Member A's server (`POST /verify`) — which is live on mock data from hour 8, so you can build the complete UI on day one without waiting for anyone.

Screens to build: three input tabs (paste text / paste URL / upload photo) → progress messages as stages finish ("Reading image... 4 claims found... 11 evidence items...") → result card with a color-coded verdict badge and confidence bar → per-claim expandable sections showing each evidence snippet, its stance, and a clickable source link → for photos, an "Is this clipping genuine?" panel (masthead match, halftone result, the ELA heatmap image from Member C) → a transparency panel listing every check with its time and status (straight from `checks_performed`) → a sidebar showing which chip each local model ran on (`device_used`) → a "Download report as PDF" button (use the `reportlab` library: verdict + claims + citations on one page).

## Your handover outputs

Sample `.json` files + UI screenshots + tests (Claude API calls are faked in tests, same respx idea as Member B; detector test: a human-written Wikipedia paragraph vs an obviously template-like AI paragraph must score in the right direction).

```json
[
  {"claim_id": "c1",
   "text": "The municipal corporation approved a new metro line on 11 July 2026",
   "entities": ["municipal corporation", "metro line"],
   "location": "New Delhi", "event_date": "2026-07-11",
   "checkability": "high", "genre": "report"},
  {"claim_id": "c2",
   "text": "The project will be completed within 18 months",
   "entities": ["metro line"], "location": "New Delhi",
   "event_date": null, "checkability": "medium", "genre": "report"}
]
```

```json
[
  {"claim_id": "c1", "evidence_id": "e1", "stance": "SUPPORTS",
   "rationale": "Snippet 1 reports the same approval and the same date"},
  {"claim_id": "c1", "evidence_id": "e4", "stance": "NEUTRAL",
   "rationale": "Snippet 4 discusses the metro but not the approval"}
]
```

\newpage

# Putting It All Together — The Integration Plan

## The mental model

Think of the pipeline as a relay race where the baton is a shared data object (`VerityState`) that collects everything: the input, then the article, then the claims, then the evidence, then the signals, and finally the verdict. Each member's function is one runner: it takes the baton, adds its piece, passes it on. Nobody imports anybody else's code — for example, Member C's `publication_guess` reaches Member B simply because C wrote it into the baton and B reads it from there. Member A owns the racetrack (the LangGraph flowchart) and the swap of mock runners for real runners.

## Sync checkpoints (the only times we must all be aligned)

| Hour | Checkpoint | It passes when... |
|---|---|---|
| 3 | **Schema freeze** | 15-min group call, everyone has read `schemas.py`, no open questions |
| 8 | **Mock milestone** | The server returns a full (fake-data) Verdict; Member D starts the UI on it |
| 12 | **Fixture exchange** | Everyone's sample `.json` files are merged and pass the shape-checking tests |
| 18 | **Text path real** | B's retrieval + D's claims/stance merged → typing a real headline gives a REAL verdict |
| 24 | **Photo path real** | C's OCR + forensics merged; D's detector merged |
| 28 | **Full test green** | The end-to-end test runs all three fixture images successfully |
| 30–34 | **Freeze & rehearse** | Only bug fixes allowed; run the 3-minute demo twice; demo queries cached |

Why text path before photo path? It needs no local ML models, so it becomes a working demo early — if everything else caught fire, we'd still have a presentable project by hour 18. The photo path then only *adds* to something already working.

## Team rules (agree on these before writing any code)

1. Merges into the shared branch happen only at checkpoints, and only after the shape-checking tests pass.
2. `schemas.py` and `prompts.py` change only by group agreement (a 5-minute vote in the group chat), and Member A then updates all mocks to match.
3. Every real module keeps its mock available behind a settings flag. **This is our demo insurance:** if any part breaks an hour before judging, we flip that one part back to its mock and the demo still runs end to end.
4. Stuck for more than 45 minutes? Post in the group chat. Debugging alone past that point is how hackathon hours disappear.

## Definition of done, per member

- **A:** server returns proper Verdicts for all 3 input types; fusion tests green; end-to-end test green; README explains how to run everything.
- **B:** with real API keys, a fresh true headline returns 4+ evidence items in under ~5 s; with zero keys, the pipeline still finishes (verdict UNVERIFIABLE, no crash).
- **C:** the fabricated fixture image visibly scores worse on authenticity than the realistic one; OCR correctly reads all four regions of both.
- **D:** the full demo script runs through the UI including the PDF download; the detector refuses short texts with the correct honesty note.

## One last thing

Every "new word" box in this document is a concept you can now explain to a judge in one sentence. That's the real prize of building it this way — by Sunday night, all four of you will have personally used schemas, mocks, async requests, caching, OCR, image forensics, FFTs, LLM structured outputs, and on-device inference. Good luck, and cache your demo queries.
