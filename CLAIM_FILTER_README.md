# Claim Check-Worthiness Filter — Change Record

Everything `agents/claim_extractor.py` and `agents/graph.py` gained when claim
extraction went from "return every sentence the LLM produced" to "return only
the sentences worth fact-checking". Written so the whole change can be
re-derived, re-applied, or reverted from this file alone.

**Scope:** `agents/claim_extractor.py`, `agents/graph.py`, `tests/fakes.py`,
`tests/test_claim_extractor.py`, `tests/test_graph.py`.

---

## 1. What the pipeline does now

```
LLM structured output  ->  N raw sentences
        |
        v
check-worthiness classifier (local, per sentence)
        |
        +--> claim.csv   : claim_id, claim, label   for ALL N (rewritten each run)
        |
        v
keep label == "Check-worthy Factual"
        |
        +-- none kept --> graph sets reject_reason -> END
        |                 verdict UNVERIFIABLE, caveat
        |                 "no factual claims found in the article"
        v
      list[Claim]  ->  retrieval -> stance -> fusion -> judge
```

The return type is unchanged: `extract_claims()` still returns `list[Claim]`
with identical element fields. Only the length differs.

---

## 2. The detector model

**Now:** `Nithiwat/mdeberta-v3-base_claimbuster` (ClaimBuster-trained, 3 classes)

| Label | Meaning | Kept? |
|---|---|---|
| `Check-worthy Factual` | verifiable and worth checking | **yes** |
| `Unimportant Factual` | true but trivial | no |
| `Non-factual` | opinion, prediction, question | no |

**Previously:** `FinnHillengass/claim-detector-deberta-v3-base` (binary
`CLAIM`/`NO_CLAIM`). It was replaced because it had the pipeline close to
backwards on real project data:

```
old=NO_CLAIM  new=Check-worthy Factual  0.896  The survey included 2,000 students from universities across India.
old=NO_CLAIM  new=Check-worthy Factual  0.992  Students who regularly review digital notes perform 18% better...
old=CLAIM     new=Non-factual           0.989  Digital note-taking will become the dominant method... five years.
old=NO_CLAIM  new=Unimportant Factual   0.740  Students can edit their notes after a class has ended.
```

Row 3 is the decisive one: the old model sent an unfalsifiable prediction into
retrieval and stance while discarding the checkable statistics around it. It
also scored 0/6 on a probe of six plainly checkable facts (e.g. "The municipal
corporation approved a new metro line on 11 July 2026" produced `NO_CLAIM` at
0.994). The replacement scored 11/11 on that same probe.

**If you swap the model again,** re-run the probe in section 6 first, then
update `_CLAIM_DETECTOR_MODEL` and the `_LABEL_*` constants together — the
label strings are model-specific, and a mismatch silently drops everything.

---

## 3. Changes in `agents/claim_extractor.py`

### 3.1 Classify once, record everything, then filter

`extract_claims` runs in this order — the CSV is written **before** filtering,
so dropped sentences stay auditable:

```python
labeled = self._label_claims(claims)     # [(Claim, label), ...]
self._write_claims_csv(labeled)          # all N rows: claim_id, claim, label
kept = [c for c, label in labeled if self._is_claim(label)]
```

Classification was previously buried inside `_write_claims_csv`. Splitting it
out into `_label_claims()` keeps it to **one classifier call per claim**,
feeding both the CSV and the filter.

### 3.2 The filter is strict

```python
_CLAIM_LABELS = frozenset({_LABEL_CHECKWORTHY.casefold()})

@staticmethod
def _is_claim(label: str) -> bool:
    """Strict: only a check-worthy factual label passes."""
    return label.strip().casefold() in _CLAIM_LABELS
```

`Unimportant Factual` and `Non-factual` both drop. To also verify
trivially-true statements, add `_LABEL_UNIMPORTANT.casefold()` to
`_CLAIM_LABELS` — that one line is the whole knob.

> An earlier revision failed *open* (unknown or unlabeled produced a keep).
> That was removed deliberately: it contradicts "only check-worthy passes".
> The dead-classifier case is handled by 3.3 instead.

### 3.3 A dead classifier raises, it does not return `[]`

A strict filter over unlabeled output drops every claim, which looks identical
to a clean run that found nothing. So:

```python
if labeled and all(label == _LABEL_UNAVAILABLE for _, label in labeled):
    raise ClaimExtractionError(
        "check-worthiness classifier could not label any claim, so none "
        "can be passed on (transformers/torch are core dependencies - "
        "repair the environment with: uv sync)"
    )
```

This is also why `transformers` + `torch` moved out of the optional
`local-ai-text` extra and into `[project.dependencies]`: a filter that
hard-fails without them cannot depend on an opt-in install. The extra is
kept as an empty no-op so `uv sync --extra local-ai-text` still resolves.

### 3.4 The classifier is injectable

`ClaimExtractor(..., claim_classifier=None)` — defaults to the lazily loaded HF
pipeline, injectable for tests. The test module's stated "fully offline"
contract was already being broken by a real model download; injection restores
it and cut the suite from ~41s to ~11s.

### 3.5 `claim.csv`

Columns `claim_id, claim, label`, from the single `_CLAIMS_CSV_COLUMNS`
constant. Opened `"w"` — **a per-run snapshot, no history by design.** A CSV
failure is caught and logged, never fatal: it is a debug artefact.

---

## 4. Changes in `agents/graph.py` — terminating the run

Raising from the extractor does **not** stop the graph: `_timed_call` catches
every exception into a `CheckLog` and returns `None`, so the run would carry on
to retrieval and fusion with an empty claim list. Termination therefore reuses
the existing `reject_reason` -> `END` short-circuit that the router already used.

```python
NO_CHECKWORTHY_CLAIMS_MESSAGE = "no factual claims found in the article"
```

In `_node_claim_extractor`, two distinct outcomes:

| Condition | `reject_reason` set to | Why kept distinct |
|---|---|---|
| `result is None` (extractor raised) | the `CheckLog` note | reporting "no factual claims" for a broken classifier points the user at the article instead of the install |
| `result == []` | `NO_CHECKWORTHY_CLAIMS_MESSAGE` | a legitimate outcome, not a malfunction |

Wiring — `claim_extractor` gained a conditional edge:

```python
def _route_after_claims(state: _GraphState) -> str:
    if state.get("reject_reason"):
        return "rejected"
    return "retrieval"

graph.add_conditional_edges(
    "claim_extractor", _route_after_claims,
    {"rejected": END, "retrieval": "retrieval"},
)
```

`run()` already converts a `reject_reason` into
`Verdict(UNVERIFIABLE, confidence=0.0, caveats=[message])`, so it needed no
change. Node statuses stay `ok` — an article with no factual claims is a
result, not a crash.

---

## 5. Latent bugs removed

| File | Bug | Impact if reintroduced |
|---|---|---|
| `claim_extractor.py` | `import torch` at top level | unused import; nothing else |
| `claim_extractor.py` | `from transformers import pipeline` at top level | **`agents/graph.py` dies at import** on any partial or broken install, and it made `_CLASSIFIER_UNAVAILABLE` unreachable dead code. Now imported lazily inside the `try`, matching what the surrounding comments always claimed. Still worth keeping lazy now that the package is a core dependency: importing torch costs seconds |
| `claim_extractor.py` | `file_exists` / `st_size` header guard in the CSV writer | dead code: `"w"` truncates before the check, so it was always true |
| `claim_extractor.py` | `@runtime_checkable` on the data-only `ArticleLike` Protocol | any `isinstance()` against it raises `TypeError` at runtime |
| `claim_extractor.py` | comments claiming "at most 5 claims" and "kept low on purpose" beside `MAX_CLAIMS_PER_ARTICLE = 100` | misleading only |

---

## 6. Verification

Full suite — 111 passing, 6 of them covering this change:

```bash
python -m pytest tests/ -q
```

The module must still import with `transformers`/`torch` unimportable - a
broken install must not take down `agents/graph.py`, which is the second bug
in section 5:

```python
# run under: python -
import builtins
real = builtins.__import__
def blocked(name, *a, **k):
    if name.split(".")[0] in {"transformers", "torch"}:
        raise ImportError(name)
    return real(name, *a, **k)
builtins.__import__ = blocked
import agents.claim_extractor as m
print("imports OK:", m.ClaimExtractor.__name__)
```

Model probe, to run before adopting any replacement detector — six checkable
facts must come back `Check-worthy Factual`, five opinions and questions must
not:

```python
from transformers import pipeline
clf = pipeline("text-classification", model="Nithiwat/mdeberta-v3-base_claimbuster")
clf("The vaccine was approved by the FDA in December 2020.", truncation=True)
# -> [{'label': 'Check-worthy Factual', 'score': 0.992}]
clf("What a beautiful day it is!", truncation=True)
# -> [{'label': 'Non-factual', 'score': 0.995}]
```

Live end-to-end results, real Groq LLM plus real classifier with `GROQ_API_KEY`
set:

- **Opinion article** — 7 sentences extracted, all 7 labeled `Non-factual`,
  `extract_claims` returned `[]`. The full pipeline stopped at
  `claim_extractor`, never reached `retrieval`, and returned verdict
  `UNVERIFIABLE` at confidence `0.0` with caveat
  `no factual claims found in the article`. All node statuses `ok`.
- **Factual report** — 6 of 6 sentences `Check-worthy Factual`, returned as
  `list[Claim]`, pipeline continued normally.

---

## 7. Known, deliberate, and unchanged

- `claim.csv` is untracked and regenerated on every run. Consider `.gitignore`.
- `MAX_CLAIMS_PER_ARTICLE = 100` is a ceiling, not a target; the filter is what
  actually limits downstream cost.
- `_LABEL_UNIMPORTANT` and `_LABEL_NON_FACTUAL` are referenced only by comments
  and by the widening knob in 3.2. They document the model's label space — keep
  them in sync with `_CLAIM_DETECTOR_MODEL`.
