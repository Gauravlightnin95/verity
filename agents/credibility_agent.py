"""Source credibility auditor (owned by Member B).

Purpose: pure-Python scoring of retrieved evidence's source trustworthiness,
weighted by source_tier from data/publications.json.
Input: evidence (list[EvidenceItem]). Output: a single Signal
(core.schemas) named "source_credibility".
"""

from core.schemas import EvidenceItem, Signal

# tier 1 = fact-checkers / top outlets, tier 2 = normal news, tier 3 = unknown
TIER_SCORES = {1: 1.0, 2: 0.6, 3: 0.25}
FUSION_WEIGHT = 0.10  # per Member A's fusion table


def score_credibility(evidence: list[EvidenceItem]) -> Signal:
    if not evidence:
        return Signal(
            name="source_credibility", score=0.0, weight=FUSION_WEIGHT,
            note="no evidence found - source credibility unknown",
        )

    score = sum(TIER_SCORES.get(e.source_tier, 0.25) for e in evidence) / len(evidence)
    counts = {t: sum(1 for e in evidence if e.source_tier == t) for t in (1, 2, 3)}
    return Signal(
        name="source_credibility",
        score=round(score, 3),
        weight=FUSION_WEIGHT,
        note=f"{counts[1]} tier-1, {counts[2]} tier-2, {counts[3]} tier-3 sources",
    )
