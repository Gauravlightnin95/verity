"""Deterministic, LLM-free verdict math: weighted-average fusion of Signals
plus per-claim stance rollup into a fully self-sufficient Verdict.

agents.judge_agent.judge() either passes this Verdict through unchanged or
asks Claude to refine its wording - it must never contradict the hard
rules enforced here. This is what makes those rules structurally
unbreakable rather than merely prompted.

Score convention: every category score is "higher = more true/authentic/
trustworthy", 0.5 = neutral/unknown. This applies to ai_text too: a high
score means the text looks human-written (trustworthy), not AI-generated.
"""

from __future__ import annotations

from app.config import FusionWeights, settings
from core.schemas import (
    CheckLog,
    Claim,
    EvidenceItem,
    PerClaimVerdict,
    Signal,
    StanceResult,
    Verdict,
    VerdictLabel,
)

# Raw Signal.name values (as emitted by forensics/credibility/ai_text
# agents) mapped to the six weighted fusion categories in FusionWeights.
# A name absent from this table is treated as informational-only: it
# never enters the weighted sum, but a low score still surfaces as a
# caveat. Confirm with the team at schema-freeze if agents need new names.
NAME_TO_CATEGORY: dict[str, str] = {
    "archive_match": "archive_match",
    "source_credibility": "source_credibility",
    "ela": "image_forensics",
    "copymove": "image_forensics",
    "halftone": "image_forensics",
    "image_forensics": "image_forensics",
    "masthead_match": "masthead",
    "masthead": "masthead",
    "ai_text": "ai_text",
}

# Score thresholds mapping the normalized weighted score to a label.
# Tunable - validated against the scenarios in test_fusion.py.
_LABEL_THRESHOLDS: list[tuple[float, VerdictLabel]] = [
    (0.85, VerdictLabel.TRUE),
    (0.65, VerdictLabel.MOSTLY_TRUE),
    (0.40, VerdictLabel.MIXED),
    (0.25, VerdictLabel.MISLEADING),
    (0.0, VerdictLabel.FALSE),
]

_NEUTRAL = 0.5
_LOW_SCORE_CAVEAT_THRESHOLD = 0.3
_LOW_EVIDENCE_CONFIDENCE_CAP = 0.25
_SATIRE_CONFIDENCE = 0.9

# A stance only counts as much as its source is trustworthy: one tier-1
# source outweighs three unknown tier-3 pages. A claim needs at least
# _MIN_CORROBORATION of tier-weighted stance mass - i.e. one reputable
# (tier-1/2) source, or three independent unknown sites - before it can
# be called TRUE or FALSE. Random webpages repeating a claim must never
# verify it: hoaxes are widely reposted by exactly such pages.
_TIER_STANCE_WEIGHT = {1: 1.0, 2: 0.7, 3: 0.3}
_MIN_CORROBORATION = 0.7


def _category_weight(category: str, weights: FusionWeights) -> float:
    if category == "ai_text":
        return weights.ai_text_max
    return getattr(weights, category)


def _rollup_claim(
    claim: Claim,
    stance_results: list[StanceResult],
    tier_by_evidence: dict[str, int],
) -> tuple[PerClaimVerdict, float]:
    """Returns the display rollup (raw counts, per the frozen schema) plus
    this claim's tier-weighted stance score for the fusion sum."""
    relevant = [s for s in stance_results if s.claim_id == claim.claim_id]
    supporting = sum(1 for s in relevant if s.stance == "SUPPORTS")
    refuting = sum(1 for s in relevant if s.stance == "REFUTES")

    def _weight(s: StanceResult) -> float:
        return _TIER_STANCE_WEIGHT.get(tier_by_evidence.get(s.evidence_id, 3), 0.3)

    weighted_support = sum(_weight(s) for s in relevant if s.stance == "SUPPORTS")
    weighted_refute = sum(_weight(s) for s in relevant if s.stance == "REFUTES")
    total_weight = weighted_support + weighted_refute

    if total_weight == 0:
        label = VerdictLabel.UNVERIFIABLE
        score = _NEUTRAL
    else:
        raw = (weighted_support - weighted_refute + total_weight) / (2 * total_weight)
        # Thin evidence shrinks the score toward neutral instead of letting
        # a single weak source claim full certainty.
        corroboration = min(1.0, total_weight / 1.0)
        score = _NEUTRAL + (raw - _NEUTRAL) * corroboration
        if max(weighted_support, weighted_refute) < _MIN_CORROBORATION:
            label = VerdictLabel.UNVERIFIABLE  # only weakly-sourced stances
        elif weighted_refute == 0:
            label = VerdictLabel.TRUE
        elif weighted_support == 0:
            label = VerdictLabel.FALSE
        else:
            label = VerdictLabel.MIXED

    pc = PerClaimVerdict(claim_id=claim.claim_id, label=label, supporting=supporting, refuting=refuting)
    return pc, score


def _label_for_score(score: float) -> VerdictLabel:
    for threshold, label in _LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return VerdictLabel.FALSE


def fuse(
    claims: list[Claim],
    evidence: list[EvidenceItem],
    stance_results: list[StanceResult],
    signals: list[Signal],
    checks_performed: list[CheckLog],
    weights: FusionWeights | None = None,
) -> Verdict:
    weights = weights or settings.fusion_weights

    tier_by_evidence = {e.evidence_id: e.source_tier for e in evidence}
    rollups = [_rollup_claim(c, stance_results, tier_by_evidence) for c in claims]
    per_claim = [pc for pc, _ in rollups]
    # "Verified" means at least one claim reached a real, adequately-sourced
    # stance - not merely that some snippet was judged.
    has_any_stance = any(pc.label != VerdictLabel.UNVERIFIABLE for pc in per_claim)

    category_scores: dict[str, list[float]] = {}
    stance_mean = _NEUTRAL
    if rollups:
        # Per-claim mean (not a raw supports/refutes aggregate) so that an
        # UNVERIFIABLE claim measurably drags the score toward neutral
        # instead of being invisible to a strongly-supported sibling claim.
        claim_scores = [score for _, score in rollups]
        stance_mean = sum(claim_scores) / len(claim_scores)
        category_scores["evidence_stance"] = [stance_mean]

    caveats: list[str] = []
    for sig in signals:
        category = NAME_TO_CATEGORY.get(sig.name)
        if sig.weight <= 0:
            caveats.append(f"{sig.name} not scored: {sig.note}" if sig.note else f"{sig.name} not scored")
        elif sig.score < _LOW_SCORE_CAVEAT_THRESHOLD:
            caveats.append(f"{sig.name}: {sig.note}" if sig.note else sig.name)
        if category is not None and sig.weight > 0:
            score = sig.score
            if category == "source_credibility":
                # Credibility measures how trustworthy the evidence is - it
                # must AMPLIFY whatever the evidence says, never fight it.
                # Credible sources refuting a claim push the score DOWN
                # (more decisively false); junk sources pull toward neutral.
                # Without this, high credibility perversely dragged refuted
                # claims back up toward MIXED.
                if stance_mean > _NEUTRAL:
                    score = _NEUTRAL + (sig.score - _NEUTRAL)
                elif stance_mean < _NEUTRAL:
                    score = _NEUTRAL - (sig.score - _NEUTRAL)
                else:
                    score = _NEUTRAL  # no stance direction to amplify
            category_scores.setdefault(category, []).append(score)

    weighted_sum = 0.0
    weight_total = 0.0
    for category, scores in category_scores.items():
        mean_score = sum(scores) / len(scores)
        cat_weight = _category_weight(category, weights)
        weighted_sum += mean_score * cat_weight
        weight_total += cat_weight
    normalized_score = weighted_sum / weight_total if weight_total > 0 else _NEUTRAL
    confidence = round(min(abs(normalized_score - _NEUTRAL) * 2, 1.0), 2)

    for cl in checks_performed:
        if cl.status in ("skipped", "failed"):
            note = f": {cl.note}" if cl.note else ""
            caveats.append(f"{cl.agent_name} check {cl.status}{note}")
    for pc in per_claim:
        if pc.label == VerdictLabel.UNVERIFIABLE:
            if pc.supporting + pc.refuting > 0:
                caveats.append(
                    f"Claim {pc.claim_id} unverifiable: only low-credibility "
                    "sources address it - not enough corroboration to call it"
                )
            else:
                caveats.append(f"Claim {pc.claim_id} unverifiable: no evidence found")

    is_satire = bool(claims) and all(c.genre in ("opinion", "satire") for c in claims)
    if is_satire:
        label = VerdictLabel.SATIRE_OPINION
        confidence = _SATIRE_CONFIDENCE
    elif not evidence:
        label = VerdictLabel.UNVERIFIABLE
        confidence = min(confidence, _LOW_EVIDENCE_CONFIDENCE_CAP)
        caveats.append("No evidence found for any claim.")
    elif claims and not has_any_stance:
        # Evidence was found but no claim reached an adequately-sourced
        # supporting/refuting stance (all NEUTRAL, or only weak tier-3
        # sources). That is "we could not verify", not "the evidence
        # conflicts": MIXED here would falsely imply contradictory sources.
        label = VerdictLabel.UNVERIFIABLE
        confidence = min(confidence, _LOW_EVIDENCE_CONFIDENCE_CAP)
        caveats.append(
            "Evidence was found but it is not sufficient to verify or "
            "falsify the claims - unable to verify either way."
        )
    else:
        label = _label_for_score(normalized_score)
        # An article where half or more of the claims could not be verified
        # must not be called TRUE/MOSTLY_TRUE just because its peripheral
        # claims checked out - the classic fake-news pattern wraps one
        # unverifiable central claim in true filler.
        unverified = sum(1 for pc in per_claim if pc.label == VerdictLabel.UNVERIFIABLE)
        if per_claim and unverified * 2 >= len(per_claim) and label in (
            VerdictLabel.TRUE,
            VerdictLabel.MOSTLY_TRUE,
        ):
            label = VerdictLabel.MIXED
            caveats.append(
                f"{unverified} of {len(per_claim)} claims could not be verified - "
                "capping the verdict at MIXED even though the remaining claims check out."
            )

    return Verdict(
        label=label,
        confidence=confidence,
        per_claim=per_claim,
        evidence_citations=evidence,
        signals=signals,
        caveats=caveats,
        checks_performed=checks_performed,
    )
