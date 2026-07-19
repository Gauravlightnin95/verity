"""8 hand-written fusion scenarios. Pure and LLM-free: no network, no mocks
package, no LangGraph - just core.fusion.fuse() against schema objects."""

from datetime import datetime, timezone

from core.fusion import fuse
from core.schemas import Claim, EvidenceItem, Signal, StanceResult, VerdictLabel

_NOW = datetime(2026, 7, 14, tzinfo=timezone.utc)


def _claim(claim_id="c1", genre="report") -> Claim:
    return Claim(claim_id=claim_id, text=f"claim {claim_id}", genre=genre)


def _evidence(evidence_id, claim_id, tier=1) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        claim_id=claim_id,
        snippet="snippet",
        url="https://example.com",
        source_name="Example",
        source_tier=tier,
        retrieved_at=_NOW,
    )


def _stance(claim_id, evidence_id, label) -> StanceResult:
    return StanceResult(claim_id=claim_id, evidence_id=evidence_id, stance=label)


def test_1_all_supporting_high_credibility_is_true():
    claims = [_claim("c1")]
    evidence = [_evidence("e1", "c1")]
    stances = [_stance("c1", "e1", "SUPPORTS")]
    signals = [Signal(name="source_credibility", score=0.9, weight=0.10, note="tier-1 sources")]

    verdict = fuse(claims, evidence, stances, signals, [])

    assert verdict.label == VerdictLabel.TRUE


def test_2_zero_evidence_is_unverifiable():
    claims = [_claim("c1")]

    verdict = fuse(claims, [], [], [], [])

    assert verdict.label == VerdictLabel.UNVERIFIABLE
    assert verdict.confidence <= 0.25
    assert any("c1" in c for c in verdict.caveats)


def test_3_mixed_support_and_refute_across_claims_is_mixed():
    claims = [_claim("c1"), _claim("c2")]
    evidence = [_evidence("e1", "c1"), _evidence("e2", "c2")]
    stances = [_stance("c1", "e1", "SUPPORTS"), _stance("c2", "e2", "REFUTES")]

    verdict = fuse(claims, evidence, stances, [], [])

    assert verdict.label == VerdictLabel.MIXED


def test_4_satire_genre_overrides_strong_supporting_evidence():
    claims = [_claim("c1", genre="satire")]
    evidence = [_evidence("e1", "c1")]
    stances = [_stance("c1", "e1", "SUPPORTS")]

    verdict = fuse(claims, evidence, stances, [], [])

    assert verdict.label == VerdictLabel.SATIRE_OPINION


def test_5_ai_text_cannot_dominate_even_with_a_rogue_weight():
    claims = [_claim("c1"), _claim("c2")]
    evidence = [_evidence("e1", "c1"), _evidence("e2", "c2")]
    stances = [_stance("c1", "e1", "SUPPORTS"), _stance("c2", "e2", "REFUTES")]

    verdict_rogue = fuse(
        claims, evidence, stances,
        [Signal(name="ai_text", score=0.95, weight=0.99, note="rogue")],
        [],
    )
    verdict_capped = fuse(
        claims, evidence, stances,
        [Signal(name="ai_text", score=0.95, weight=0.15, note="honest cap")],
        [],
    )

    # A signal claiming 99% weight must have no more influence than the
    # configured 15% cap - fusion ignores the incoming weight entirely.
    assert verdict_rogue.confidence == verdict_capped.confidence
    assert verdict_rogue.label == verdict_capped.label
    assert verdict_rogue.label != VerdictLabel.TRUE


def test_6_all_refuting_high_credibility_is_false():
    claims = [_claim("c1")]
    evidence = [_evidence("e1", "c1")]
    stances = [_stance("c1", "e1", "REFUTES")]
    signals = [Signal(name="source_credibility", score=0.9, weight=0.10, note="tier-1 sources")]

    verdict = fuse(claims, evidence, stances, signals, [])

    assert verdict.label == VerdictLabel.FALSE


def test_7_one_unverified_claim_prevents_full_true_despite_a_strong_sibling():
    claims = [_claim("c1"), _claim("c2")]
    evidence = [_evidence("e1", "c1"), _evidence("e2", "c1"), _evidence("e3", "c1")]
    stances = [
        _stance("c1", "e1", "SUPPORTS"),
        _stance("c1", "e2", "SUPPORTS"),
        _stance("c1", "e3", "SUPPORTS"),
    ]
    # c2 has zero evidence/stance at all.

    verdict = fuse(claims, evidence, stances, [], [])

    assert verdict.label != VerdictLabel.TRUE
    per_claim_by_id = {pc.claim_id: pc for pc in verdict.per_claim}
    assert per_claim_by_id["c1"].label == VerdictLabel.TRUE
    assert per_claim_by_id["c2"].label == VerdictLabel.UNVERIFIABLE
    assert any("c2" in c for c in verdict.caveats)


def test_9_all_neutral_stances_are_unverifiable_not_mixed():
    # Search found evidence, but none of it supports or refutes anything.
    # That is "could not verify" - calling it MIXED would falsely imply
    # contradictory sources.
    claims = [_claim("c1")]
    evidence = [_evidence("e1", "c1"), _evidence("e2", "c1")]
    stances = [_stance("c1", "e1", "NEUTRAL"), _stance("c1", "e2", "NEUTRAL")]
    signals = [Signal(name="source_credibility", score=0.8, weight=0.10, note="decent sources")]

    verdict = fuse(claims, evidence, stances, signals, [])

    assert verdict.label == VerdictLabel.UNVERIFIABLE
    assert verdict.confidence <= 0.25
    assert any("unable to verify" in c for c in verdict.caveats)


def test_10_credible_refutation_reaches_false_despite_human_text_style():
    # The realistic text-path false-news case: credible sources refute the
    # claim, the article reads human-written (high ai_text trust score).
    # Credibility must AMPLIFY the refutation, not drag the score back up,
    # and the stylistic ai_text signal must not rescue a refuted claim.
    claims = [_claim("c1")]
    evidence = [_evidence("e1", "c1"), _evidence("e2", "c1")]
    stances = [_stance("c1", "e1", "REFUTES"), _stance("c1", "e2", "REFUTES")]
    signals = [
        Signal(name="source_credibility", score=0.8, weight=0.10, note="tier-1/2 sources"),
        Signal(name="ai_text", score=0.85, weight=0.15, note="looks human-written"),
    ]

    verdict = fuse(claims, evidence, stances, signals, [])

    assert verdict.label == VerdictLabel.FALSE
    assert verdict.confidence >= 0.5


def test_11_single_unknown_source_cannot_verify_a_claim():
    # The misinformation trap: one random tier-3 webpage repeating an
    # absurd claim ("aliens landed on Mars") gets judged SUPPORTS by
    # stance. That must NOT produce TRUE - a lone uncorroborated unknown
    # source is not verification.
    claims = [_claim("c1")]
    evidence = [_evidence("e1", "c1", tier=3)]
    stances = [_stance("c1", "e1", "SUPPORTS")]

    verdict = fuse(claims, evidence, stances, [], [])

    assert verdict.label == VerdictLabel.UNVERIFIABLE
    per_claim = verdict.per_claim[0]
    assert per_claim.label == VerdictLabel.UNVERIFIABLE
    assert per_claim.supporting == 1  # the raw count stays visible/honest
    assert any("low-credibility" in c for c in verdict.caveats)


def test_11b_two_unknown_sources_are_still_not_enough():
    # Hoaxes get reposted by many random sites - two unknown tier-3 pages
    # repeating a claim (2 x 0.3 = 0.6 < 0.7) still must not verify it.
    claims = [_claim("c1")]
    evidence = [_evidence("e1", "c1", tier=3), _evidence("e2", "c1", tier=3)]
    stances = [_stance("c1", "e1", "SUPPORTS"), _stance("c1", "e2", "SUPPORTS")]

    verdict = fuse(claims, evidence, stances, [], [])

    assert verdict.per_claim[0].label == VerdictLabel.UNVERIFIABLE
    assert verdict.label == VerdictLabel.UNVERIFIABLE


def test_12_single_reputable_source_still_verifies():
    # Corroboration must not overcorrect: one tier-1 source (Reuters, a
    # fact-checker) supporting a claim is legitimate verification.
    claims = [_claim("c1")]
    evidence = [_evidence("e1", "c1", tier=1)]
    stances = [_stance("c1", "e1", "SUPPORTS")]

    verdict = fuse(claims, evidence, stances, [], [])

    assert verdict.per_claim[0].label == VerdictLabel.TRUE
    assert verdict.label in (VerdictLabel.TRUE, VerdictLabel.MOSTLY_TRUE)


def test_13_mostly_unverified_article_cannot_be_mostly_true():
    # The classic fake-news pattern: one unverifiable central claim
    # ("aliens on Mars") wrapped in true filler claims. When half or more
    # of the claims are unverified, the verdict caps at MIXED even if the
    # verified minority scores well.
    claims = [_claim("c1"), _claim("c2"), _claim("c3")]
    evidence = [_evidence("e1", "c3"), _evidence("e2", "c3"), _evidence("e3", "c3")]
    stances = [
        _stance("c3", "e1", "SUPPORTS"),
        _stance("c3", "e2", "SUPPORTS"),
        _stance("c3", "e3", "SUPPORTS"),
    ]
    # c1 and c2 have no usable stance at all.

    verdict = fuse(claims, evidence, stances, [], [])

    assert verdict.label not in (VerdictLabel.TRUE, VerdictLabel.MOSTLY_TRUE)
    assert any("could not be verified" in c for c in verdict.caveats)


def test_8_strong_evidence_outweighs_weak_forensics_but_flags_it():
    claims = [_claim("c1")]
    evidence = [_evidence("e1", "c1"), _evidence("e2", "c1"), _evidence("e3", "c1")]
    stances = [
        _stance("c1", "e1", "SUPPORTS"),
        _stance("c1", "e2", "SUPPORTS"),
        _stance("c1", "e3", "SUPPORTS"),
    ]
    signals = [
        Signal(name="halftone", score=0.1, weight=0.10, note="no print dot pattern found"),
        Signal(name="masthead_match", score=0.1, weight=0.05, note="masthead does not match"),
    ]

    verdict = fuse(claims, evidence, stances, signals, [])

    # The claims are well-corroborated (45% weight) so the verdict is not
    # dragged down to MIXED/FALSE by forensics/masthead (only 15% combined)
    # - but the artifact-authenticity concern must still be visible.
    assert verdict.label in (VerdictLabel.TRUE, VerdictLabel.MOSTLY_TRUE)
    assert any("halftone" in c for c in verdict.caveats)
    assert any("masthead_match" in c for c in verdict.caveats)
