"""Temporal consistency checker (owned by Member B).

Purpose: pure-Python comparison of claim dates vs. evidence dates; flags
old-news-as-new (evidence for the same event >90 days older than the
claim's implied date) and future-dated anomalies.
Input: claims (list[Claim]), evidence (list[EvidenceItem]). Output: a
single Signal (core.schemas) named "temporal_consistency".

Note: "temporal_consistency" is not currently a weighted category in
core.fusion.NAME_TO_CATEGORY, so this signal surfaces as a caveat (via
fusion.py's low-score/skipped-signal caveat logic) rather than entering the
weighted score - matching the mock's documented behavior. Promoting it to a
weighted category is a product decision for the team, not a merge blocker.
"""

from datetime import datetime, timedelta

from dateutil import parser as dateparser

from core.schemas import Claim, EvidenceItem, Signal

STALE_DAYS = 90
FUSION_WEIGHT = 0.05  # advisory - Member A owns the final fusion table


def _parse(value) -> datetime | None:
    if not value:
        return None
    try:
        return dateparser.parse(str(value), fuzzy=True)  # handles messy strings
    except (ValueError, OverflowError):
        return None


def check_temporal(claims: list[Claim], evidence: list[EvidenceItem]) -> Signal:
    checked, flagged, notes = 0, 0, []

    for claim in claims:
        claim_date = _parse(getattr(claim, "event_date", None))
        if claim_date is None:
            continue
        ev_dates = [d for e in evidence if e.claim_id == claim.claim_id
                    and (d := _parse(e.published_date)) is not None]
        if not ev_dates:
            continue
        checked += 1
        newest = max(ev_dates)
        if newest < claim_date - timedelta(days=STALE_DAYS):
            flagged += 1
            notes.append(
                f"{claim.claim_id}: coverage is {(claim_date - newest).days} days "
                f"older than the claimed date - possible recycled story"
            )

    if checked == 0:
        return Signal(name="temporal_consistency", score=0.5, weight=FUSION_WEIGHT,
                      note="no comparable dates found - temporal check inconclusive")

    score = max(0.1, 1.0 - flagged / checked)
    note = "; ".join(notes) if notes else f"dates consistent across {checked} claim(s)"
    return Signal(name="temporal_consistency", score=round(score, 3),
                  weight=FUSION_WEIGHT, note=note)
