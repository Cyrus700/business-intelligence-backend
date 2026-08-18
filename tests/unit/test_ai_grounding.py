"""The reply-verification layer: a figure with no evidence must not pass.

These guard both directions. Flagging an invented number is the point, but
flagging a correct one is worse than useless — it throws away a good answer
and drops the user to the template engine — so most of these cases assert
that legitimate reformatting, rounding, arithmetic and date-quoting stay
clean.
"""

from app.services.ai.grounding import (
    extract_claims,
    repair_instruction,
    unsupported_figures,
)

EVIDENCE = [
    """KPIs 2026-06-01 → 2026-06-30:
- revenue: 4,076,608 (+17.6% vs previous period)
- orders: 1,204
- avg_order_value: 3,386
Top product revenue 2026-06-01 → 2026-06-30:
- Wai Wai: 6,080 (12.4% share, 2 orders)
"""
]


def test_reformatted_currency_is_grounded():
    """The model rewrites 4076608 as रू 40,76,608; that is the same number."""
    reply = "Revenue for 1 Jun – 30 Jun 2026 was रू 40,76,608, up 17.6% on the previous period."
    assert unsupported_figures(reply, EVIDENCE) == []


def test_rounding_is_grounded():
    reply = "Revenue was रू 40,76,608 (+17.6%). Wai Wai brought रू 6,080 across 2 orders."
    assert unsupported_figures(reply, EVIDENCE) == []


def test_invented_currency_is_flagged():
    reply = "Revenue was रू 52,10,000 in June 2026."
    assert unsupported_figures(reply, EVIDENCE) == ["रू 52,10,000"]


def test_invented_percentage_is_flagged():
    reply = "Margin improved 63.2% year on year."
    assert unsupported_figures(reply, EVIDENCE) == ["63.2%"]


def test_arithmetic_over_evidence_is_grounded():
    """Shares and totals are what the assistant is for; they are not invention."""
    reply = "Wai Wai's रू 6,080 is about 0.1% of the रू 40,76,608 total."
    assert unsupported_figures(reply, EVIDENCE) == []


def test_dates_are_never_claims():
    """Quoting the period is required behaviour — it must not read as a figure."""
    reply = "Between 2026-06-01 and 2026-06-30, and again on 10 June 2026, across 1,204 orders."
    assert unsupported_figures(reply, EVIDENCE) == []


def test_list_markers_are_not_claims():
    reply = "1. revenue रू 40,76,608\n2. orders 1,204"
    assert unsupported_figures(reply, EVIDENCE) == []


def test_small_bare_counts_are_not_challenged():
    reply = "There were 3 open alerts."
    assert unsupported_figures(reply, EVIDENCE) == []


def test_no_evidence_means_no_verdict():
    """With nothing to check against, judging the reply would be guesswork."""
    assert unsupported_figures("Revenue was रू 99,99,999.", []) == []


def test_untagged_large_number_is_still_checked():
    assert unsupported_figures("We shipped 8,412,000 units.", EVIDENCE) == ["8,412,000"]


def test_extract_claims_skips_years_and_keeps_money():
    claims = extract_claims("In 2026 revenue hit रू 40,76,608 (+17.6%).")
    values = [v for _, v in claims]
    assert 2026 not in values
    assert 4076608 in values
    assert 17.6 in values


def test_repair_instruction_names_the_offending_figures():
    text = repair_instruction(["रू 52,10,000", "63.2%"])
    assert "रू 52,10,000" in text
    assert "63.2%" in text
    assert "Do not call any tools" in text
