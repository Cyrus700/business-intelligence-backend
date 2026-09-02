"""Numeric grounding: does every figure in a reply trace back to real data?

Prompting alone does not stop a model inventing a number. It stops it *most*
of the time, which is worse — the failures that get through are fluent,
well-formatted and indistinguishable from the correct answers. This module
checks the finished reply against the evidence that produced it (tool results
plus the live snapshot), so an unsupported figure triggers one repair round
instead of reaching the user as fact.

The check is deliberately permissive: a false flag costs a good answer, so a
figure counts as supported when it matches an evidence number within rounding
slack, or is plainly derived from two of them (a sum, a difference, a share or
a percentage change). Only figures that survive all of that are reported.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: Relative slack for a figure to count as "the same number" as its evidence.
#: The model reformats and rounds constantly (17.63% → 17.6%), and that is
#: correct behaviour, not a hallucination.
TOLERANCE = 0.005
#: Absolute slack, so small integers survive rounding too.
ABS_TOLERANCE = 0.51
#: A bare number below this is a count, a rank or a list index — not a claim
#: worth challenging. Currency- and percent-tagged figures are checked at any
#: magnitude.
BARE_FIGURE_FLOOR = 1000.0
#: Pairwise derivation is O(n²); past this many evidence numbers we only do
#: the direct match, which is still the case that matters.
MAX_PAIRWISE = 120
MAX_EVIDENCE_NUMBERS = 400

_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

# Substrings that contain digits but assert nothing about the business.
# Masked before extraction so "10 Jun 2026" never reads as the figures 10 and
# 2026 — quoting the period back is required behaviour, not a claim.
_MASK_PATTERNS = (
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?\s*\d{{0,4}}\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{0,4}}\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(r"\bQ[1-4]\s*\d{4}\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\b"),  # a bare year
    re.compile(r"\b\d{1,2}:\d{2}\b"),  # clock time
    re.compile(r"^\s*\d{1,2}[.)]\s", re.MULTILINE),  # ordered-list markers
)

# A number, optionally signed, with either grouping convention, plus whatever
# unit sits against it. Both currency spellings the dashboard uses are
# recognised so "रू 40,76,608" and "Rs 40,76,608" are treated alike.
_FIGURE_RE = re.compile(
    r"(?P<cur>रू|Rs\.?|NPR|₹)?\s*"
    r"(?P<num>[-+]?\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent)?",
    re.IGNORECASE,
)


def _mask(text: str) -> str:
    for pattern in _MASK_PATTERNS:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace("+", ""))
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    """Every numeric value in a block of text, for use as evidence."""
    out: list[float] = []
    for m in _FIGURE_RE.finditer(text or ""):
        value = _to_float(m.group("num"))
        if value is not None:
            out.append(value)
        if len(out) >= MAX_EVIDENCE_NUMBERS:
            break
    return out


def extract_claims(reply: str) -> list[tuple[str, float]]:
    """Figures a reply asserts as fact, as (rendered_text, value) pairs.

    Dates, times, years and list markers are masked out first: quoting the
    period a number covers is required of the assistant, and penalising it for
    doing so would flag every correct answer.
    """
    masked = _mask(reply or "")
    claims: list[tuple[str, float]] = []
    for m in _FIGURE_RE.finditer(masked):
        value = _to_float(m.group("num"))
        if value is None:
            continue
        tagged = bool(m.group("cur") or m.group("unit"))
        if not tagged and abs(value) < BARE_FIGURE_FLOOR:
            continue
        claims.append((m.group(0).strip(), value))
    return claims


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOLERANCE, abs(b) * TOLERANCE)


def _derivable(value: float, evidence: list[float]) -> bool:
    """True when the figure is ordinary arithmetic over two evidence numbers.

    Totals, gaps, shares and period-over-period changes are exactly what a BI
    assistant is supposed to compute, so they must not read as invention.
    """
    if len(evidence) > MAX_PAIRWISE:
        return False
    for a in evidence:
        for b in evidence:
            if _close(value, a + b) or _close(value, a - b):
                return True
            if b:
                if _close(value, a / b * 100.0):  # share
                    return True
                if _close(value, (a - b) / abs(b) * 100.0):  # change vs previous
                    return True
    return False


def unsupported_figures(reply: str, evidence_texts: list[str]) -> list[str]:
    """Figures in ``reply`` that no evidence text can account for.

    An empty list means the reply is grounded as far as this check can tell —
    it proves the numbers came from somewhere real, not that the analysis
    around them is right.
    """
    claims = extract_claims(reply)
    if not claims:
        return []

    evidence: list[float] = []
    for text in evidence_texts:
        evidence.extend(extract_numbers(text))
    if not evidence:
        # Nothing to check against: refusing to judge beats flagging a whole
        # reply because the evidence never made it through.
        return []

    bad: list[str] = []
    for rendered, value in claims:
        if any(_close(value, e) for e in evidence):
            continue
        if _derivable(value, evidence):
            continue
        if rendered not in bad:
            bad.append(rendered)
    return bad


def repair_instruction(bad: list[str]) -> str:
    """The correction turn sent back to the model for a second attempt."""
    listed = ", ".join(bad[:8])
    return (
        "STOP. These figures in your last reply do not appear in any tool result "
        f"or in the live snapshot: {listed}. "
        "Rewrite the answer now using ONLY figures that appear verbatim in the tool "
        "results or the snapshot above. If a number you wanted is not there, say "
        "plainly that it is not available and name the dashboard panel that has it. "
        "Do not call any tools. Do not apologise or mention this correction — return "
        "the corrected answer only."
    )
