"""Streaming must not answer questions it has no tools to answer.

Token streaming runs without the tool loop, so anything the 30-day snapshot
cannot cover has to be routed to the tool-grounded path instead — otherwise the
model composes a fluent answer about channels or regions from a prompt that
never contained them.
"""

from app.services.ai.intents import Intent, detect_intent
from app.services.ai.service import _needs_live_tools


def _route(question: str) -> bool:
    return _needs_live_tools(question, detect_intent(question))


def test_dated_questions_need_tools():
    assert _route("What was revenue on 10 June?")
    assert _route("How did we do yesterday?")
    assert _route("Revenue for 2026-06-01 to 2026-06-30?")


def test_snapshot_blind_intents_need_tools():
    """Channels, regions and comparisons are absent from the snapshot."""
    assert _route("Which channel performs best?")
    assert _route("Break revenue down by region")
    assert _route("Compare online versus store")


def test_detail_requests_need_tools():
    assert _route("Give me the top 10 products")
    assert _route("Show revenue day by day")
    assert _route("What's the breakdown by category?")
    assert _route("Have we seen this anomaly before?")


def test_snapshot_answerable_questions_stream():
    """These are covered by the snapshot, so they keep the faster path."""
    assert not _route("How is revenue doing?")
    assert not _route("What are my top products?")
    assert not _route("Anything I should worry about?")


def test_greetings_never_reach_the_router():
    assert detect_intent("hello there") is Intent.GREETING


def test_diagnostic_questions_need_tools():
    """'Why' has a decomposition tool behind it; the snapshot cannot answer it."""
    assert _route("Why is revenue down?")
    assert _route("What caused the drop?")
    assert _route("Explain the change in sales")


def test_forward_looking_questions_need_tools():
    assert _route("Are we on track this month?")
    assert _route("What if we raised order value 10%?")
    assert _route("How exposed are we to one product?")
