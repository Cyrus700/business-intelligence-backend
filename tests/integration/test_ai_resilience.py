"""The chat endpoint must survive a failing analytics query.

Postgres aborts the entire transaction on the first error, so one bad read
(a lagging migration, a dropped column) used to cascade: every later query
failed with InFailedSQLTransactionError, the final commit blew up, and the
user got a 500 with no reply and a lost question.
"""

import inspect

import pytest
from sqlalchemy import text

from app.services.ai import service as ai_service
from tests.conftest import auth


@pytest.fixture
def break_context(monkeypatch):
    """Make the answer path issue a query that aborts the transaction."""

    async def poisoned(db, *args, **kwargs):
        try:
            await db.execute(text("SELECT column_that_does_not_exist FROM sales_transactions"))
        except Exception:
            pass  # the transaction is now aborted, exactly as in the real bug
        return "context unavailable"

    monkeypatch.setattr(ai_service, "build_business_context", poisoned)


async def test_chat_survives_a_poisoned_transaction(client, user_token, break_context):
    _, token = user_token
    resp = await client.post(
        "/api/v1/ai/chat",
        headers=auth(token),
        json={"message": "How is revenue doing?"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["reply"].strip()
    assert body["conversation_id"]


async def test_user_message_survives_a_failed_answer(client, user_token, break_context):
    """The question is committed before generation, so it is never lost."""
    _, token = user_token
    resp = await client.post("/api/v1/ai/chat", headers=auth(token), json={"message": "revenue please"})
    conv_id = resp.json()["conversation_id"]

    messages = await client.get(f"/api/v1/ai/conversations/{conv_id}/messages", headers=auth(token))
    assert messages.status_code == 200
    roles = [m["role"] for m in messages.json()]
    assert "user" in roles


async def test_stream_endpoint_accepts_the_user_argument():
    """Regression: ai.py passes user=..., which stream_answer used to reject.

    The TypeError was swallowed into a generic 'stream failed' SSE event, so
    streaming silently never worked.
    """
    params = inspect.signature(ai_service.stream_answer).parameters
    assert "user" in params


async def test_streaming_chat_returns_events(client, user_token):
    _, token = user_token
    resp = await client.post(
        "/api/v1/ai/chat/stream",
        headers=auth(token),
        json={"message": "hello"},
    )
    assert resp.status_code == 200
    assert "conversation_id" in resp.text
    assert "I hit an error while generating a reply" not in resp.text
