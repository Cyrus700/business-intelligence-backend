"""Manual smoke test: does the assistant actually reach live data?

Seeds a tiny warehouse in the test database, then asks date-specific questions
through the real provider and prints which tools ran. Not part of pytest — it
needs a live GROQ_API_KEY and makes real API calls.

    uv run python scripts/_ai_smoke.py
"""

import asyncio
import os
import uuid

os.environ.setdefault("ENV", "test")
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:54329/bi_test"
)

from app.core.database import dispose_engine, get_session_factory  # noqa: E402
from app.models import Product, Profile, SalesTransaction  # noqa: E402
from app.services.ai.provider import AIMessage  # noqa: E402
from app.services.ai.service import answer_question  # noqa: E402

DAYS = {"2026-06-10": 7040, "2026-06-11": 640, "2026-06-12": 1280}


async def seed(session) -> Profile:
    from sqlalchemy import delete

    await session.execute(delete(SalesTransaction))
    await session.execute(delete(Product))
    product = Product(sku="BEV-001", name="Everest Tea", category="Beverages", unit_cost=200)
    session.add(product)
    await session.flush()
    for day, amount in DAYS.items():
        session.add(
            SalesTransaction(
                txn_date=__import__("datetime").date.fromisoformat(day),
                product_id=product.id,
                quantity=1,
                unit_price=amount,
                discount=0,
                total_amount=amount,
                channel="store",
                region="Bagmati",
                row_hash=uuid.uuid4().hex,
            )
        )
    user = Profile(id=uuid.uuid4(), email=f"smoke-{uuid.uuid4().hex[:6]}@x.com", role="admin")
    session.add(user)
    await session.commit()
    return user


QUESTIONS = [
    "What was revenue on 2026-06-10?",
    "Compare 10 June and 12 June 2026 revenue.",
    "What was revenue on 2019-01-01?",
    "How did we do in the last 7 days?",
]


async def main() -> None:
    async with get_session_factory()() as session:
        user = await seed(session)
        for q in QUESTIONS:
            result = await answer_question(
                session, "admin", q, history=[AIMessage(role="user", content=q)], user=user
            )
            print("=" * 78)
            print("Q:", q)
            print(f"[source={result.source} tool_rounds={result.tool_calls}]")
            print(result.reply.strip()[:900])
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
