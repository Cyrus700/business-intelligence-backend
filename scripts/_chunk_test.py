import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ai.provider import SYSTEM_PROMPT_DASHBOARD, AIMessage, get_ai_stream


async def main() -> None:
    msgs = [AIMessage(role="user", content="What's our revenue trend this month?")]
    n = 0
    with open("chunks.txt", "w", encoding="utf-8") as fh:
        async for chunk in get_ai_stream(msgs, system_prompt=SYSTEM_PROMPT_DASHBOARD):
            n += 1
            fh.write(f"[{n}] {chunk!r}\n")
    print(f"total chunks: {n}")


asyncio.run(main())
