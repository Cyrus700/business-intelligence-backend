import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class AIMessage:
    role: str
    content: str


class AIProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[AIMessage], system_prompt: str | None = None) -> str:
        ...


class GroqProvider(AIProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model or "llama-3.3-70b-versatile"

    async def chat(self, messages: list[AIMessage], system_prompt: str | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured")
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self.api_key)
            msgs: list[dict] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            for m in messages:
                msgs.append({"role": m.role, "content": m.content})
            resp = await client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=0.7,
                max_tokens=4096,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("Groq API error: %s", e)
            raise


class GeminiProvider(AIProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model or "gemini-2.0-flash"

    async def chat(self, messages: list[AIMessage], system_prompt: str | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not configured")
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            msgs: list[types.Content] = []
            for m in messages:
                msgs.append(types.Content(role=m.role, parts=[types.Part.from_text(text=m.content)]))
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=4096,
                temperature=0.7,
            )
            resp = client.models.generate_content(
                model=self.model,
                contents=msgs,
                config=config,
            )
            return resp.text or ""
        except Exception as e:
            logger.warning("Gemini API error: %s", e)
            raise


SYSTEM_PROMPT_DASHBOARD = """You are Insightful AI, a smart business intelligence assistant for a retail BI dashboard. You help users understand their sales, expenses, inventory, and forecasts data. You have access to real-time analytics data through the dashboard.

Rules:
1. Be concise and data-driven. Use numbers and percentages.
2. When asked about specific metrics, reference the data available.
3. If you don't have specific data, suggest what the user can look at in their dashboard.
4. Keep responses under 200 words unless asked for detail.
5. Never make up numbers — direct users to their dashboard metrics.
6. Use Nepali-friendly formatting for currency (e.g., रू 1,23,456)."""


async def get_ai_response(
    messages: list[AIMessage],
    system_prompt: str | None = None,
) -> str:
    settings = get_settings()
    system_prompt = system_prompt or SYSTEM_PROMPT_DASHBOARD
    providers: list[AIProvider] = []

    if settings.groq_api_key:
        providers.append(GroqProvider())
    if settings.gemini_api_key:
        providers.append(GeminiProvider())

    if not providers:
        return _fallback_response(messages)

    last_error: Exception | None = None
    for provider in providers:
        try:
            return await provider.chat(messages, system_prompt)
        except Exception as e:
            last_error = e
            logger.warning("Provider %s failed, trying next: %s", type(provider).__name__, e)
            continue

    if last_error:
        logger.error("All AI providers failed: %s", last_error)
    return _fallback_response(messages)


def _fallback_response(messages: list[AIMessage]) -> str:
    last = messages[-1].content.lower() if messages else ""
    if "revenue" in last or "sale" in last:
        return "I can see your revenue data on the dashboard KPIs at the top of the page. Check the **Revenue** card for current period totals and trend."
    if "expense" in last:
        return "Expense data is shown in the **Revenue vs Expenses** chart on your dashboard. It breaks down spending across categories."
    if "forecast" in last or "predict" in last:
        return "The **Revenue Forecast** panel shows 30-day projections using our ML models (Prophet/ARIMA). View it on the dashboard for confidence intervals and accuracy metrics."
    if "inventory" in last or "stock" in last:
        return "The **Low Stock** panel on your dashboard highlights products below reorder level. Visit the Inventory section for detailed levels."
    if "anomaly" in last:
        return "The **Anomaly Alerts** section on your dashboard lists detected anomalies with severity, observed vs expected values, and deviation percentages."
    return (
        "I'm your BI assistant. I can help you understand your dashboard metrics — try asking about "
        "**revenue trends**, **expense breakdown**, **forecast predictions**, **inventory levels**, "
        "or **anomaly alerts**. Your data is displayed live on the dashboard panels."
    )
