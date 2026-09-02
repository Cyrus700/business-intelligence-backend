"""Lightweight retrieval over past insights / anomalies / recommendations (RAG).

The past-decisions layer that lets the assistant answer "have we seen this
before?" from the same tables the insight engine and anomaly scanner write to —
so anything the AI claims is traceable to stored rows (R5).

Index: a deterministic hashing embedding (sign-hashed BLAKE2s → 512-dim,
L2-normalised) with cosine relevance, built in-memory on first use and refreshed
on a TTL. Zero ML dependencies, deterministic across restarts, and the same
mathematics drops straight into a pgvector column if the warehouse ever
upgrades to one.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Anomaly, Insight

logger = logging.getLogger(__name__)

EMBED_DIM = 512
INDEX_TTL_SECONDS = 300  # refresh the in-memory index at most every 5 min
MAX_DOCS = 500


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def embed(text: str) -> np.ndarray:
    """Deterministic hashing-embedding → unit vector of length EMBED_DIM."""
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for tok in _tokens(text):
        h = int.from_bytes(hashlib.blake2s(tok.encode(), digest_size=4).digest(), "big")
        idx = h % EMBED_DIM
        sign = 1.0 if (h >> 31) & 1 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


@dataclass
class RetrieverResult:
    kind: str  # insight | recommendation | anomaly
    title: str
    text: str
    score: float


def _fmt_evidence(evidence: dict | None) -> str:
    if not evidence:
        return ""
    return " ".join(f"{k}: {v}" for k, v in evidence.items() if not isinstance(v, dict))


class InsightRetriever:
    """TTL-cached in-memory index; rebuilt quietly when stale or empty."""

    def __init__(self) -> None:
        self._docs: list[RetrieverResult]
        self._matrix: np.ndarray | None = None
        self._built_at: float = 0.0
        self._empty_check: float = 0.0

    def _stale(self) -> bool:
        return time.monotonic() - self._built_at > INDEX_TTL_SECONDS

    async def _build(self, db: AsyncSession) -> None:
        docs = self._load_docs(db)
        self._docs = docs
        if docs:
            self._matrix = np.stack([embed(d.text) for d in docs])
        else:
            self._matrix = None
        self._built_at = time.monotonic()

    def _load_docs(self, db: AsyncSession) -> list[RetrieverResult]:
        # kept synchronous-returning for clarity; executed from the async path
        raise NotImplementedError

    async def _refresh(self, db: AsyncSession, org_id=None) -> list[RetrieverResult]:
        """Full rebuild from the warehouse tables (org-scoped)."""
        # Org scoping: Insights/Anomalies are per-business; cache is global but we filter here.
        # For full per-org cache you'd key by org_id; v1 filters rows after load. Super-admin sees all.
        all_rows: list[dict] = []

        iq = select(Insight).order_by(Insight.generated_at.desc()).limit(MAX_DOCS)
        if org_id is not None:
            iq = iq.where(Insight.org_id == org_id)
        insights = ((await db.execute(iq)).scalars().all())
        for ins in insights:
            all_rows.append(
                {
                    "kind": "recommendation" if ins.insight_type == "recommendation" else "insight",
                    "title": ins.title,
                    "text": (
                        f"{ins.title}. {ins.body} "
                        f"{_fmt_evidence(ins.evidence)}"
                    ),
                    "created_at": ins.generated_at.isoformat() if ins.generated_at else None,
                }
            )

        aq = select(Anomaly).order_by(Anomaly.detected_at.desc()).limit(MAX_DOCS)
        if org_id is not None:
            aq = aq.where(Anomaly.org_id == org_id)
        anomalies = ((await db.execute(aq)).scalars().all())
        for a in anomalies:
            ctx = a.context or {}
            date_str = ctx.get("date", "?")
            all_rows.append(
                {
                    "kind": "anomaly",
                    "title": f"Anomaly: {a.metric} {date_str}",
                    "text": (
                        f"{a.metric} anomaly detected on {date_str}: observed "
                        f"{float(a.observed_value):,.0f} vs expected "
                        f"{float(a.expected_value or 0):,.0f} "
                        f"({ctx.get('pct_deviation', '?')}% {ctx.get('direction', 'deviation')}, "
                        f"severity {a.severity}). Root cause: "
                        f"{ctx.get('root_cause') or 'not analysed'}"
                    ),
                    "created_at": a.detected_at.isoformat() if a.detected_at else None,
                }
            )

        docs = [
            RetrieverResult(
                kind=row["kind"],
                title=row["title"],
                text=row["text"],
                score=0.0,
            )
            for row in all_rows
        ]
        return docs

    async def search(self, db: AsyncSession, query: str, top_k: int = 5, org_id=None) -> list[RetrieverResult]:
        # Per-org: we rebuild filtered by org_id each call when org changes would otherwise leak.
        # Keep global TTL but force refresh when org differs from last build's org.
        # Simplest v1: always refresh with org filter if cache stale or org provided (org-scoped correctness over speed).
        needs_refresh = not getattr(self, "_docs", None) or self._stale()
        # If caller is org-scoped, ensure docs belong to that org — rebuild filtered
        if org_id is not None:
            needs_refresh = True
        if needs_refresh:
            self._docs = await self._refresh(db, org_id=org_id)
            self._built_at = time.monotonic()
            if self._docs:
                self._matrix = np.stack([embed(d.text) for d in self._docs])
            else:
                self._matrix = None
        if self._matrix is None or not self._docs:
            return []
        q = embed(query)
        scores = self._matrix @ q  # rows L2-normalised → cosine similarity
        order = np.argsort(-scores)[:top_k]
        results = []
        for idx in order:
            doc = self._docs[int(idx)]
            results.append(
                RetrieverResult(
                    kind=doc.kind,
                    title=doc.title,
                    text=doc.text,
                    score=round(float(scores[int(idx)]), 3),
                )
            )
        return results

    async def refresh(self, db: AsyncSession, org_id=None) -> int:
        self._docs = await self._refresh(db, org_id=org_id)
        self._matrix = np.stack([embed(d.text) for d in self._docs]) if self._docs else None
        self._built_at = time.monotonic()
        return len(self._docs)


_retriever: InsightRetriever | None = None


def get_retriever() -> InsightRetriever:
    global _retriever
    if _retriever is None:
        _retriever = InsightRetriever()
    return _retriever


def reset_retriever() -> None:
    global _retriever
    _retriever = None