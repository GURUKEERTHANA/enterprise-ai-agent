"""
Latency Profiling Utilities.

Provides decorators and context managers for measuring execution time
across all pipeline stages: embedding, retrieval, reranking, LLM call.

Notes:
    In our ITSM agent profiling, LLM calls account for ~76% of total latency,
    embedding API ~21%, and local compute (BM25, reranking, RRF) ~3%.

    This breakdown tells you exactly where to optimize:
    - If LLM is 76%: use a faster model, add semantic caching, reduce prompt size
    - If embedding is 21%: batch embed, cache embeddings, use a smaller model
    - If BM25/reranker is significant: they're CPU-bound, consider async or precompute

    Without profiling, engineers guess wrong. This data is what justifies
    architectural decisions

    ServiceNow equivalent: Performance analytics and transaction log timing in the platform.
"""

import functools
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class LatencyRecord:
    """Single timing observation for a labeled stage."""
    stage: str
    duration_ms: float
    metadata: dict = field(default_factory=dict)


class LatencyProfiler:
    """
    Collects and reports latency measurements across pipeline stages.

    Usage:
        profiler = LatencyProfiler()

        with profiler.measure("bm25_retrieval"):
            results = bm25.retrieve(query)

        with profiler.measure("embedding"):
            embedding = embedder(query)

        profiler.report()  # prints breakdown table

    Notes:
        We instrument every stage with a context manager rather than
        wrapping functions. This keeps the measurement logic orthogonal
        to the business logic — we can add/remove profiling without
        changing the pipeline code.
    """

    # Stage-name → coarse category. Order matters: first matching prefix wins.
    # The category buckets are what produces the headline "76 / 21 / 3" split
    # (LLM / Embedding / Local compute) — see category_summary().
    _CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
        ("llm",           ("llm_call", "router_llm", "synthesizer_llm")),
        ("embedding",     ("embedding_query", "embed")),
        ("local_compute", ("bm25", "chroma", "rrf", "rerank", "injection")),
    ]

    def __init__(self):
        self._records: list[LatencyRecord] = []
        self._stage_totals: dict[str, list[float]] = defaultdict(list)

    @classmethod
    def _categorize(cls, stage: str) -> str:
        s = stage.lower()
        for category, prefixes in cls._CATEGORY_RULES:
            if any(p in s for p in prefixes):
                return category
        return "other"

    @contextmanager
    def measure(self, stage: str, **metadata):
        """
        Context manager that measures the duration of the enclosed block.

        Args:
            stage: Label for this measurement (e.g., 'bm25_retrieval', 'llm_call').
            **metadata: Optional key-value pairs attached to this measurement.

        Usage:
            with profiler.measure("llm_call", model="gpt-4o", tokens=1200):
                response = openai_client.chat.completions.create(...)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            record = LatencyRecord(
                stage=stage,
                duration_ms=duration_ms,
                metadata=metadata
            )
            self._records.append(record)
            self._stage_totals[stage].append(duration_ms)

    def total_ms(self) -> float:
        """Total elapsed time across all measured stages."""
        return sum(r.duration_ms for r in self._records)

    def stage_summary(self) -> dict[str, dict]:
        """
        Per-stage summary: total, mean, count, pct_of_total.

        Returns dict suitable for logging or display.
        """
        total = self.total_ms()
        summary = {}
        for stage, durations in self._stage_totals.items():
            stage_total = sum(durations)
            summary[stage] = {
                "total_ms": round(stage_total, 2),
                "mean_ms": round(stage_total / len(durations), 2),
                "count": len(durations),
                "pct_of_total": round((stage_total / total * 100) if total > 0 else 0, 1)
            }
        return summary

    def category_summary(self) -> dict[str, dict]:
        """
        Coarse breakdown by category (llm / embedding / local_compute / other).

        This is the headline view that gives the LLM/Embedding/Local split —
        the one architectural-decision number to optimize against.
        """
        total = self.total_ms()
        buckets: dict[str, float] = defaultdict(float)
        for stage, durations in self._stage_totals.items():
            buckets[self._categorize(stage)] += sum(durations)
        return {
            category: {
                "total_ms": round(ms, 2),
                "pct_of_total": round((ms / total * 100) if total > 0 else 0, 1),
            }
            for category, ms in buckets.items()
        }

    def report(self, title: str = "Latency Breakdown") -> str:
        """
        Generate a formatted latency report string with per-stage timings
        and a coarse category breakdown (LLM / Embedding / Local compute).

        Example output:
            ────────────────────────────────────────────────────────────
            Latency Breakdown — Total: 1,250.0ms
            ────────────────────────────────────────────────────────────
            Stage                          Total      Mean   Count    %
            llm_call                       950.0ms   950.0ms     1   76%
            embedding_query                263.0ms   263.0ms     1   21%
            chroma_retrieval                25.0ms    25.0ms     1    2%
            reranking                        8.0ms     8.0ms     1    1%
            bm25_retrieval                   2.0ms     2.0ms     1    0%
            rrf_fusion                       1.7ms     1.7ms     1    0%
            injection_check                  0.3ms     0.3ms     1    0%
            ────────────────────────────────────────────────────────────
            By category:
            llm                            950.0ms   76%
            embedding                      263.0ms   21%
            local_compute                   37.0ms    3%
            ────────────────────────────────────────────────────────────
        """
        summary = self.stage_summary()
        total = self.total_ms()

        lines = [
            "─" * 60,
            f"{title} — Total: {total:,.1f}ms",
            "─" * 60,
            f"{'Stage':<28} {'Total':>8}  {'Mean':>8}  {'Count':>5}  {'%':>4}",
        ]

        # Sort by total_ms descending (biggest bottleneck first)
        for stage, stats in sorted(summary.items(), key=lambda x: x[1]["total_ms"], reverse=True):
            lines.append(
                f"{stage:<28} {stats['total_ms']:>7.1f}ms "
                f"{stats['mean_ms']:>7.1f}ms  {stats['count']:>5}  "
                f"{stats['pct_of_total']:>3.0f}%"
            )

        cat_summary = self.category_summary()
        if cat_summary:
            lines.append("─" * 60)
            lines.append("By category:")
            ordered = sorted(
                cat_summary.items(), key=lambda x: x[1]["total_ms"], reverse=True
            )
            for category, stats in ordered:
                lines.append(
                    f"{category:<28} {stats['total_ms']:>7.1f}ms  "
                    f"{stats['pct_of_total']:>3.0f}%"
                )

        lines.append("─" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded measurements. Call between requests."""
        self._records.clear()
        self._stage_totals.clear()

    def to_dict(self) -> dict:
        """
        Serialize latency data for logging (e.g., to Langfuse or CloudWatch).

        Returns flat dict suitable for structured logging:
            {"total_ms": 1243, "bm25_ms": 2.1, "llm_ms": 945.8, ...}
        """
        result = {"total_ms": round(self.total_ms(), 2)}
        for stage, stats in self.stage_summary().items():
            result[f"{stage}_ms"] = stats["mean_ms"]
            result[f"{stage}_pct"] = stats["pct_of_total"]
        for category, stats in self.category_summary().items():
            result[f"cat_{category}_ms"] = stats["total_ms"]
            result[f"cat_{category}_pct"] = stats["pct_of_total"]
        return result


# ------------------------------------------------------------------
# Decorator for simple function timing
# ------------------------------------------------------------------

def timed(stage: Optional[str] = None, profiler: Optional[LatencyProfiler] = None):
    """
    Decorator to time a function and optionally record in a LatencyProfiler.

    Usage:
        @timed("embed_query")
        def embed(text: str) -> list[float]:
            return openai.embeddings.create(input=text).data[0].embedding

        # Or with a shared profiler:
        profiler = LatencyProfiler()

        @timed("embed_query", profiler=profiler)
        def embed(text: str) -> list[float]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        label = stage or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            if profiler is not None:
                profiler._records.append(LatencyRecord(stage=label, duration_ms=duration_ms))
                profiler._stage_totals[label].append(duration_ms)
            else:
                # If no profiler, just print (useful during development)
                print(f"[LATENCY] {label}: {duration_ms:.1f}ms")

            return result
        return wrapper
    return decorator


# ------------------------------------------------------------------
# Request-scoped profiler for LangGraph state
# ------------------------------------------------------------------

def new_profiler() -> LatencyProfiler:
    """
    Create a fresh profiler for a new request.

    Usage in LangGraph:
        def entry_node(state: AgentState) -> AgentState:
            return {**state, "profiler": new_profiler()}

        def llm_node(state: AgentState) -> AgentState:
            profiler = state["profiler"]
            with profiler.measure("llm_call", model="gpt-4o"):
                response = call_llm(...)
            return {**state, "response": response}

        def exit_node(state: AgentState) -> AgentState:
            print(state["profiler"].report())
            return state
    """
    return LatencyProfiler()
