"""
RAG Evaluation Pipeline — 5 Retrieval Metrics.

Evaluates retrieval quality against a golden eval set using:
    1. Hit@1    — is the correct chunk the very first result?
    2. Hit@3    — is it in the top 3?
    3. Hit@5    — is it in the top 5?
    4. MRR      — Mean Reciprocal Rank (captures rank quality)
    5. Precision@5 — fraction of top-5 results that are relevant

Interview talking point:
    These metrics answer different questions:
    - Hit@1: precision — critical for single-answer agents
    - Hit@5: recall — what you feed to the reranker; low Hit@5 = reranker can't save you
    - MRR: rank quality — does the right answer appear early or late?
    - Precision@5: multi-relevant query quality — for queries with multiple valid answers

    MRR of 0.955 means the correct chunk appears at rank 1 for most queries and
    occasionally at rank 2. MRR of 0.5 means you're usually at rank 2 — that's
    a signal your reranker is doing the right thing but your BM25/dense leg is weak.

    Key lesson from our eval work: wrong expected chunk IDs silently corrupt all metrics.
    ALWAYS verify ground truth by reading the retrieved text, not just checking IDs.

    ServiceNow equivalent: Now Assist accuracy benchmarking on golden test cases.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalQuery:
    """Single entry in the golden eval set."""
    query_id: str
    query: str
    expected_chunk_ids: list[str]     # one or more correct chunks (ground truth)
    department_id: Optional[str] = None
    category: str = "general"         # factual / procedural / keyword / out_of_scope


@dataclass
class QueryResult:
    """Result for a single eval query."""
    query_id: str
    query: str
    retrieved_chunk_ids: list[str]    # in rank order
    expected_chunk_ids: list[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float            # 1/rank of first correct hit, 0 if not in top-k
    precision_at_5: float
    first_hit_rank: Optional[int]     # rank (1-indexed) of first correct chunk, None if missed


@dataclass
class EvalReport:
    """Aggregated evaluation report across all queries."""
    n_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float                         # Mean Reciprocal Rank
    precision_at_5: float
    query_results: list[QueryResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"\n{'─'*50}\n"
            f"RAG Evaluation Report — {self.n_queries} queries\n"
            f"{'─'*50}\n"
            f"  Hit@1:         {self.hit_at_1:.3f}  ({self.hit_at_1*100:.1f}%)\n"
            f"  Hit@3:         {self.hit_at_3:.3f}  ({self.hit_at_3*100:.1f}%)\n"
            f"  Hit@5:         {self.hit_at_5:.3f}  ({self.hit_at_5*100:.1f}%)\n"
            f"  MRR:           {self.mrr:.3f}\n"
            f"  Precision@5:   {self.precision_at_5:.3f}  ({self.precision_at_5*100:.1f}%)\n"
            f"{'─'*50}"
        )

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "hit_at_1": round(self.hit_at_1, 4),
            "hit_at_3": round(self.hit_at_3, 4),
            "hit_at_5": round(self.hit_at_5, 4),
            "mrr": round(self.mrr, 4),
            "precision_at_5": round(self.precision_at_5, 4),
        }


class RAGEvaluator:
    """
    Evaluates a retriever against a golden eval set.

    Works with any retriever that has a .retrieve(query, top_k, department_id) method
    returning objects with a .chunk_id attribute. Compatible with BM25Retriever,
    HybridRetriever, or any retriever implementing the same interface.

    Interview talking point:
        We built this evaluator before optimizing our retriever — not after.
        Eval-first means every change to chunking, embedding model, BM25 parameters,
        or RRF k constant gets measured immediately. Without this, you're flying blind.
        Our 4-way comparison (BM25 vs dense vs hybrid vs hybrid+reranker) was only
        possible because we had the evaluator ready first.
    """

    def __init__(self, retriever, top_k: int = 5):
        """
        Args:
            retriever: Any retriever with .retrieve(query, top_k, department_id) method.
            top_k: Number of results to retrieve per query (used for all Hit@k and Precision@5).
                   Must be >= 5 to compute all metrics meaningfully.
        """
        if top_k < 5:
            raise ValueError("top_k must be >= 5 to compute Hit@5 and Precision@5")
        self.retriever = retriever
        self.top_k = top_k

    def evaluate(
        self,
        eval_set: list[EvalQuery],
        verbose: bool = True
    ) -> EvalReport:
        """
        Run evaluation across all queries in the eval set.

        Args:
            eval_set: List of EvalQuery with ground truth chunk IDs.
            verbose: If True, print per-query results.

        Returns:
            EvalReport with aggregated metrics and per-query breakdown.
        """
        query_results = []

        for eq in eval_set:
            result = self._evaluate_single(eq, verbose=verbose)
            query_results.append(result)

        return self._aggregate(query_results)

    def _evaluate_single(self, eq: EvalQuery, verbose: bool) -> QueryResult:
        """Evaluate a single query."""
        # Retrieve
        retrieved = self.retriever.retrieve(
            query=eq.query,
            top_k=self.top_k,
            department_id=eq.department_id
        )

        retrieved_ids = [r.chunk_id for r in retrieved]
        expected_set = set(eq.expected_chunk_ids)

        # --- Hit@k ---
        hit_at_1 = any(cid in expected_set for cid in retrieved_ids[:1])
        hit_at_3 = any(cid in expected_set for cid in retrieved_ids[:3])
        hit_at_5 = any(cid in expected_set for cid in retrieved_ids[:5])

        # --- Reciprocal Rank ---
        reciprocal_rank = 0.0
        first_hit_rank = None
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in expected_set:
                reciprocal_rank = 1.0 / rank
                first_hit_rank = rank
                break

        # --- Precision@5 ---
        top5_ids = retrieved_ids[:5]
        relevant_in_top5 = sum(1 for cid in top5_ids if cid in expected_set)
        precision_at_5 = relevant_in_top5 / min(5, len(top5_ids)) if top5_ids else 0.0

        result = QueryResult(
            query_id=eq.query_id,
            query=eq.query,
            retrieved_chunk_ids=retrieved_ids,
            expected_chunk_ids=eq.expected_chunk_ids,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            reciprocal_rank=reciprocal_rank,
            precision_at_5=precision_at_5,
            first_hit_rank=first_hit_rank
        )

        if verbose:
            hit_symbol = "✓" if hit_at_5 else "✗"
            print(f"  [{hit_symbol}] {eq.query_id}: rank={first_hit_rank or 'miss'}, "
                  f"RR={reciprocal_rank:.3f}, P@5={precision_at_5:.2f}")

        return result

    @staticmethod
    def _aggregate(query_results: list[QueryResult]) -> EvalReport:
        """Aggregate per-query results into summary metrics."""
        n = len(query_results)
        if n == 0:
            return EvalReport(n_queries=0, hit_at_1=0, hit_at_3=0,
                              hit_at_5=0, mrr=0, precision_at_5=0)

        return EvalReport(
            n_queries=n,
            hit_at_1=sum(r.hit_at_1 for r in query_results) / n,
            hit_at_3=sum(r.hit_at_3 for r in query_results) / n,
            hit_at_5=sum(r.hit_at_5 for r in query_results) / n,
            mrr=sum(r.reciprocal_rank for r in query_results) / n,
            precision_at_5=sum(r.precision_at_5 for r in query_results) / n,
            query_results=query_results
        )


# ------------------------------------------------------------------
# Eval set loader
# ------------------------------------------------------------------

def load_eval_set(path: str) -> list[EvalQuery]:
    """
    Load golden eval set from JSON file.

    Expected JSON format:
        [
            {
                "query_id": "q001",
                "query": "How do I reset my VPN credentials?",
                "expected_chunk_ids": ["kb_12345_chunk_2"],
                "department_id": "IT_OPS",
                "category": "procedural"
            },
            ...
        ]

    Interview talking point:
        The eval set is version-controlled alongside the code.
        When we update chunk boundaries (re-chunk the corpus), chunk IDs change —
        we must re-verify ground truth manually. Stale chunk IDs silently zero out
        your Hit@k metrics. This is the most common eval pipeline bug.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        EvalQuery(
            query_id=item["query_id"],
            query=item["query"],
            expected_chunk_ids=item["expected_chunk_ids"],
            department_id=item.get("department_id"),
            category=item.get("category", "general")
        )
        for item in data
    ]


def save_eval_set(eval_set: list[EvalQuery], path: str) -> None:
    """Save eval set to JSON."""
    data = [
        {
            "query_id": eq.query_id,
            "query": eq.query,
            "expected_chunk_ids": eq.expected_chunk_ids,
            "department_id": eq.department_id,
            "category": eq.category
        }
        for eq in eval_set
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(eval_set)} eval queries to {path}")
