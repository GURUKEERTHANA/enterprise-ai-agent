"""
Hybrid Retriever — RRF fusion of BM25 + ChromaDB dense retrieval.

Implements Reciprocal Rank Fusion (RRF) to combine ranked results from
two complementary retrievers into a single merged ranking.

Notes:
    BM25 catches exact terminology (incident IDs, error codes, product names).
    Dense retrieval catches semantic similarity (paraphrased questions, conceptual queries).
    RRF is rank-based not score-based — this is critical because BM25 scores (e.g. 18.4)
    and cosine similarity scores (e.g. 0.73) are on completely different scales.
    You cannot average them. RRF sidesteps this by only using rank position.

RRF formula:
    RRF_score(doc) = Σ_retriever  1 / (k + rank_in_retriever)
    where k=60 is a constant that dampens the impact of very high ranks.
    k=60 means rank 1 contributes 1/61, rank 2 contributes 1/62, etc.
    This prevents any single top result from dominating the fusion.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .bm25_retriever import BM25Retriever, BM25Result


@dataclass
class HybridResult:
    """Single result from hybrid retrieval with RRF score."""
    chunk_id: str
    text: str
    rrf_score: float
    bm25_rank: Optional[int] = None    # None if not in BM25 results
    dense_rank: Optional[int] = None   # None if not in dense results
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword search and ChromaDB dense retrieval
    via Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever(bm25, chroma_collection, embedder)
        retriever.index(chunks)  # builds BM25 index (ChromaDB already indexed)
        results = retriever.retrieve("database connection error", top_k=5,
                                     department_id="IT_OPS")
    """

    RRF_K = 60  # Standard RRF constant. Changing this rarely helps — don't tune it.

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        chroma_collection,          # chromadb.Collection
        embedder,                   # callable: text -> list[float]
        async_embedder: Optional[Callable[[str], Awaitable[list[float]]]] = None,
        bm25_top_k: int = 20,       # candidates from BM25 before fusion
        dense_top_k: int = 20,      # candidates from ChromaDB before fusion
    ):
        self.bm25 = bm25_retriever
        self.chroma = chroma_collection
        self.embedder = embedder
        self.async_embedder = async_embedder
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k

    def index(self, chunks: list[dict]) -> None:
        """
        Build BM25 index. ChromaDB collection is assumed to be pre-indexed.

        Args:
            chunks: Same chunk list used to populate ChromaDB. BM25 needs
                    the raw text since ChromaDB only stores embeddings + metadata.
        """
        self.bm25.index(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        department_id: Optional[str] = None
    ) -> list[HybridResult]:
        """
        Retrieve top-k chunks using RRF-fused hybrid search.

        Pipeline:
            1. BM25 retrieves top bm25_top_k candidates (keyword match)
            2. ChromaDB retrieves top dense_top_k candidates (semantic match)
            3. RRF merges both ranked lists into a single ranking
            4. Return top_k from merged ranking

        Args:
            query: User query string.
            top_k: Final number of results to return.
            department_id: Tenant filter applied to BOTH retrievers independently.

        Returns:
            List of HybridResult sorted by RRF score descending.

        Notes:
            We retrieve 20 candidates from each retriever (not just 5) before fusion.
            This is because a document ranked #15 in BM25 but #1 in dense retrieval
            should score well after fusion. Fetching only top_k from each would miss it.
            The 20→5 funnel is the right pattern.
        """
        # --- Step 1: BM25 retrieval ---
        bm25_results: list[BM25Result] = self.bm25.retrieve(
            query, top_k=self.bm25_top_k, department_id=department_id
        )

        # --- Step 2: Dense retrieval via ChromaDB ---
        dense_results = self._dense_retrieve(query, department_id=department_id)

        # --- Step 3: RRF fusion ---
        return self._rrf_fuse(bm25_results, dense_results, top_k=top_k)

    async def aretrieve(
        self,
        query: str,
        top_k: int = 5,
        department_id: Optional[str] = None,
        profiler=None,
    ) -> list[HybridResult]:
        """
        Async hybrid retrieval. Runs BM25 (CPU-bound, off-loop) and dense
        retrieval (network I/O) concurrently via asyncio.gather, then fuses
        with RRF. Cuts wall-clock retrieval to max(BM25, dense) instead of sum.

        If a profiler is provided, inner stages (bm25_retrieval, embedding_query,
        chroma_retrieval, rrf_fusion) are timed individually so the latency
        breakdown surfaces the embedding-API share separately from local compute.
        """
        async def timed_bm25():
            if profiler is not None:
                with profiler.measure("bm25_retrieval"):
                    return await asyncio.to_thread(
                        self.bm25.retrieve, query, self.bm25_top_k, department_id
                    )
            return await asyncio.to_thread(
                self.bm25.retrieve, query, self.bm25_top_k, department_id
            )

        bm25_results, dense_results = await asyncio.gather(
            timed_bm25(),
            self._adense_retrieve(query, department_id=department_id, profiler=profiler),
        )

        if profiler is not None:
            with profiler.measure("rrf_fusion"):
                return self._rrf_fuse(bm25_results, dense_results, top_k=top_k)
        return self._rrf_fuse(bm25_results, dense_results, top_k=top_k)

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    def _dense_retrieve(
        self,
        query: str,
        department_id: Optional[str] = None
    ) -> list[dict]:
        """
        Query ChromaDB with pre-computed query embedding.

        Returns list of dicts with keys: chunk_id, text, score, metadata.

        Notes:
            We pass query_embeddings (pre-computed OpenAI vector), NOT query_texts.
            ChromaDB's built-in embedding model is 384-dim; OpenAI text-embedding-3-small
            is 1536-dim. Passing query_texts would cause a dimension mismatch error.
            Always pre-compute embeddings with the same model used at ingestion time.
        """
        # Build metadata filter for tenant isolation
        where_filter = None
        if department_id:
            where_filter = {"department_id": {"$eq": department_id}}

        # Embed the query using the same model as ingestion
        query_embedding = self.embedder(query)

        # Query ChromaDB
        results = self.chroma.query(
            query_embeddings=[query_embedding],
            n_results=self.dense_top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Normalize output format
        dense_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns L2 distance; convert to similarity score
                distance = results["distances"][0][i]
                similarity = 1.0 / (1.0 + distance)  # monotone conversion

                dense_results.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "score": similarity,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })

        return dense_results

    async def _adense_retrieve(
        self,
        query: str,
        department_id: Optional[str] = None,
        profiler=None,
    ) -> list[dict]:
        """
        Async dense retrieval. The embedding call is true async I/O via
        AsyncOpenAI; the ChromaDB query is sync and off-loaded to a thread.
        """
        where_filter = None
        if department_id:
            where_filter = {"department_id": {"$eq": department_id}}

        if profiler is not None:
            with profiler.measure("embedding_query"):
                if self.async_embedder is not None:
                    query_embedding = await self.async_embedder(query)
                else:
                    query_embedding = await asyncio.to_thread(self.embedder, query)
        else:
            if self.async_embedder is not None:
                query_embedding = await self.async_embedder(query)
            else:
                query_embedding = await asyncio.to_thread(self.embedder, query)

        chroma_call = asyncio.to_thread(
            lambda: self.chroma.query(
                query_embeddings=[query_embedding],
                n_results=self.dense_top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        )
        if profiler is not None:
            with profiler.measure("chroma_retrieval"):
                results = await chroma_call
        else:
            results = await chroma_call

        dense_results: list[dict] = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1.0 / (1.0 + distance)
                dense_results.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "score": similarity,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })
        return dense_results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        bm25_results: list[BM25Result],
        dense_results: list[dict],
        top_k: int
    ) -> list[HybridResult]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF formula:
            score(doc) = 1/(k + rank_bm25) + 1/(k + rank_dense)
            where rank is 1-indexed, k=60

        Documents appearing in only one list still get a score from that list.
        Documents appearing in both lists get additive scores — they are rewarded
        for cross-retriever agreement.

        Notes:
            If a chunk appears in BOTH BM25 top-5 and dense top-5, it almost certainly
            deserves to be in the final top-5. RRF captures this intuition mathematically.
            Score from BM25-only rank-1: 1/61 ≈ 0.0164
            Score from dense-only rank-1: 1/61 ≈ 0.0164
            Score from both rank-1: 2/61 ≈ 0.0328 — double, as expected.
        """
        # Map chunk_id → HybridResult for accumulation
        fused: dict[str, HybridResult] = {}

        # --- BM25 contributions ---
        for rank, result in enumerate(bm25_results, start=1):
            rrf_contribution = 1.0 / (self.RRF_K + rank)
            if result.chunk_id not in fused:
                fused[result.chunk_id] = HybridResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    rrf_score=0.0,
                    metadata=result.metadata
                )
            fused[result.chunk_id].rrf_score += rrf_contribution
            fused[result.chunk_id].bm25_rank = rank
            fused[result.chunk_id].bm25_score = result.score

        # --- Dense contributions ---
        for rank, result in enumerate(dense_results, start=1):
            rrf_contribution = 1.0 / (self.RRF_K + rank)
            chunk_id = result["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = HybridResult(
                    chunk_id=chunk_id,
                    text=result["text"],
                    rrf_score=0.0,
                    metadata=result["metadata"]
                )
            fused[chunk_id].rrf_score += rrf_contribution
            fused[chunk_id].dense_rank = rank
            fused[chunk_id].dense_score = result["score"]

        # Sort by RRF score descending and return top_k
        sorted_results = sorted(fused.values(), key=lambda x: x.rrf_score, reverse=True)
        return sorted_results[:top_k]
