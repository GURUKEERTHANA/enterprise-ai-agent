"""
BM25 Retriever for ITSM corpus.

Implements BM25 (Best Match 25) ranking over pre-chunked ITSM documents.
Used as the keyword/exact-match leg of hybrid retrieval.

Interview talking point:
    BM25 catches exact terminology — incident numbers, error codes, product names —
    that semantic embeddings miss because they compress meaning into vectors.
    In ITSM, a user searching for 'ORA-12541 error' needs exact match, not semantic similarity.
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BM25Result:
    """Single retrieval result from BM25."""
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class BM25Retriever:
    """
    BM25 retriever over ITSM document chunks.

    BM25 parameters:
        k1 (float): Term frequency saturation. Default 1.5.
                    Higher = more weight to term frequency.
                    Lower = faster saturation (diminishing returns on repeat terms).
        b  (float): Document length normalization. Default 0.75.
                    1.0 = full normalization. 0.0 = no normalization.

    Interview talking point:
        k1=1.5, b=0.75 are empirically validated defaults from the original paper.
        For short ITSM chunks (avg ~200 tokens), b=0.75 is appropriate — chunks are
        similar in length so length normalization matters less than in web search.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Corpus state — populated on index()
        self._chunks: list[dict] = []         # raw chunk dicts
        self._doc_freqs: dict[str, int] = {}  # term → number of docs containing term
        self._tf: list[dict[str, int]] = []   # per-chunk term frequencies
        self._doc_lengths: list[int] = []     # token count per chunk
        self._avgdl: float = 0.0              # average document length
        self._n_docs: int = 0
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, chunks: list[dict]) -> None:
        """
        Build BM25 index over a list of chunk dicts.

        Args:
            chunks: List of dicts, each must have:
                    - 'chunk_id' (str): unique identifier
                    - 'text' (str): chunk content
                    - 'metadata' (dict): department_id, source, etc.

        Complexity: O(N * L) where N = chunks, L = avg tokens per chunk.
        """
        self._chunks = chunks
        self._n_docs = len(chunks)
        self._tf = []
        self._doc_freqs = defaultdict(int)
        self._doc_lengths = []

        for chunk in chunks:
            tokens = self._tokenize(chunk["text"])
            self._doc_lengths.append(len(tokens))

            # Term frequency for this document
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self._tf.append(dict(tf))

            # Document frequency — each term counted once per doc
            for term in set(tokens):
                self._doc_freqs[term] += 1

        self._avgdl = sum(self._doc_lengths) / self._n_docs if self._n_docs > 0 else 1.0
        self._indexed = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        department_id: Optional[str] = None
    ) -> list[BM25Result]:
        """
        Retrieve top-k chunks by BM25 score.

        Args:
            query: Natural language or keyword query.
            top_k: Number of results to return.
            department_id: If provided, filter to this tenant's documents only.
                           This is the multi-tenancy guard — same pattern as
                           ChromaDB's where clause. Pre-filter, not post-filter.

        Returns:
            List of BM25Result sorted by score descending.

        Interview talking point:
            Pre-filtering by department_id BEFORE scoring is critical for security.
            Post-filtering (score all, then filter) leaks document existence across tenants
            via timing side-channels and wastes compute on unauthorized chunks.
        """
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve()")

        query_tokens = self._tokenize(query)
        scores = []

        for idx, chunk in enumerate(self._chunks):
            # Multi-tenancy: skip chunks outside the requesting department
            if department_id and chunk.get("metadata", {}).get("department_id") != department_id:
                continue

            score = self._score(query_tokens, idx)
            if score > 0:
                scores.append((score, idx))

        # Sort descending by score
        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scores[:top_k]:
            chunk = self._chunks[idx]
            results.append(BM25Result(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                score=score,
                metadata=chunk.get("metadata", {})
            ))
        return results

    # ------------------------------------------------------------------
    # BM25 scoring
    # ------------------------------------------------------------------

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a single document.

        Formula:
            score(D, Q) = Σ IDF(qi) * [TF(qi, D) * (k1 + 1)] / [TF(qi, D) + k1 * (1 - b + b * |D| / avgdl)]

        where:
            IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
            TF(qi, D) = raw term frequency of qi in document D
            |D| = document length in tokens
            avgdl = average document length across corpus
        """
        score = 0.0
        doc_tf = self._tf[doc_idx]
        doc_len = self._doc_lengths[doc_idx]

        for term in query_tokens:
            if term not in doc_tf:
                continue

            tf = doc_tf[term]
            df = self._doc_freqs.get(term, 0)

            # IDF with smoothing (Robertson-Sparck Jones variant)
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)

            # TF normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
            )

            score += idf * tf_norm

        return score

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple whitespace + punctuation tokenizer with lowercasing.

        Interview talking point:
            For ITSM, we deliberately keep numbers and alphanumeric strings intact
            (e.g., 'INC0012345', 'ORA-12541') because these are high-signal identifiers.
            A stemmer would break them. BM25's strength in ITSM is exact-match on these.
        """
        # Lowercase
        text = text.lower()
        # Split on whitespace and common punctuation, but keep alphanumeric+hyphen
        tokens = re.findall(r"[a-z0-9][a-z0-9\-]*", text)
        # Remove single characters (noise)
        return [t for t in tokens if len(t) > 1]
