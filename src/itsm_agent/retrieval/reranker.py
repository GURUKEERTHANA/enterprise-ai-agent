# src/itsm_agent/retrieval/reranker.py
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[str], top_k: int = 3) -> list[str]:
    if not candidates:
        return []
    pairs = [[query, doc] for doc in candidates]
    scores = reranker_model.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
