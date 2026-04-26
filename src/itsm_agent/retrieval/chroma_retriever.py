# src/itsm_agent/retrieval/chroma_retriever.py
import os
import chromadb
from openai import OpenAI

_CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/processed/chromadb")

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)

kb_collection = chroma_client.get_or_create_collection(name="kb_articles")
incident_collection = chroma_client.get_or_create_collection(name="incidents")


def query_collection(
    collection,
    query: str,
    n_results: int = 3,
    where: dict = None,
) -> dict:
    response = openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
    )
    query_embedding = response.data[0].embedding

    kwargs = dict(query_embeddings=[query_embedding], n_results=n_results)
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding


def _load_chunks(collection) -> list[dict]:
    result = collection.get(include=["documents", "metadatas"])
    if not result["ids"]:
        return []
    metadatas = result["metadatas"] or [{}] * len(result["ids"])
    return [
        {"chunk_id": cid, "text": doc, "metadata": meta}
        for cid, doc, meta in zip(result["ids"], result["documents"], metadatas)
    ]


from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever


def _build_hybrid(collection, dense_top_k: int = 20) -> HybridRetriever:
    bm25 = BM25Retriever()
    bm25.index(_load_chunks(collection))
    return HybridRetriever(bm25, collection, embed_text, bm25_top_k=20, dense_top_k=dense_top_k)


kb_hybrid = _build_hybrid(kb_collection)
incident_hybrid = _build_hybrid(incident_collection)
