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
