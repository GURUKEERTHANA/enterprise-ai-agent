"""
build_index.py — Full ingestion pipeline: load → chunk → embed → index.

Usage:
    python scripts/build_index.py \
        --kb-csv data/raw/kb_knowledge.csv \
        --incident-csv data/raw/incident.csv \
        --chroma-path data/processed/chromadb
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import chromadb
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from src.itsm_agent.ingestion.loader import load_kb_articles, clean_html
from src.itsm_agent.ingestion.chunker import (
    build_kb_chunks,
    build_incident_docs,
    chunk_articles,
)


def embed_and_index(docs: list[dict], collection, id_key: str, batch_size: int = 500):
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    total = len(docs)
    for i in tqdm(range(0, total, batch_size), desc=f"Indexing {collection.name}"):
        batch = docs[i : i + batch_size]
        texts = [d["text"] for d in batch]
        ids = [d[id_key] for d in batch]
        metadatas = [{k: v for k, v in d.items() if k != "text"} for d in batch]

        while True:
            try:
                response = openai_client.embeddings.create(
                    input=texts, model="text-embedding-3-small"
                )
                break
            except Exception as e:
                print(f"Rate limit hit, retrying in 10s… ({e})")
                time.sleep(10)

        embeddings = [r.embedding for r in response.data]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB index from raw CSVs")
    parser.add_argument("--kb-csv", default="data/raw/kb_knowledge.csv")
    parser.add_argument("--incident-csv", default="data/raw/incident.csv")
    parser.add_argument("--chroma-path", default="data/processed/chromadb")
    args = parser.parse_args()

    chroma = chromadb.PersistentClient(path=args.chroma_path)
    kb_col = chroma.get_or_create_collection("kb_articles")
    inc_col = chroma.get_or_create_collection("incidents")

    # --- KB articles ---
    print(f"\nLoading KB articles from {args.kb_csv}…")
    df_kb = pd.read_csv(args.kb_csv, encoding="latin-1")
    df_kb["clean_text"] = df_kb["text"].apply(
        lambda x: clean_html(x) if isinstance(x, str) else ""
    )
    df_kb["content"] = df_kb.apply(
        lambda row: row["clean_text"] if row["clean_text"] else row.get("short_description", ""),
        axis=1,
    )
    kb_chunks = build_kb_chunks(df_kb)
    print(f"Total KB chunks after dedup: {len(kb_chunks)}")
    embed_and_index(kb_chunks, kb_col, id_key="chunk_id")

    # --- Incidents ---
    print(f"\nLoading incidents from {args.incident_csv}…")
    df_inc = pd.read_csv(args.incident_csv, encoding="latin-1")
    inc_docs = build_incident_docs(df_inc)
    # Rename incident_id → chunk_id for embed_and_index compatibility
    for d in inc_docs:
        d["chunk_id"] = d.pop("incident_id")
    print(f"Total incident docs: {len(inc_docs)}")
    embed_and_index(inc_docs, inc_col, id_key="chunk_id", batch_size=100)

    print(f"\nDone. KB: {kb_col.count()} | Incidents: {inc_col.count()}")


if __name__ == "__main__":
    main()
