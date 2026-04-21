"""
debug_chroma.py — Inspect ChromaDB collections and run ad-hoc vector queries.

Usage:
    # Count docs per department
    python scripts/debug_chroma.py --collection incidents --department DT-GPS

    # Run a test query
    python scripts/debug_chroma.py --collection kb_articles \
        --query "how to reset password" --department "Software"
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import chromadb
from openai import OpenAI

CHROMA_PATH = os.environ.get("CHROMA_PATH", "data/processed/chromadb")


def main():
    parser = argparse.ArgumentParser(description="Debug ChromaDB collections")
    parser.add_argument(
        "--collection",
        default="incidents",
        choices=["incidents", "kb_articles"],
        help="Which collection to inspect",
    )
    parser.add_argument("--department", default=None, help="Filter by department_id")
    parser.add_argument("--query", default=None, help="Optional semantic search query")
    parser.add_argument("--n-results", type=int, default=3, help="Number of results")
    args = parser.parse_args()

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    col = chroma.get_collection(args.collection)
    print(f"Collection '{args.collection}' total count: {col.count()}")

    if args.department:
        results = col.get(
            where={"department_id": args.department},
            include=["metadatas"],
        )
        print(f"Documents for department '{args.department}': {len(results['ids'])}")
        if results["ids"]:
            print(f"Sample metadata: {results['metadatas'][0]}")

    if args.query:
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = openai_client.embeddings.create(
            input=[args.query], model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=args.n_results,
        )
        if args.department:
            kwargs["where"] = {"department_id": args.department}

        query_results = col.query(**kwargs)
        print(f"\nTop {args.n_results} results for: '{args.query}'")
        print("─" * 60)
        for doc, meta in zip(
            query_results["documents"][0], query_results["metadatas"][0]
        ):
            dept = meta.get("department_id", "N/A")
            print(f"[{dept}] {doc[:150]}")
            print()


if __name__ == "__main__":
    main()
