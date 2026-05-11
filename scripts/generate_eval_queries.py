"""
generate_eval_queries.py — Expand the golden eval set with synthetic, validated queries.

For each sampled KB chunk:
  1. Ask GPT-4o for a paraphrased question whose answer is in that chunk.
  2. Run the question through the same HybridRetriever the eval uses.
  3. Keep only queries where the source chunk lands in the top-20 — otherwise
     the ground truth is unreachable and the eval would be measuring noise.

The accepted queries are merged onto the existing hand-curated eval set so the
final JSON contains both methodologies. The "category" field on synthetic
queries is set to "synthetic" to keep the source of each query traceable.

Usage:
    python scripts/generate_eval_queries.py --n 37
"""
import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import chromadb
from openai import AsyncOpenAI

from src.itsm_agent.retrieval.bm25_retriever import BM25Retriever
from src.itsm_agent.retrieval.chroma_retriever import aembed_text
from src.itsm_agent.retrieval.hybrid_retriever import HybridRetriever

PROMPT = """\
Below is an excerpt from an enterprise knowledge base article.
Write a SINGLE natural question that an employee might ask whose answer is contained in this excerpt.

REQUIREMENTS:
- Paraphrase — do NOT copy specific phrases verbatim from the excerpt.
- Frame it as something a normal employee would type into a support search box.
- The question must be answerable from this excerpt, not from general knowledge.
- Maximum 25 words.

EXCERPT:
\"\"\"
{chunk_text}
\"\"\"

Return only the question text. No preamble, no quotes, no explanation.
"""


async def generate_query(client: AsyncOpenAI, chunk_text: str) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": PROMPT.format(chunk_text=chunk_text[:1500])}],
        temperature=0.7,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip().strip('"').strip("'")


async def main(
    n_target: int,
    chroma_path: str,
    out_path: str,
    seed: int,
    min_chars: int,
    min_dept_size: int,
    max_per_dept: int,
) -> None:
    client = chromadb.PersistentClient(path=chroma_path)
    kb = client.get_collection("kb_articles")
    data = kb.get(include=["documents", "metadatas"])

    chunks = [
        {"chunk_id": cid, "text": doc, "metadata": meta or {}}
        for cid, doc, meta in zip(
            data["ids"],
            data["documents"],
            data["metadatas"] or [{}] * len(data["ids"]),
        )
    ]
    print(f"Loaded {len(chunks)} chunks from kb_articles")

    bm25 = BM25Retriever()
    bm25.index(chunks)
    hybrid = HybridRetriever(
        bm25,
        kb,
        embedder=lambda t: None,
        async_embedder=aembed_text,
    )

    candidates = [
        c
        for c in chunks
        if c["metadata"].get("department_id", "UNKNOWN") != "UNKNOWN"
        and len(c["text"]) >= min_chars
    ]
    by_dept: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_dept[c["metadata"]["department_id"]].append(c)

    big_depts = [d for d, items in by_dept.items() if len(items) >= min_dept_size]
    rng = random.Random(seed)
    rng.shuffle(big_depts)

    sampled: list[dict] = []
    for dept in big_depts:
        items = list(by_dept[dept])
        rng.shuffle(items)
        sampled.extend(items[:max_per_dept])
        if len(sampled) >= n_target * 3:
            break

    print(
        f"Sampled {len(sampled)} candidate chunks "
        f"from {len(big_depts)} departments (target {n_target})"
    )

    oai = AsyncOpenAI()
    accepted: list[dict] = []
    rejected = 0

    next_id_start = _next_query_id(out_path)

    for i, chunk in enumerate(sampled):
        if len(accepted) >= n_target:
            break
        try:
            query = await generate_query(oai, chunk["text"])
        except Exception as e:
            print(f"  [{i:3d}] generation failed: {e}")
            continue

        try:
            results = await hybrid.aretrieve(
                query,
                top_k=20,
                department_id=chunk["metadata"]["department_id"],
            )
        except Exception as e:
            print(f"  [{i:3d}] retrieval failed: {e}")
            continue

        retrieved_ids = [r.chunk_id for r in results]
        if chunk["chunk_id"] not in retrieved_ids:
            rejected += 1
            print(
                f"  [reject {i:3d}] source not in top-20 — "
                f"chunk={chunk['chunk_id']} q='{query[:70]}'"
            )
            continue

        rank = retrieved_ids.index(chunk["chunk_id"]) + 1
        qid = f"q{next_id_start + len(accepted):03d}"
        accepted.append(
            {
                "query_id": qid,
                "query": query,
                "expected_chunk_ids": [chunk["chunk_id"]],
                "department_id": chunk["metadata"]["department_id"],
                "category": "synthetic",
            }
        )
        print(
            f"  [accept {len(accepted):2d}/{n_target}] {qid} "
            f"chunk={chunk['chunk_id']} rank={rank} q='{query[:70]}'"
        )

    print(f"\nAccepted {len(accepted)} queries (rejected {rejected})")

    with open(out_path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    merged = existing + accepted

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(merged)} queries total to {out_path}")


def _next_query_id(eval_path: str) -> int:
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        return 1
    max_id = 0
    for item in existing:
        qid = item.get("query_id", "")
        if qid.startswith("q") and qid[1:].isdigit():
            max_id = max(max_id, int(qid[1:]))
    return max_id + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=37, help="Number of synthetic queries to accept")
    parser.add_argument("--chroma-path", default="data/processed/chromadb")
    parser.add_argument("--out", default="src/itsm_agent/eval/eval_set.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--min-dept-size", type=int, default=50)
    parser.add_argument("--max-per-dept", type=int, default=3)
    args = parser.parse_args()

    asyncio.run(
        main(
            n_target=args.n,
            chroma_path=args.chroma_path,
            out_path=args.out,
            seed=args.seed,
            min_chars=args.min_chars,
            min_dept_size=args.min_dept_size,
            max_per_dept=args.max_per_dept,
        )
    )
