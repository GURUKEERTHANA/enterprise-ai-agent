# src/itsm_agent/ingestion/indexer.py

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from tqdm import tqdm

from src.itsm_agent.guardrails.prompt_injection import check_prompt_injection


def chunk_documents(documents: list[dict], chunk_size: int = 512,
                    chunk_overlap: int = 50) -> list[dict]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Preserves metadata (department_id, source, sys_id) on each chunk.

    Drops chunks whose text matches a prompt-injection pattern (indirect
    injection from attacker-planted KB/incident content). Counts are logged
    by injection type for audit.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []
    dropped_by_type: dict[str, int] = {}
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, split in enumerate(splits):
            scan = check_prompt_injection(split)
            if scan.is_injection:
                key = scan.injection_type.value if scan.injection_type else "unknown"
                dropped_by_type[key] = dropped_by_type.get(key, 0) + 1
                continue
            chunks.append({
                "chunk_id": f"{doc['sys_id']}_chunk_{i}",
                "text": split,
                "metadata": {
                    "department_id": doc["department_id"],
                    "source": doc["source"],
                    "sys_id": doc["sys_id"],
                    "number": doc.get("number", ""),
                    "short_description": doc.get("short_description", ""),
                }
            })
    if dropped_by_type:
        total_dropped = sum(dropped_by_type.values())
        print(
            f"[INGESTION GUARD] Dropped {total_dropped} chunk(s) with injection patterns: "
            + ", ".join(f"{k}={v}" for k, v in dropped_by_type.items())
        )
    return chunks


def build_chroma_index(
    chunks: list[dict],
    chroma_path: str,
    collection_name: str = "itsm_chunks",
    openai_api_key: str = None,
    batch_size: int = 100
) -> chromadb.Collection:
    """
    Embed chunks with OpenAI text-embedding-3-small and index in ChromaDB.

    CRITICAL: Always pass query_embeddings (pre-computed vectors) when querying,
    NOT query_texts. ChromaDB's built-in model is 384-dim; OpenAI is 1536-dim.
    Mixing them causes a dimension mismatch error.
    """
    client_oai = OpenAI(api_key=openai_api_key)
    chroma = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection to rebuild cleanly
    try:
        chroma.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma.create_collection(collection_name)

    print(f"Indexing {len(chunks)} chunks into ChromaDB...")
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        # Embed with OpenAI
        response = client_oai.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        embeddings = [e.embedding for e in response.data]

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in batch]
        )

    print(f"✓ Indexed {len(chunks)} chunks")
    return collection