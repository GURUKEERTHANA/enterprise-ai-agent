# src/itsm_agent/ingestion/chunker.py
import re
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_articles(text: str, article_id: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    if not text or len(text) == 0:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [
        {
            "text": chunk,
            "chunk_id": f"{article_id}_chunk_{i}",
            "article_id": article_id,
        }
        for i, chunk in enumerate(chunks)
    ]


def build_kb_chunks(df_kb: pd.DataFrame) -> list[dict]:
    all_chunks = []
    for row in df_kb.itertuples():
        chunks = chunk_articles(row.content, row.number)
        for chunk in chunks:
            chunk["department_id"] = row.kb_category
            chunk["source_type"] = "kb_article"
            chunk["knowledge_base"] = row.kb_knowledge_base
        all_chunks.extend(chunks)

    seen_ids: set[str] = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk["chunk_id"] not in seen_ids:
            seen_ids.add(chunk["chunk_id"])
            unique_chunks.append(chunk)

    return unique_chunks


def clean_incident_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    boilerplate_patterns = [
        r"Hello, I am ServiceNow AI.*?(?=Issue:|Steps:|Solution:|Best regards|$)",
        r"We are moving your Incident to Resolved.*?(?=Issue:|Steps:|Solution:|$)",
        r"If we have addressed your concerns.*?(?=Issue:|Steps:|Solution:|$)",
        r"you may accept the solution.*?(?=Issue:|Steps:|Solution:|$)",
        r"Best regards,\nServiceNow AI.*$",
        r"Best regards,\nServiceNow Support Engineer.*$",
    ]
    cleaned = text
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return " ".join(cleaned.split())


def build_incident_doc(row) -> dict:
    def safe_str(val):
        return str(val).strip() if pd.notna(val) else ""

    full_text = (
        safe_str(row["short_description"])
        + " "
        + safe_str(row["description"])
        + " "
        + clean_incident_text(safe_str(row["close_notes"]))
    )
    full_text = " ".join(full_text.split())
    return {
        "text": full_text,
        "incident_id": row["number"],
        "department_id": str(row["assignment_group"]).strip(),
        "source_type": "incident",
        "priority": row["priority"],
        "category": row["category"],
    }


def build_incident_docs(df_incidents: pd.DataFrame) -> list[dict]:
    return [build_incident_doc(row) for _, row in df_incidents.iterrows()]
