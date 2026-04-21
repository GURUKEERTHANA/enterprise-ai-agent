# src/itsm_agent/ingestion/loader.py

import csv
import re
from bs4 import BeautifulSoup
from typing import Optional


def load_kb_articles(csv_path: str) -> list[dict]:
    """
    Load KB articles from ServiceNow CSV export.
    Handles latin-1 encoding (Windows-1252 default from ServiceNow exports).
    """
    articles = []
    with open(csv_path, encoding="latin-1", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = get_content(row)
            if content:
                articles.append({
                    "sys_id": row.get("sys_id", ""),
                    "number": row.get("number", ""),
                    "short_description": row.get("short_description", ""),
                    "content": content,
                    "department_id": row.get("assignment_group", "IT_OPS"),
                    "source": "kb_article"
                })
    return articles


def get_content(row: dict) -> Optional[str]:
    """
    Extract clean text from KB article row.
    Falls back to short_description if body text is empty.
    This handles the ~2,019 articles with no body text.
    """
    body = row.get("text", "") or row.get("wiki", "") or ""
    if body.strip():
        return clean_html(body)
    # Fallback to short_description
    fallback = row.get("short_description", "").strip()
    return fallback if fallback else None


def clean_html(html: str) -> str:
    """Strip HTML tags using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text