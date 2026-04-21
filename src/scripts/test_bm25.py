"""Quick smoke test for BM25Retriever."""
import sys
sys.path.insert(0, ".")

from src.itsm_agent.retrieval.bm25_retriever import BM25Retriever

# --- Fake chunks to test with (no real data needed) ---
chunks = [
    {
        "chunk_id": "chunk_001",
        "text": "To reset VPN credentials, navigate to the IT portal and click forgot password.",
        "metadata": {"department_id": "IT_OPS", "source": "KB"}
    },
    {
        "chunk_id": "chunk_002",
        "text": "Password expiry policy for service accounts is 90 days.",
        "metadata": {"department_id": "IT_OPS", "source": "KB"}
    },
    {
        "chunk_id": "chunk_003",
        "text": "Outlook mobile sync issues can be resolved by removing and re-adding the account.",
        "metadata": {"department_id": "IT_OPS", "source": "KB"}
    },
    {
        "chunk_id": "chunk_004",
        "text": "Finance department budget approvals require CFO sign-off.",
        "metadata": {"department_id": "FINANCE", "source": "KB"}
    },
]

# --- Build index ---
bm25 = BM25Retriever()
bm25.index(chunks)
print(f"✓ Indexed {len(chunks)} chunks")

# --- Test 1: Basic retrieval ---
results = bm25.retrieve("VPN reset password", top_k=3)



print(f"\nQuery: 'VPN reset password'")
for r in results:
    print(f"  [{r.chunk_id}] score={r.score:.3f} | {r.text[:60]}...")

assert results[0].chunk_id == "chunk_001", "VPN chunk should be rank 1"
print("✓ Correct chunk ranked first")

# --- Test 2: Tenant filtering ---
results_finance = bm25.retrieve("budget approval", top_k=3, department_id="FINANCE")
results_it = bm25.retrieve("budget approval", top_k=3, department_id="IT_OPS")

print(f"\nTenant filter test:")
print(f"  FINANCE results: {[r.chunk_id for r in results_finance]}")
print(f"  IT_OPS results: {[r.chunk_id for r in results_it]}")

assert all(r.metadata["department_id"] == "FINANCE" for r in results_finance), \
    "Finance filter should only return FINANCE chunks"
print("✓ Tenant isolation working")

# --- Test 3: Empty query ---
results_empty = bm25.retrieve("xyzzy nonsense", top_k=3)
print(f"\nOut-of-scope query results: {len(results_empty)} (expected 0 or low scores)")

print("\n✓ All BM25 tests passed")