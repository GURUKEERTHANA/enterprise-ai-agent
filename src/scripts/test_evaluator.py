"""Quick smoke test for RAGEvaluator using a mock retriever."""
import sys
sys.path.insert(0, ".")

from dataclasses import dataclass
from src.itsm_agent.eval.evaluator import RAGEvaluator, EvalQuery

# --- Mock retriever: returns predetermined results ---
@dataclass
class MockChunk:
    chunk_id: str
    rrf_score: float = 1.0

class MockRetriever:
    """Returns hardcoded results for predictable testing."""
    def retrieve(self, query, top_k=5, department_id=None):
        # Simulate: first query hits at rank 1, second at rank 2, third misses
        results_map = {
            "How do I reset my VPN credentials?": [
                MockChunk("kb_001"), MockChunk("kb_002"), MockChunk("kb_003"),
                MockChunk("kb_004"), MockChunk("kb_005")
            ],
            "What is the password expiry policy?": [
                MockChunk("kb_999"), MockChunk("kb_002"), MockChunk("kb_003"),
                MockChunk("kb_004"), MockChunk("kb_005")
            ],
            "Outlook not syncing": [
                MockChunk("kb_999"), MockChunk("kb_888"), MockChunk("kb_777"),
                MockChunk("kb_666"), MockChunk("kb_555")
            ],
        }
        return results_map.get(query, [])

eval_set = [
    EvalQuery("q1", "How do I reset my VPN credentials?",
              expected_chunk_ids=["kb_001"], department_id="IT_OPS"),
    EvalQuery("q2", "What is the password expiry policy?",
              expected_chunk_ids=["kb_002"], department_id="IT_OPS"),
    EvalQuery("q3", "Outlook not syncing",
              expected_chunk_ids=["kb_003"], department_id="IT_OPS"),
]

# Expected:
#   q1: Hit@1=True, RR=1.0 (found at rank 1)
#   q2: Hit@1=False, Hit@3=True, RR=0.5 (found at rank 2)
#   q3: Miss, RR=0.0

evaluator = RAGEvaluator(retriever=MockRetriever(), top_k=5)
report = evaluator.evaluate(eval_set, verbose=True)
print(report)

# Validate expected values
assert abs(report.hit_at_1 - 0.333) < 0.01, f"Hit@1 should be 0.333, got {report.hit_at_1}"
assert abs(report.hit_at_3 - 0.667) < 0.01, f"Hit@3 should be 0.667, got {report.hit_at_3}"
assert abs(report.mrr - 0.5) < 0.01, f"MRR should be 0.5, got {report.mrr}"

print("\n✓ Evaluator metrics are mathematically correct")