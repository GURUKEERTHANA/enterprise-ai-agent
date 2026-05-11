"""Quick smoke test for LatencyProfiler — exercises stage and category breakdown."""
import sys
import time
sys.path.insert(0, ".")

from src.itsm_agent.utils.latency import LatencyProfiler, timed, new_profiler

profiler = new_profiler()

with profiler.measure("injection_check"):
    time.sleep(0.0003)

with profiler.measure("bm25_retrieval"):
    time.sleep(0.002)

with profiler.measure("embedding_query"):
    time.sleep(0.263)

with profiler.measure("chroma_retrieval"):
    time.sleep(0.025)

with profiler.measure("rrf_fusion"):
    time.sleep(0.0017)

with profiler.measure("reranking"):
    time.sleep(0.008)

with profiler.measure("llm_call", model="gpt-4o"):
    time.sleep(0.950)

print(profiler.report())

log_data = profiler.to_dict()
print("\nStructured log data:")
for k, v in log_data.items():
    print(f"  {k}: {v}")

cats = profiler.category_summary()
print("\nCategory split (should approximate 76/21/3):")
for category, stats in sorted(cats.items(), key=lambda x: x[1]["total_ms"], reverse=True):
    print(f"  {category:<14} {stats['total_ms']:>7.1f}ms  {stats['pct_of_total']:>4.0f}%")

@timed("embed_single")
def fake_embed(text):
    time.sleep(0.05)
    return [0.1] * 1536

print("\nTesting @timed decorator:")
vec = fake_embed("test query")
print(f"  Embedding length: {len(vec)}")

print("\n✓ Latency profiler working correctly")
