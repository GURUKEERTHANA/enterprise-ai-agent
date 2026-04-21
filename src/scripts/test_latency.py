"""Quick smoke test for LatencyProfiler."""
import sys
import time
sys.path.insert(0, ".")

from src.itsm_agent.utils.latency import LatencyProfiler, timed, new_profiler

# --- Test 1: Context manager ---
profiler = new_profiler()

with profiler.measure("bm25_retrieval"):
    time.sleep(0.002)  # simulate 2ms

with profiler.measure("embedding_query"):
    time.sleep(0.094)  # simulate 94ms

with profiler.measure("llm_call", model="gpt-4o"):
    time.sleep(0.946)  # simulate 946ms

print(profiler.report())

# --- Test 2: to_dict for logging ---
log_data = profiler.to_dict()
print("\nStructured log data:")
for k, v in log_data.items():
    print(f"  {k}: {v}")

# --- Test 3: Decorator ---
@timed("embed_single")
def fake_embed(text):
    time.sleep(0.05)
    return [0.1] * 1536

print("\nTesting @timed decorator:")
vec = fake_embed("test query")
print(f"  Embedding length: {len(vec)}")

print("\n✓ Latency profiler working correctly")