"""
run_agent.py — CLI for running a single query through the ITSM RAG Agent.

Usage:
    python scripts/run_agent.py \
        --query "I can't access Okta, how do I fix it?" \
        --tenant "DT-GPS"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.itsm_agent.agent.graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="Run a query through the ITSM agent")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument(
        "--tenant",
        required=True,
        help="Verified tenant/department ID (e.g. 'DT-GPS')",
    )
    args = parser.parse_args()

    app = build_graph()
    print("Graph compiled successfully.")

    inputs = {
        "messages": [("user", args.query)],
        "verified_tenant_id": args.tenant,
        "kb_results": [],
        "incident_results": [],
        "security_violation": False,
        "blocked": False,
        "escalate": False,
    }

    print(f"\nQuery:  {args.query}")
    print(f"Tenant: {args.tenant}")
    print("─" * 60)

    final_state = None
    for output in app.stream(inputs):
        final_state = output

    if final_state:
        node_name = list(final_state.keys())[-1]
        state = final_state[node_name]
        answer = state.get("final_answer", "(no answer generated)")
        print(f"\nAnswer:\n{answer}")
    else:
        print("No output from agent.")


if __name__ == "__main__":
    main()
