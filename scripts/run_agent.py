"""
run_agent.py — CLI for running a single query through the ITSM RAG Agent.

Usage:
    python scripts/run_agent.py \
        --query "I can't access Okta, how do I fix it?" \
        --tenant "DT-GPS"
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.itsm_agent.agent.graph import build_graph


async def run(query: str, tenant: str) -> None:
    app = build_graph()
    print("Graph compiled successfully.")

    inputs = {
        "messages": [("user", query)],
        "verified_tenant_id": tenant,
        "kb_results": [],
        "incident_results": [],
        "security_violation": False,
        "blocked": False,
        "escalate": False,
    }

    print(f"\nQuery:  {query}")
    print(f"Tenant: {tenant}")
    print("─" * 60)

    final_state = None
    async for output in app.astream(inputs):
        final_state = output

    if final_state:
        node_name = list(final_state.keys())[-1]
        state = final_state[node_name]
        answer = state.get("final_answer", "(no answer generated)")
        print(f"\nAnswer:\n{answer}")
    else:
        print("No output from agent.")


def main():
    parser = argparse.ArgumentParser(description="Run a query through the ITSM agent")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument(
        "--tenant",
        required=True,
        help="Verified tenant/department ID (e.g. 'DT-GPS')",
    )
    args = parser.parse_args()

    asyncio.run(run(args.query, args.tenant))


if __name__ == "__main__":
    main()
