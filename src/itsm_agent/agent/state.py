# src/itsm_agent/agent/state.py
import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph import add_messages

from src.itsm_agent.utils.latency import LatencyProfiler


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]          # full conversation (LangGraph managed)
    verified_tenant_id: str                          # ACL-enforced department for this session
    route: Optional[object]                          # RouteAction from router_node
    kb_results: Annotated[list[str], operator.add]   # accumulated KB context chunks
    incident_results: Annotated[list[str], operator.add]  # accumulated incident context chunks
    final_answer: str                                # synthesized response sent to the user
    security_violation: bool                         # set by workers on cross-tenant access
    blocked: bool                                    # set by validate_input on injection detection
    block_reason: Optional[str]                      # injection type string if blocked
    profiler: Optional[LatencyProfiler]              # per-request latency tracker
    escalate: bool                                   # True when confidence gate triggers handoff
