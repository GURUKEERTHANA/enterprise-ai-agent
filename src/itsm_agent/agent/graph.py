# src/itsm_agent/agent/graph.py
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    validate_input_node,
    router_node,
    kb_worker_node,
    incident_worker_node,
    synthesizer_node,
    security_alert_node,
    refusal_node,
    should_continue,
    check_worker_security,
)


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("router_node", router_node)
    workflow.add_node("kb_worker_node", kb_worker_node)
    workflow.add_node("incident_worker_node", incident_worker_node)
    workflow.add_node("refusal_node", refusal_node)
    workflow.add_node("synthesizer_node", synthesizer_node)
    workflow.add_node("security_alert_node", security_alert_node)

    workflow.add_edge(START, "validate_input")

    workflow.add_conditional_edges(
        "validate_input",
        lambda s: "refusal_node" if s.get("blocked") else "router_node",
    )

    workflow.add_conditional_edges(
        "router_node",
        should_continue,
        {
            "kb_worker":       "kb_worker_node",
            "incident_worker": "incident_worker_node",
            "refusal":         "refusal_node",
            "synthesizer":     "synthesizer_node",
        },
    )

    workflow.add_conditional_edges(
        "kb_worker_node",
        check_worker_security,
        {"security_alert": "security_alert_node", "synthesizer": "synthesizer_node"},
    )
    workflow.add_conditional_edges(
        "incident_worker_node",
        check_worker_security,
        {"security_alert": "security_alert_node", "synthesizer": "synthesizer_node"},
    )

    workflow.add_edge("synthesizer_node", END)
    workflow.add_edge("refusal_node", END)
    workflow.add_edge("security_alert_node", END)

    return workflow.compile()
