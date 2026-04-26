# src/itsm_agent/agent/nodes.py
from __future__ import annotations

from .state import AgentState
from .router_schema import RouteAction
from .registry import DEPARTMENT_REGISTRY
from .llms import router_llm, synthesizer_llm
from src.itsm_agent.guardrails.prompt_injection import check_prompt_injection
from src.itsm_agent.guardrails.confidence import ConfidenceEvaluator, ConfidenceVerdict

# RRF scores range 0.013–0.033 (not 0–1 like cosine similarity).
# answer_threshold=0.020 → result in top-3 of at least one retriever.
# escalate_threshold=0.014 → only barely in top-20 of one retriever.
# score_gap_threshold=0.001 → true score tie (same rank in both retrievers).
_confidence = ConfidenceEvaluator(
    answer_threshold=0.020,
    escalate_threshold=0.014,
    score_gap_threshold=0.001,
)
from src.itsm_agent.retrieval.chroma_retriever import kb_hybrid, incident_hybrid
from src.itsm_agent.retrieval.reranker import reranker_model
from src.itsm_agent.utils.latency import new_profiler


def validate_input_node(state: AgentState) -> dict:
    profiler = new_profiler()
    user_query = state["messages"][-1].content if state.get("messages") else ""
    with profiler.measure("injection_check"):
        result = check_prompt_injection(user_query)
    if result.is_injection:
        return {
            "blocked": True,
            "final_answer": "I can't help with that request.",
            "block_reason": result.injection_type,
            "profiler": profiler,
        }
    return {"blocked": False, "profiler": profiler}


def router_node(state: AgentState) -> dict:
    allowed_depts = ", ".join(DEPARTMENT_REGISTRY.keys())
    registry_parts = [
        f"DEPT: {dept}\nDESCRIPTION: {info['description']}\nKEYWORDS: {', '.join(info['keywords'])}"
        for dept, info in DEPARTMENT_REGISTRY.items()
    ]
    registry_text = "\n\n".join(registry_parts)

    system_prompt = f"""
You are a ServiceNow Router. Your task is to map queries to the correct department.

1. ALLOWED DEPARTMENTS (You MUST pick one of these keys):
{allowed_depts}

2. MAPPING TAXONOMY (Use keywords to find the match):
{registry_text}

3. CLASSIFICATION STEPS:
   - Identify the tool in the query (e.g., 'Okta').
   - Find which DEPT has that tool in its keywords.
   - Return the EXACT key from the ALLOWED DEPARTMENTS list.
   - If 'Okta' or 'VPN' is found, you are FORBIDDEN from returning None; you MUST return 'DT-GPS'.
"""
    structured_router = router_llm.with_structured_output(RouteAction, method="function_calling")
    user_message = state["messages"][-1].content

    profiler = state.get("profiler")
    if profiler:
        with profiler.measure("router_llm"):
            decision: RouteAction = structured_router.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ])
    else:
        decision: RouteAction = structured_router.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ])

    print(f"--- ROUTING DECISION: {decision.source_type} | {decision.department_id} ---")
    return {"route": decision}


def kb_worker_node(state: AgentState) -> dict:
    route: RouteAction = state.get("route")
    if not route:
        return {"kb_results": ["Error: No routing instructions found."]}

    query = route.refined_query
    requested_dept = route.department_id
    authorized_dept = state.get("verified_tenant_id", "UNKNOWN_TENANT")
    profiler = state.get("profiler")

    if requested_dept and requested_dept != authorized_dept:
        print(f"--- SECURITY BLOCK: {authorized_dept} attempted to query {requested_dept} ---")
        return {
            "security_violation": True,
            "kb_results": [f"Violation: {requested_dept} vs {authorized_dept}"],
        }

    print(f"--- KB WORKER: Authorized search in '{authorized_dept}' for '{query}' ---")
    try:
        if profiler:
            with profiler.measure("hybrid_retrieval"):
                results = kb_hybrid.retrieve(query, top_k=10, department_id=authorized_dept)
        else:
            results = kb_hybrid.retrieve(query, top_k=10, department_id=authorized_dept)

        if not results:
            return {"kb_results": [], "security_violation": False}

        conf = _confidence.evaluate(results, score_attr="rrf_score")
        print(f"--- CONFIDENCE (KB): {conf.verdict.value} | top={conf.top_score:.3f} | gap={conf.score_gap:.3f} ---")
        if conf.verdict in (ConfidenceVerdict.ABSTAIN, ConfidenceVerdict.ESCALATE):
            return {
                "security_violation": False,
                "kb_results": [conf.escalation_message],
                "escalate": True,
            }

        candidates = [r.text for r in results]
        pairs = [[query, doc] for doc in candidates]
        if profiler:
            with profiler.measure("reranking"):
                scores = reranker_model.predict(pairs)
        else:
            scores = reranker_model.predict(pairs)

        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:3]]
        print(f"--- RERANKER: Selected top {len(top_docs)} from {len(candidates)} KB candidates (hybrid) ---")
    except Exception as e:
        top_docs = [f"Error retrieving KB records: {str(e)}"]

    return {"security_violation": False, "kb_results": top_docs}


def incident_worker_node(state: AgentState) -> dict:
    route: RouteAction = state.get("route")
    if not route:
        return {"incident_results": ["Error: No routing instructions found."]}

    query = route.refined_query
    authorized_dept = state.get("verified_tenant_id", "UNKNOWN_TENANT")
    requested_dept = route.department_id
    profiler = state.get("profiler")

    if requested_dept and requested_dept != authorized_dept:
        print(f"--- SECURITY BLOCK (INCIDENT): Access denied for {requested_dept} ---")
        return {
            "security_violation": True,
            "incident_results": [f"Violation: {requested_dept} vs {authorized_dept}"],
        }

    print(f"--- INCIDENT WORKER: Multi-stage hybrid search in '{authorized_dept}' ---")
    try:
        if profiler:
            with profiler.measure("hybrid_retrieval"):
                results = incident_hybrid.retrieve(query, top_k=25, department_id=authorized_dept)
        else:
            results = incident_hybrid.retrieve(query, top_k=25, department_id=authorized_dept)

        if not results:
            return {"incident_results": [], "security_violation": False}

        conf = _confidence.evaluate(results, score_attr="rrf_score")
        print(f"--- CONFIDENCE (INCIDENT): {conf.verdict.value} | top={conf.top_score:.3f} | gap={conf.score_gap:.3f} ---")
        if conf.verdict in (ConfidenceVerdict.ABSTAIN, ConfidenceVerdict.ESCALATE):
            return {
                "security_violation": False,
                "incident_results": [conf.escalation_message],
                "escalate": True,
            }

        candidates = [r.text for r in results]
        pairs = [[query, doc] for doc in candidates]
        if profiler:
            with profiler.measure("reranking"):
                scores = reranker_model.predict(pairs)
        else:
            scores = reranker_model.predict(pairs)

        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:3]]
        print(f"--- RERANKER: Selected top {len(top_docs)} from {len(candidates)} incident candidates (hybrid) ---")
    except Exception as e:
        top_docs = [f"Error in incident retrieval: {str(e)}"]

    return {"security_violation": False, "incident_results": top_docs}


def synthesizer_node(state: AgentState) -> dict:
    profiler = state.get("profiler")

    if state.get("escalate"):
        kb_data = state.get("kb_results", [])
        inc_data = state.get("incident_results", [])
        msg = (kb_data or inc_data or [
            "I'm unable to find a reliable answer. Please contact support directly or raise a ticket."
        ])[0]
        if profiler:
            print(profiler.report())
        return {"final_answer": msg}

    kb_data = state.get("kb_results", [])
    inc_data = state.get("incident_results", [])

    context_parts = (
        [f"[SOURCE: Knowledge Base]\n{doc}" for doc in kb_data]
        + [f"[SOURCE: Past Incident Resolution]\n{doc}" for doc in inc_data]
    )
    all_context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a ServiceNow Support Expert. Provide a technical solution using the context below.

CONTEXT:
{all_context if all_context.strip() else "NO DATA FOUND"}

USER QUERY: {state['messages'][-1].content}

INSTRUCTIONS:
1. If the CONTEXT contains resolution steps, present them as the SOLUTION.
2. DO NOT say "I couldn't find a solution" if troubleshooting steps exist in the context.
3. Use an authoritative, helpful tone.
4. If the context is 'NO DATA FOUND', only then state that you couldn't find a record.
"""
    if profiler:
        with profiler.measure("llm_call"):
            response = synthesizer_llm.invoke(prompt)
        print(profiler.report())
    else:
        response = synthesizer_llm.invoke(prompt)

    return {"final_answer": response.content}


def refusal_node(state: AgentState) -> dict:
    route: RouteAction | None = state.get("route")
    detail = route.refined_query if route else "I cannot assist with this."
    return {"final_answer": f"I'm sorry, I'm authorized only for ServiceNow ITSM and HR tasks. {detail}"}


def security_alert_node(state: AgentState) -> dict:
    violation_details = (state.get("kb_results") or state.get("incident_results") or [""])[0]
    prompt = f"""
You are a strict Enterprise Security Assistant.
The user attempted to access restricted data.
Write a brief, polite, but firm 'Access Denied' message.
Details: {violation_details}
"""
    response = router_llm.invoke(prompt)
    return {"final_answer": response.content}


# ---------------------------------------------------------------------------
# Conditional edge helpers
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    if state.get("blocked"):
        return "refusal"
    route: RouteAction | None = state.get("route")
    if not route or route.source_type == "out_of_scope":
        return "refusal"
    if route.source_type == "kb_article":
        return "kb_worker"
    if route.source_type == "incident":
        return "incident_worker"
    return "synthesizer"


def check_worker_security(state: AgentState) -> str:
    return "security_alert" if state.get("security_violation") else "synthesizer"
