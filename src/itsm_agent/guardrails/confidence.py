"""
Confidence Threshold Guardrail.

Determines whether retrieved context is good enough to generate an answer,
or whether the agent should abstain, escalate, or ask for clarification.

Notes:
    Hallucination happens when an LLM generates an answer despite having
    no relevant context. The fix is NOT a better prompt — it's a retrieval
    quality gate BEFORE the LLM call. If retrieval confidence is below threshold,
    skip the LLM entirely. This saves cost AND prevents hallucination.

    In ITSM: a low-confidence retrieval means "we don't have KB coverage for this query."
    The right answer is "I don't have enough information — escalate to L2 support,"
    not a hallucinated answer that might cause the wrong fix to be applied.

    ServiceNow equivalent: Confidence scoring in Now Assist before generating a response.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConfidenceVerdict(str, Enum):
    """Decision from confidence evaluation."""
    ANSWER      = "answer"       # Context good enough — proceed to LLM
    ABSTAIN     = "abstain"      # No relevant context — return canned response
    ESCALATE    = "escalate"     # Context exists but low confidence — human handoff
    CLARIFY     = "clarify"      # Multiple possible intents — ask user to refine


@dataclass
class ConfidenceResult:
    """Output from confidence threshold check."""
    verdict: ConfidenceVerdict
    top_score: float                    # highest retrieval score in the result set
    mean_score: float                   # mean across top_k results
    score_gap: float                    # gap between rank-1 and rank-2 (measures clarity)
    n_results: int                      # how many chunks were retrieved
    reasoning: str                      # human-readable explanation
    escalation_message: Optional[str] = None  # shown to user on ESCALATE/ABSTAIN


class ConfidenceEvaluator:
    """
    Evaluates retrieval result quality to decide whether to answer, abstain,
    escalate to a human, or ask for clarification.

    Thresholds are tuned for ITSM KB + incident data. Adjust based on your
    eval set results — if your Hit@3 is 95%+ you can tighten the ANSWER threshold.

    Notes:
        Threshold tuning is an eval-driven decision, not a gut-feel one.
        We run the eval set, plot score distributions for correct vs incorrect
        retrievals, and set thresholds at the intersection point.
        In our ITSM eval: correct retrievals cluster above 0.72, incorrect below 0.55.
        We set ANSWER_THRESHOLD=0.65 to catch most correct cases with low false-positive rate.
    """

    def __init__(
        self,
        answer_threshold: float = 0.65,    # minimum top_score to attempt answer
        escalate_threshold: float = 0.45,  # below this → escalate to human
        min_results: int = 1,              # minimum chunks required to answer
        score_gap_threshold: float = 0.10, # minimum rank1-rank2 gap for clear intent
    ):
        """
        Args:
            answer_threshold:   Top retrieval score above which we attempt an LLM answer.
            escalate_threshold: Top score below which we escalate to human agent.
                                Between escalate and answer threshold → ESCALATE verdict.
            min_results:        If fewer than this many chunks retrieved → ABSTAIN.
            score_gap_threshold: If rank1 and rank2 scores are very close, query may
                                  be ambiguous → CLARIFY verdict.
        """
        self.answer_threshold = answer_threshold
        self.escalate_threshold = escalate_threshold
        self.min_results = min_results
        self.score_gap_threshold = score_gap_threshold

    def evaluate(
        self,
        retrieval_results: list,         # list of HybridResult or BM25Result
        score_attr: str = "rrf_score"    # attribute name containing the score
    ) -> ConfidenceResult:
        """
        Evaluate retrieval result quality.

        Args:
            retrieval_results: Ranked list from HybridRetriever.retrieve() or BM25.
            score_attr: Name of the score attribute on each result object.
                        Use 'rrf_score' for HybridResult, 'score' for BM25Result.

        Returns:
            ConfidenceResult with verdict and supporting evidence.

        Decision logic:
            1. No results or too few → ABSTAIN
            2. top_score < escalate_threshold → ESCALATE
            3. top_score < answer_threshold → ESCALATE (weak context)
            4. score_gap < gap_threshold and n_results >= 2 → CLARIFY (ambiguous query)
            5. Otherwise → ANSWER
        """
        if not retrieval_results or len(retrieval_results) < self.min_results:
            return ConfidenceResult(
                verdict=ConfidenceVerdict.ABSTAIN,
                top_score=0.0,
                mean_score=0.0,
                score_gap=0.0,
                n_results=len(retrieval_results) if retrieval_results else 0,
                reasoning="No relevant documents found in knowledge base.",
                escalation_message=(
                    "I don't have enough information to answer this question. "
                    "Please contact the IT support team directly or raise a ticket."
                )
            )

        # Extract scores
        scores = [getattr(r, score_attr, 0.0) for r in retrieval_results]
        top_score = scores[0]
        mean_score = sum(scores) / len(scores)
        score_gap = (scores[0] - scores[1]) if len(scores) >= 2 else scores[0]

        # Decision tree
        if top_score < self.escalate_threshold:
            return ConfidenceResult(
                verdict=ConfidenceVerdict.ESCALATE,
                top_score=top_score,
                mean_score=mean_score,
                score_gap=score_gap,
                n_results=len(retrieval_results),
                reasoning=f"Top retrieval score {top_score:.3f} below escalation threshold "
                           f"{self.escalate_threshold}. Insufficient context for safe answer.",
                escalation_message=(
                    "I'm not confident I have the right information for this query. "
                    "I'm escalating to an L2 support agent who can help you directly."
                )
            )

        if top_score < self.answer_threshold:
            return ConfidenceResult(
                verdict=ConfidenceVerdict.ESCALATE,
                top_score=top_score,
                mean_score=mean_score,
                score_gap=score_gap,
                n_results=len(retrieval_results),
                reasoning=f"Top score {top_score:.3f} below answer threshold "
                           f"{self.answer_threshold}. Context exists but confidence is low.",
                escalation_message=(
                    "I found some related information but I'm not confident enough to "
                    "give you a reliable answer. Routing to a human agent."
                )
            )

        if score_gap < self.score_gap_threshold and len(retrieval_results) >= 2:
            # Top two results are very similar in score — ambiguous query
            return ConfidenceResult(
                verdict=ConfidenceVerdict.CLARIFY,
                top_score=top_score,
                mean_score=mean_score,
                score_gap=score_gap,
                n_results=len(retrieval_results),
                reasoning=f"Score gap between rank-1 and rank-2 is {score_gap:.3f}, "
                           f"below threshold {self.score_gap_threshold}. Query may be ambiguous.",
                escalation_message=None
            )

        # All checks passed — safe to answer
        return ConfidenceResult(
            verdict=ConfidenceVerdict.ANSWER,
            top_score=top_score,
            mean_score=mean_score,
            score_gap=score_gap,
            n_results=len(retrieval_results),
            reasoning=f"Top score {top_score:.3f} above answer threshold. "
                       f"Score gap {score_gap:.3f} above clarity threshold. Proceeding.",
            escalation_message=None
        )


# ------------------------------------------------------------------
# Convenience function for LangGraph nodes
# ------------------------------------------------------------------

_default_evaluator = ConfidenceEvaluator()


def evaluate_confidence(retrieval_results: list, score_attr: str = "rrf_score") -> ConfidenceResult:
    """
    Convenience function for use in LangGraph guardrail nodes.

    Usage in LangGraph:
        def retrieval_gate_node(state: AgentState) -> AgentState:
            conf = evaluate_confidence(state["retrieval_results"])
            if conf.verdict == ConfidenceVerdict.ABSTAIN:
                return {**state, "response": conf.escalation_message, "done": True}
            if conf.verdict == ConfidenceVerdict.ESCALATE:
                return {**state, "escalate": True, "reason": conf.reasoning}
            return {**state, "confidence": conf}

    Notes:
        The confidence gate sits between the retriever and the LLM call.
        It's the single most important guardrail for preventing hallucination.
        We abort the LLM call entirely on ABSTAIN — no hallucination possible.
        On ESCALATE, we route to a human agent via ServiceNow assignment rules.
    """
    return _default_evaluator.evaluate(retrieval_results, score_attr=score_attr)
