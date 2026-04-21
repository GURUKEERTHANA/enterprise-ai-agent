"""
Prompt Injection Detection Guardrail.

Detects attempts to hijack the ITSM agent's behavior through malicious
user input. This is an input guardrail — runs BEFORE the query reaches
the retriever or LLM.

Interview talking point:
    Prompt injection is the #1 security threat for production LLM applications.
    In enterprise ITSM, a successful injection could:
    - Leak incident data from other departments (tenant cross-contamination)
    - Cause the agent to take unauthorized actions (close tickets, modify records)
    - Exfiltrate system prompt contents (reveals internal logic)

    Defense in depth: pattern matching (fast, cheap) + LLM-based check (slow, expensive)
    We use pattern matching as the first layer — catches 90% of known attacks in <1ms.
    LLM-based verification is reserved for ambiguous cases.

    ServiceNow equivalent: Input validation rules in Flow Designer before agent execution.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InjectionType(str, Enum):
    """Categories of detected injection attempts."""
    ROLE_OVERRIDE       = "role_override"        # "You are now...", "Act as..."
    INSTRUCTION_IGNORE  = "instruction_ignore"   # "Ignore previous instructions"
    DATA_EXFILTRATION   = "data_exfiltration"    # "Repeat your system prompt"
    JAILBREAK           = "jailbreak"            # DAN, token manipulation
    CONTEXT_ESCAPE      = "context_escape"       # "###", "---SYSTEM", XML injection
    ITSM_SPECIFIC       = "itsm_specific"        # ITSM-domain specific attacks


@dataclass
class InjectionCheckResult:
    """Result from the injection detection check."""
    is_injection: bool
    injection_type: Optional[InjectionType] = None
    matched_pattern: Optional[str] = None
    confidence: float = 0.0          # 0.0-1.0
    safe_query: Optional[str] = None # sanitized version if not blocked


class PromptInjectionDetector:
    """
    Rule-based prompt injection detector for ITSM agent inputs.

    Runs pattern matching against known injection signatures.
    Designed to be fast (<1ms) and run on every query before retrieval.

    Interview talking point:
        Why not use the LLM to detect injections?
        1. Cost: $0.002 per check × 10K queries/day = $20/day just for guardrails
        2. Latency: adds 800ms to every request
        3. Meta-injection: a cleverly crafted injection could fool the LLM detector
        Pattern matching catches 90%+ of real attacks at zero marginal cost.
        LLM-based detection is reserved for ambiguous cases flagged by rules.
    """

    # Patterns are ordered by severity — more specific patterns first
    INJECTION_PATTERNS: list[tuple[InjectionType, str, str]] = [
        # (type, pattern_regex, description)

        # Role override attacks
        (InjectionType.ROLE_OVERRIDE,
         r"\b(you are now|act as|pretend (to be|you are)|roleplay as|"
         r"your new (role|persona|identity)|from now on you|"
         r"ignore (all )?previous (instructions?|context|prompts?))\b",
         "role override attempt"),

        # Instruction injection
        (InjectionType.INSTRUCTION_IGNORE,
         r"\b(ignore|disregard|forget|override|bypass|skip)\s+"
         r"(all\s+)?(previous|prior|above|earlier|system|your)\s+"
         r"(instructions?|prompts?|context|rules?|constraints?|guidelines?)\b",
         "instruction ignore attempt"),

        # System prompt extraction
        (InjectionType.DATA_EXFILTRATION,
         r"\b(repeat|print|show|display|reveal|output|tell me|what (is|are))\s+"
         r"(your|the)\s+(system\s+)?(prompt|instructions?|context|"
         r"initial\s+message|configuration)\b",
         "system prompt extraction attempt"),

        # Jailbreak patterns
        (InjectionType.JAILBREAK,
         r"\b(DAN|do anything now|developer mode|jailbreak|"
         r"unrestricted mode|god mode|no restrictions?)\b",
         "jailbreak attempt"),

        # Context escape via special tokens
        (InjectionType.CONTEXT_ESCAPE,
         r"(###\s*(system|instruction|user|assistant)|"
         r"<\|?(system|endofprompt|im_start|im_end)\|?>|"
         r"---+(system|instruction|override)---+|\[INST\]|\[\/INST\])",
         "context escape via special tokens"),

        # ITSM-specific attacks
        (InjectionType.ITSM_SPECIFIC,
         r"\b(close all (tickets?|incidents?)|assign to (yourself|me all)|"
         r"delete (all\s+)?(records?|data|incidents?|tickets?)|"
         r"show (all\s+)?(users?|tickets?|incidents?) (from|in|across) "
         r"(all\s+)?(departments?|tenants?|teams?))\b",
         "ITSM-specific unauthorized action attempt"),
    ]

    # Compile patterns once at class load time for performance
    _compiled_patterns: list[tuple[InjectionType, re.Pattern, str]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self, block_on_detection: bool = True):
        """
        Args:
            block_on_detection: If True, detected injections are blocked.
                                 If False, they are flagged but allowed through
                                 (useful for logging/monitoring without blocking).
        """
        self.block_on_detection = block_on_detection
        self._compiled = [
            (itype, re.compile(pattern, re.IGNORECASE | re.MULTILINE), desc)
            for itype, pattern, desc in self.INJECTION_PATTERNS
        ]

    def check(self, query: str) -> InjectionCheckResult:
        """
        Check a query for prompt injection attempts.

        Args:
            query: Raw user input string.

        Returns:
            InjectionCheckResult with detection details.

        Complexity: O(P * L) where P = number of patterns, L = query length.
                    Typically <0.5ms for queries under 1000 characters.
        """
        if not query or not query.strip():
            return InjectionCheckResult(
                is_injection=False,
                safe_query=query,
                confidence=0.0
            )

        # Check each pattern
        for injection_type, pattern, description in self._compiled:
            match = pattern.search(query)
            if match:
                return InjectionCheckResult(
                    is_injection=True,
                    injection_type=injection_type,
                    matched_pattern=match.group(0),
                    confidence=0.95,
                    safe_query=None  # blocked — no safe version provided
                )

        # No injection detected — pass through with sanitized query
        safe_query = self._sanitize(query)
        return InjectionCheckResult(
            is_injection=False,
            safe_query=safe_query,
            confidence=0.0
        )

    @staticmethod
    def _sanitize(query: str) -> str:
        """
        Light sanitization for non-injection queries.

        Removes control characters and excessively long whitespace sequences
        without altering the semantic content of legitimate queries.
        """
        # Remove control characters (except newlines and tabs)
        query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
        # Collapse multiple whitespace
        query = re.sub(r"[ \t]+", " ", query)
        # Trim
        return query.strip()


# ------------------------------------------------------------------
# Standalone check function for use in LangGraph nodes
# ------------------------------------------------------------------

_default_detector = PromptInjectionDetector(block_on_detection=True)


def check_prompt_injection(query: str) -> InjectionCheckResult:
    """
    Convenience function for use in LangGraph guardrail nodes.

    Usage in LangGraph:
        def validate_input_node(state: AgentState) -> AgentState:
            result = check_prompt_injection(state["user_query"])
            if result.is_injection:
                return {**state, "blocked": True, "block_reason": result.injection_type}
            return {**state, "user_query": result.safe_query}

    Interview talking point:
        This runs as the first node in every agent graph execution.
        If is_injection=True, the graph routes to a REFUSAL node, not the retriever.
        The user sees: "I can't help with that request" — no information about why,
        to avoid helping attackers tune their injection.
    """
    return _default_detector.check(query)
