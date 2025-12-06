"""
Forensic Analysis Engine - Traces failures to root cause.

Given a failure, analyzes the signature chain to determine:
- Which agent introduced the bug (attribution)
- Which phase the failure was introduced (phase analysis)
- Confidence in the attribution (multi-phase probability)

Reference: docs/IMPLEMENTATION_PLAN_LEARNING.md Section 7

Layer: L2 (Forensic Analysis)
Status: Production
"""
import re
from typing import List, Optional, Tuple
import msgspec

from agents.schemas import (
    SignatureChain,
    CyclePhase,
    FailureCode,
    AgentSignature,
    SignatureAction,
    NodeOutcome,
)
from infrastructure.training_store import TrainingStore


class AttributionResult(msgspec.Struct, kw_only=True, frozen=True):
    """Result of forensic analysis."""
    failure_code: FailureCode
    attributed_agent_id: str
    attributed_model_id: str
    attributed_phase: CyclePhase
    confidence: float  # 0.0-1.0
    reasoning: str
    contributing_agents: list[str]  # Other agents involved


class ForensicAnalyzer:
    """
    Traces widget failures to their root cause.

    Uses signature chains and failure taxonomy (F1-F5) to
    attribute failures to specific agents and phases.
    """

    # Phase-specific confidence weights (from CLAUDE.md 6.3 and docs/IMPLEMENTATION_PLAN_LEARNING.md Section 5)
    PHASE_CONFIDENCE = {
        CyclePhase.INIT: 0.5,          # Initialization issues
        CyclePhase.DIALECTIC: 0.6,     # May not manifest until BUILD
        CyclePhase.RESEARCH: 0.65,     # Research gaps
        CyclePhase.CLARIFICATION: 0.55, # Ambiguity issues
        CyclePhase.PLAN: 0.7,          # Architecture issues
        CyclePhase.BUILD: 0.9,         # Direct causal link
        CyclePhase.TEST: 0.8,          # Test quality affects accuracy
        CyclePhase.PASSED: 0.3,        # Shouldn't have failures
        CyclePhase.FAILED: 0.3,        # Post-failure phase
    }

    # Error type to failure code mapping patterns
    ERROR_PATTERNS = {
        # F2: Implementation Failures (code generation issues)
        FailureCode.F2: [
            r"SyntaxError",
            r"IndentationError",
            r"NameError",
            r"TypeError.*missing.*argument",
            r"AttributeError",
            r"ImportError",
            r"ModuleNotFoundError",
            r"UnboundLocalError",
        ],
        # F3: Verification Failures (test quality issues)
        FailureCode.F3: [
            r"AssertionError",
            r"test.*fail",
            r"expected.*got",
        ],
        # F4: External Failures
        FailureCode.F4: [
            r"ConnectionError",
            r"TimeoutError",
            r"NetworkError",
            r"HTTPError",
            r"APIError",
        ],
    }

    def __init__(self, store: TrainingStore):
        """
        Initialize the forensic analyzer.

        Args:
            store: TrainingStore instance for querying historical data
        """
        self.store = store

    def analyze_failure(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
        failed_node_id: str | None = None,
        signature_chain: SignatureChain | None = None,
    ) -> AttributionResult:
        """
        Perform forensic analysis on a failure.

        Args:
            session_id: The session that failed
            error_type: Type of error (SyntaxError, TypeError, etc.)
            error_message: Full error message
            failed_node_id: Optional specific node that failed
            signature_chain: Optional signature chain for the failed node

        Returns:
            AttributionResult with root cause analysis
        """
        # If we have a failed node and can retrieve its signature chain, use it
        if failed_node_id and signature_chain is None:
            signature_chain = self.store.get_signature_chain(failed_node_id)

        # Determine the phase where failure likely occurred
        if signature_chain and signature_chain.signatures:
            # Get the most recent signature
            latest_sig = signature_chain.signatures[-1]
            failure_phase = latest_sig.phase

            # Get all signatures to identify contributing agents
            signatures = signature_chain.signatures
        else:
            # Fall back to analyzing error type alone
            failure_phase = self._infer_phase_from_error(error_type, error_message)
            signatures = []

        # Classify the failure type (F1-F5)
        failure_code = self._classify_failure(error_type, error_message, failure_phase)

        # Determine attribution and confidence
        if signatures:
            attributed_agent, attributed_model, confidence, contributing = \
                self._calculate_attribution_confidence(signatures, failure_code, failure_phase)
        else:
            # No signature data - low confidence fallback
            attributed_agent = "unknown"
            attributed_model = "unknown"
            confidence = 0.3
            contributing = []

        # Build reasoning
        reasoning = self._build_reasoning(
            failure_code, failure_phase, error_type, error_message,
            signatures, attributed_agent
        )

        return AttributionResult(
            failure_code=failure_code,
            attributed_agent_id=attributed_agent,
            attributed_model_id=attributed_model,
            attributed_phase=failure_phase,
            confidence=confidence,
            reasoning=reasoning,
            contributing_agents=contributing,
        )

    def _classify_failure(
        self,
        error_type: str,
        error_message: str,
        phase: CyclePhase
    ) -> FailureCode:
        """
        Classify failure as F1-F5 based on error and phase.

        Classification rules:
        - F1: Research Failure (Dialectic phase, topology design issues)
        - F2: Implementation Failure (Build phase, code generation issues)
        - F3: Verification Failure (Test phase, test quality issues)
        - F4: External Failure (Network, API, hardware)
        - F5: Indeterminate (cannot classify)

        Args:
            error_type: Type of error
            error_message: Error message content
            phase: Phase where error occurred

        Returns:
            FailureCode classification
        """
        combined_error = f"{error_type} {error_message}"

        # Check external failures first (highest priority)
        for pattern in self.ERROR_PATTERNS[FailureCode.F4]:
            if re.search(pattern, combined_error, re.IGNORECASE):
                return FailureCode.F4

        # Phase-based classification
        if phase in (CyclePhase.DIALECTIC, CyclePhase.RESEARCH, CyclePhase.CLARIFICATION):
            # Research/planning phase failures
            return FailureCode.F1

        if phase == CyclePhase.TEST:
            # Test phase - check if it's a test quality issue
            for pattern in self.ERROR_PATTERNS[FailureCode.F3]:
                if re.search(pattern, combined_error, re.IGNORECASE):
                    return FailureCode.F3
            # Test failed but not a test quality issue - could be F2
            for pattern in self.ERROR_PATTERNS[FailureCode.F2]:
                if re.search(pattern, combined_error, re.IGNORECASE):
                    return FailureCode.F2
            return FailureCode.F3  # Default for test phase

        if phase == CyclePhase.BUILD:
            # Build phase - check implementation failures
            for pattern in self.ERROR_PATTERNS[FailureCode.F2]:
                if re.search(pattern, combined_error, re.IGNORECASE):
                    return FailureCode.F2

        # Pattern-based classification for unclear phases
        for failure_code, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_error, re.IGNORECASE):
                    return failure_code

        # Could not classify
        return FailureCode.F5

    def _infer_phase_from_error(self, error_type: str, error_message: str) -> CyclePhase:
        """
        Infer the likely phase based on error type when no signature chain is available.

        Args:
            error_type: Type of error
            error_message: Error message content

        Returns:
            Inferred CyclePhase
        """
        combined = f"{error_type} {error_message}".lower()

        # Test-related errors
        if any(word in combined for word in ["test", "assert", "expected"]):
            return CyclePhase.TEST

        # Code/syntax errors
        if any(word in combined for word in ["syntax", "indent", "import", "name", "attribute"]):
            return CyclePhase.BUILD

        # Network/external errors
        if any(word in combined for word in ["connection", "network", "timeout", "http"]):
            return CyclePhase.BUILD  # Occurred during build/test execution

        # Default to build phase (most common)
        return CyclePhase.BUILD

    def _trace_signature_chain(self, node_id: str) -> list[AgentSignature]:
        """
        Get all signatures that touched a node.

        Args:
            node_id: Node identifier

        Returns:
            List of AgentSignature objects, or empty list if not found
        """
        chain = self.store.get_signature_chain(node_id)
        if chain:
            return chain.signatures
        return []

    def _calculate_attribution_confidence(
        self,
        signatures: list[AgentSignature],
        failure_code: FailureCode,
        phase: CyclePhase,
    ) -> tuple[str, str, float, list[str]]:
        """
        Determine most likely agent and confidence.

        Attribution logic:
        1. If failure is F2 (implementation), attribute to last BUILD agent
        2. If failure is F3 (verification), attribute to last TEST agent
        3. If failure is F1 (research), attribute to last DIALECTIC/RESEARCH agent
        4. If failure is F4 (external), lower confidence for all
        5. If failure is F5 (indeterminate), lowest confidence

        Args:
            signatures: List of agent signatures in chronological order
            failure_code: Classified failure type
            phase: Phase where failure occurred

        Returns:
            Tuple of (agent_id, model_id, confidence, contributing_agents)
        """
        if not signatures:
            return ("unknown", "unknown", 0.3, [])

        # Get base confidence from phase
        base_confidence = self.PHASE_CONFIDENCE.get(phase, 0.5)

        # Adjust confidence based on failure code
        if failure_code == FailureCode.F4:
            # External failures - not agent's fault
            base_confidence *= 0.5
        elif failure_code == FailureCode.F5:
            # Indeterminate - very low confidence
            base_confidence *= 0.4

        # Find the most relevant signature based on failure code
        attributed_sig = signatures[-1]  # Default to latest

        if failure_code == FailureCode.F1:
            # Research failure - find last dialectic/research/clarification agent
            for sig in reversed(signatures):
                if sig.phase in (CyclePhase.DIALECTIC, CyclePhase.RESEARCH, CyclePhase.CLARIFICATION):
                    attributed_sig = sig
                    break

        elif failure_code == FailureCode.F2:
            # Implementation failure - find last build agent
            for sig in reversed(signatures):
                if sig.phase == CyclePhase.BUILD:
                    attributed_sig = sig
                    break

        elif failure_code == FailureCode.F3:
            # Verification failure - find last test agent
            for sig in reversed(signatures):
                if sig.phase == CyclePhase.TEST:
                    attributed_sig = sig
                    break

        # Get contributing agents (all unique agents except the attributed one)
        contributing_agents = list(set(
            sig.agent_id for sig in signatures
            if sig.agent_id != attributed_sig.agent_id
        ))

        # Boost confidence if multiple agents from same phase (consensus)
        phase_agents = [sig for sig in signatures if sig.phase == attributed_sig.phase]
        if len(phase_agents) > 1:
            base_confidence *= 1.1
            base_confidence = min(base_confidence, 0.95)  # Cap at 0.95

        # Reduce confidence if action was REJECTED (someone already flagged issues)
        if attributed_sig.action == SignatureAction.REJECTED:
            base_confidence *= 0.7

        # Increase confidence if action was CREATED (original author)
        if attributed_sig.action == SignatureAction.CREATED:
            base_confidence *= 1.05

        # Ensure confidence is in valid range
        confidence = max(0.1, min(0.95, base_confidence))

        return (
            attributed_sig.agent_id,
            attributed_sig.model_id,
            confidence,
            contributing_agents,
        )

    def _build_reasoning(
        self,
        failure_code: FailureCode,
        failure_phase: CyclePhase,
        error_type: str,
        error_message: str,
        signatures: list[AgentSignature],
        attributed_agent: str,
    ) -> str:
        """
        Build human-readable reasoning for the attribution.

        Args:
            failure_code: Classified failure type
            failure_phase: Phase where failure occurred
            error_type: Type of error
            error_message: Error message (truncated if needed)
            signatures: Agent signatures involved
            attributed_agent: The agent being blamed

        Returns:
            Reasoning string
        """
        # Truncate error message if too long
        short_error = error_message[:200] + "..." if len(error_message) > 200 else error_message

        # Base reasoning on failure code
        code_explanations = {
            FailureCode.F1: "Research/topology design failure - ambiguities or unclear requirements",
            FailureCode.F2: "Implementation failure - code generation or syntax issues",
            FailureCode.F3: "Verification failure - inadequate test coverage or test quality",
            FailureCode.F4: "External failure - network, API, or infrastructure issues",
            FailureCode.F5: "Indeterminate failure - root cause unclear",
        }

        reasoning_parts = [
            f"Failure classified as {failure_code.value}: {code_explanations[failure_code]}.",
            f"Error occurred in {failure_phase.value} phase: {error_type}.",
        ]

        if signatures:
            reasoning_parts.append(
                f"Attributed to agent '{attributed_agent}' based on signature chain analysis "
                f"({len(signatures)} total agent interactions)."
            )
        else:
            reasoning_parts.append(
                f"No signature chain available - attribution based on error pattern analysis only."
            )

        reasoning_parts.append(f"Error details: {short_error}")

        return " ".join(reasoning_parts)

    def analyze_session_failures(
        self,
        session_id: str,
        failures: list[dict],
    ) -> list[AttributionResult]:
        """
        Analyze multiple failures from a session.

        Args:
            session_id: Session identifier
            failures: List of failure dicts with keys:
                     - node_id (optional)
                     - error_type
                     - error_message
                     - signature_chain (optional)

        Returns:
            List of AttributionResult objects, one per failure
        """
        results = []

        for failure in failures:
            result = self.analyze_failure(
                session_id=session_id,
                error_type=failure.get("error_type", "UnknownError"),
                error_message=failure.get("error_message", ""),
                failed_node_id=failure.get("node_id"),
                signature_chain=failure.get("signature_chain"),
            )
            results.append(result)

        return results

    def get_agent_failure_stats(
        self,
        agent_id: str | None = None,
        phase: CyclePhase | None = None,
    ) -> dict:
        """
        Get failure statistics for an agent or phase.

        Requires historical data in TrainingStore.

        Args:
            agent_id: Optional agent to filter by
            phase: Optional phase to filter by

        Returns:
            Dict with statistics:
            - total_attributions: Total failures attributed
            - failure_code_distribution: Dict of FailureCode -> count
            - average_confidence: Average attribution confidence
        """
        # This would query the training store for historical attribution data
        # For now, return a placeholder structure
        return {
            "total_attributions": 0,
            "failure_code_distribution": {
                fc.value: 0 for fc in FailureCode
            },
            "average_confidence": 0.0,
            "note": "Statistics require historical data - use record_attribution() to populate"
        }
