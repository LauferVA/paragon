#!/usr/bin/env python3
"""
Simple Forensic Analysis Demo - Quick examples of attribution analysis.

This script can be run directly or imported as a module.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
from datetime import datetime

from infrastructure.attribution import ForensicAnalyzer
from infrastructure.training_store import TrainingStore
from agents.schemas import (
    AgentSignature,
    SignatureChain,
    CyclePhase,
    SignatureAction,
)


def demo_simple_syntax_error():
    """Example 1: Simple syntax error without signature chain."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Syntax Error")
    print("="*80)

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        store = TrainingStore(db_path=Path(f.name))
        analyzer = ForensicAnalyzer(store=store)

        result = analyzer.analyze_failure(
            session_id="demo_1",
            error_type="SyntaxError",
            error_message="missing closing parenthesis at line 42",
        )

        print(f"Failure Code: {result.failure_code.value}")
        print(f"Phase: {result.attributed_phase.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"\nReasoning: {result.reasoning[:200]}...")

        Path(f.name).unlink()


def demo_with_signature_chain():
    """Example 2: Build failure with complete signature chain."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Build Failure with Signature Chain")
    print("="*80)

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        store = TrainingStore(db_path=Path(f.name))
        analyzer = ForensicAnalyzer(store=store)

        # Create signature chain
        chain = SignatureChain(
            node_id="code_42",
            state_id="state_abc",
            signatures=[
                AgentSignature(
                    agent_id="architect_v1",
                    model_id="claude-sonnet-4-5",
                    phase=CyclePhase.PLAN,
                    action=SignatureAction.CREATED,
                    temperature=0.7,
                    context_constraints={},
                    timestamp=datetime.now().isoformat(),
                ),
                AgentSignature(
                    agent_id="builder_v2",
                    model_id="claude-sonnet-4-5",
                    phase=CyclePhase.BUILD,
                    action=SignatureAction.CREATED,
                    temperature=0.3,
                    context_constraints={},
                    timestamp=datetime.now().isoformat(),
                ),
            ],
        )

        result = analyzer.analyze_failure(
            session_id="demo_2",
            error_type="NameError",
            error_message="name 'undefined_var' is not defined",
            signature_chain=chain,
        )

        print(f"Failure Code: {result.failure_code.value}")
        print(f"Attributed Agent: {result.attributed_agent_id}")
        print(f"Attributed Model: {result.attributed_model_id}")
        print(f"Phase: {result.attributed_phase.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Contributing Agents: {result.contributing_agents}")
        print(f"\nReasoning: {result.reasoning[:200]}...")

        Path(f.name).unlink()


def demo_multi_failure_analysis():
    """Example 3: Multiple failures in one session."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Multiple Failures in One Session")
    print("="*80)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        store = TrainingStore(db_path=Path(f.name))
        analyzer = ForensicAnalyzer(store=store)

        failures = [
            {
                "error_type": "SyntaxError",
                "error_message": "Missing colon in function definition",
            },
            {
                "error_type": "AssertionError",
                "error_message": "Test failed: expected 42, got 0",
            },
            {
                "error_type": "ConnectionError",
                "error_message": "API timeout after 30 seconds",
            },
        ]

        results = analyzer.analyze_session_failures(
            session_id="demo_3",
            failures=failures,
        )

        print(f"\nAnalyzed {len(results)} failures:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.failure_code.value} ({result.attributed_phase.value} phase)")
            print(f"   Confidence: {result.confidence:.2f}")

        Path(f.name).unlink()


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("FORENSIC ANALYSIS ENGINE - SIMPLE DEMO")
    print("="*80)
    print("\nDemonstrating failure classification and attribution...")

    demo_simple_syntax_error()
    demo_with_signature_chain()
    demo_multi_failure_analysis()

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nThe ForensicAnalyzer can:")
    print("  ✓ Classify failures (F1-F5)")
    print("  ✓ Attribute to specific agents")
    print("  ✓ Calculate confidence scores")
    print("  ✓ Track contributing agents")
    print("  ✓ Handle multiple failures")


if __name__ == "__main__":
    main()
