#!/usr/bin/env python3
"""
Interactive Dialectic Runner - Real-time observation and control

This script runs the TDD orchestrator with an ambiguous prompt and lets you:
1. See the dialectic analysis in real-time
2. Answer clarification questions interactively
3. Observe the research artifact being created
4. Watch the full pipeline execute

Usage:
    python run_dialectic.py
"""

import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Rich console for better output (fallback to print if not available)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

def print_header(text: str):
    if RICH_AVAILABLE:
        console.print(Panel(text, style="bold blue"))
    else:
        print(f"\n{'='*60}\n{text}\n{'='*60}\n")

def print_phase(phase: str, detail: str = ""):
    if RICH_AVAILABLE:
        console.print(f"[bold green]▶ PHASE:[/bold green] [yellow]{phase}[/yellow] {detail}")
    else:
        print(f"▶ PHASE: {phase} {detail}")

def print_finding(title: str, content: str):
    if RICH_AVAILABLE:
        console.print(Panel(content, title=title, border_style="cyan"))
    else:
        print(f"\n--- {title} ---\n{content}\n")

def print_question(question: str, context: str = "", suggested_answer: str = None):
    if RICH_AVAILABLE:
        content = f"[bold]{question}[/bold]\n\n{context}"
        if suggested_answer:
            content += f"\n\n[dim]Suggested answer:[/dim] [italic green]{suggested_answer}[/italic green]"
        console.print(Panel(content, title="❓ Clarification Needed", border_style="yellow"))
    else:
        print(f"\n❓ CLARIFICATION NEEDED:\n{question}\n{context}")
        if suggested_answer:
            print(f"\nSuggested answer: {suggested_answer}\n")

def get_user_input(prompt: str) -> str:
    if RICH_AVAILABLE:
        return Prompt.ask(f"[bold cyan]{prompt}[/bold cyan]")
    else:
        return input(f"{prompt}: ")

def get_answer_with_suggestion(question_num: int, total: int, suggested_answer: str = None) -> str:
    """
    Get user's answer, offering them the choice to accept the suggested answer or provide their own.
    """
    if suggested_answer:
        if RICH_AVAILABLE:
            console.print(f"\n[bold cyan]Options:[/bold cyan]")
            console.print(f"  [1] Accept suggested answer")
            console.print(f"  [2] Provide your own answer")
            choice = Prompt.ask("Your choice", choices=["1", "2"], default="1")
        else:
            print(f"\nOptions:")
            print(f"  [1] Accept suggested answer")
            print(f"  [2] Provide your own answer")
            choice = input("Your choice (1/2) [1]: ").strip() or "1"

        if choice == "1":
            if RICH_AVAILABLE:
                console.print(f"[green]✓ Using suggested answer[/green]")
            else:
                print("✓ Using suggested answer")
            return suggested_answer
        else:
            return get_user_input("Your answer")
    else:
        return get_user_input("Your answer")

def run_interactive_dialectic():
    """Run the orchestrator with interactive dialectic."""

    print_header("PARAGON INTERACTIVE DIALECTIC RUNNER")
    print("This will run an ambiguous prompt through the full pipeline.\n")
    print("You will be asked to clarify ambiguities in real-time.\n")

    # Import orchestrator components
    from agents.orchestrator import (
        TDDOrchestrator, CyclePhase,
        dialectic_node, clarification_node, research_node,
        plan_node, init_node,
    )
    from agents.tools import set_db, get_graph_stats
    from core.graph_db import ParagonDB

    # Create fresh database
    db = ParagonDB()
    set_db(db)

    # Define an ambiguous spec
    ambiguous_specs = [
        """Build a fast sorting function.
It should handle large datasets efficiently.
Make it user-friendly and robust.""",

        """Create a caching system for our API.
It needs to be performant and scalable.
Handle edge cases appropriately.""",

        """Implement a notification service.
It should be reliable and fast.
Support multiple channels.""",
    ]

    print("Choose an ambiguous spec to test:\n")
    for i, spec in enumerate(ambiguous_specs, 1):
        print(f"  [{i}] {spec[:50]}...")
    print(f"  [4] Enter custom spec")

    choice = get_user_input("\nSelect (1-4)")

    if choice == "4":
        spec = get_user_input("Enter your ambiguous spec")
    else:
        try:
            spec = ambiguous_specs[int(choice) - 1]
        except (ValueError, IndexError):
            spec = ambiguous_specs[0]

    print_header("SPEC TO ANALYZE")
    print(spec)
    print()

    # Initialize state
    state = {
        "session_id": f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "task_id": "dialectic_test",
        "spec": spec,
        "phase": CyclePhase.INIT.value,
        "messages": [],
        "requirements": [],
        "errors": [],
        "iteration": 0,
        "max_iterations": 3,
        "artifacts": [],
        "test_results": [],
        "ambiguities": [],
        "clarification_questions": [],
        "research_findings": [],
        "dialectic_passed": False,
        "research_complete": False,
    }

    # PHASE 1: INIT
    print_phase("INIT", "Initializing TDD cycle...")
    result = init_node(state)
    state.update(result)
    print(f"  → Next phase: {state['phase']}")

    # PHASE 2: DIALECTIC
    print_phase("DIALECTIC", "Analyzing for ambiguity...")
    print("  (This calls the LLM to detect subjective terms, undefined references, etc.)\n")

    result = dialectic_node(state)
    state.update(result)

    # Show what was found
    ambiguities = state.get("ambiguities", [])
    questions = state.get("clarification_questions", [])

    if ambiguities:
        print_finding(f"Found {len(ambiguities)} Ambiguities",
            "\n".join([f"• [{a.get('category', 'UNKNOWN')}] \"{a.get('text', 'N/A')}\" - {a.get('impact', 'N/A')}"
                      for a in ambiguities]))
    else:
        print("  No ambiguities detected (spec is clear)")

    # PHASE 3: CLARIFICATION (if needed)
    if state["phase"] == CyclePhase.CLARIFICATION.value and questions:
        print_phase("CLARIFICATION", f"Need {len(questions)} answers from you")

        user_responses = []
        for i, q in enumerate(questions, 1):
            question_text = q.get("question", "Please clarify")
            category = q.get("category", "")
            ambiguous_text = q.get("text", "")
            suggested_answer = q.get("suggested_answer")

            print_question(
                f"Question {i}/{len(questions)}: {question_text}",
                f"Category: {category}\nAmbiguous phrase: \"{ambiguous_text}\"",
                suggested_answer=suggested_answer
            )

            response = get_answer_with_suggestion(i, len(questions), suggested_answer)
            user_responses.append({
                "question": question_text,
                "response": response,
            })

        # Store responses and process clarification
        state["human_response"] = user_responses
        result = clarification_node(state)
        state.update(result)

        print_finding("Augmented Spec", state.get("spec", "")[:500] + "...")
    else:
        print("  → Spec is clear, proceeding to research")
        state["dialectic_passed"] = True

    # PHASE 4: RESEARCH
    print_phase("RESEARCH", "Creating Research Artifact (sufficient statistic)...")
    print("  (This creates input/output contracts, examples, complexity bounds)\n")

    result = research_node(state)
    state.update(result)

    # Show research findings
    findings = state.get("research_findings", [])
    if findings:
        for f in findings:
            print_finding("Research Artifact", f"""
Task Category: {f.get('task_category', 'N/A')}
Input Contract: {f.get('input_contract', 'N/A')}
Output Contract: {f.get('output_contract', 'N/A')}
Happy Path Examples: {f.get('happy_path_count', 0)}
Edge Cases: {f.get('edge_case_count', 0)}
Error Cases: {f.get('error_case_count', 0)}
Complexity: {f.get('complexity_bounds', 'N/A')}
Security: {f.get('security_posture', 'N/A')}
""")

    # Show augmented spec
    print_finding("Final Augmented Spec", state.get("spec", "")[:1000])

    # Ask if user wants to continue to PLAN/BUILD/TEST
    print()
    if RICH_AVAILABLE:
        continue_build = Confirm.ask("Continue to PLAN → BUILD → TEST phases?")
    else:
        continue_build = input("Continue to PLAN → BUILD → TEST phases? (y/n): ").lower() == 'y'

    if continue_build:
        print_phase("PLAN", "Creating implementation plan...")
        result = plan_node(state)
        state.update(result)

        # Show artifacts created
        stats = get_graph_stats()
        print_finding("Graph State After Planning", f"""
Nodes: {stats.get('node_count', 0)}
Edges: {stats.get('edge_count', 0)}
Node Types: {stats.get('by_type', {})}
""")

        print("\n✅ Dialectic → Clarification → Research → Plan flow complete!")
        print(f"   Session ID: {state['session_id']}")
        print(f"   Final Phase: {state['phase']}")
    else:
        print("\n✅ Dialectic → Clarification → Research flow complete!")

    # Summary
    print_header("SESSION SUMMARY")
    summary = f"""
Session: {state['session_id']}
Ambiguities Found: {len(ambiguities)}
Questions Asked: {len(questions)}
Research Complete: {state.get('research_complete', False)}
Final Phase: {state['phase']}
"""
    print(summary)

    return state


if __name__ == "__main__":
    try:
        run_interactive_dialectic()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
