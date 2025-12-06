"""
Learning System Demo - Demonstrates the full learning pipeline.

This script showcases:
1. Study Mode - Collecting data without biasing
2. Learning Stats - Tracking progress
3. Transition to Production - When ready
4. Production Mode - Using learned patterns
5. Adaptive Questioning - Optimizing questions based on patterns

Run this to see the learning system in action.
"""
import tempfile
from pathlib import Path
from datetime import datetime

from infrastructure.learning import LearningManager, LearningMode
from agents.adaptive_questioner import AdaptiveQuestioner, UserPriorities
from agents.schemas import (
    AgentSignature,
    SignatureAction,
    CyclePhase,
    FailureCode,
    NodeOutcome,
    AmbiguityMarker,
)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def simulate_sessions(manager: LearningManager, num_sessions: int = 10, start_index: int = 0):
    """
    Simulate a number of sessions with varying outcomes.

    This represents the system being used and learning from outcomes.
    """
    print(f"Simulating {num_sessions} sessions...")

    models = ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929", "claude-haiku-3-5-20241022"]
    phases = [CyclePhase.DIALECTIC, CyclePhase.PLAN, CyclePhase.BUILD, CyclePhase.TEST]

    for i in range(num_sessions):
        session_id = f"session_{start_index + i}"

        # Randomly select model and phase
        model_id = models[i % len(models)]
        phase = phases[i % len(phases)]

        # Create a signature
        sig = AgentSignature(
            agent_id=f"agent_{phase.value}",
            model_id=model_id,
            phase=phase,
            action=SignatureAction.CREATED,
            temperature=0.7,
            context_constraints={"max_tokens": 4096},
            timestamp=datetime.utcnow().isoformat(),
        )

        # Record attribution
        manager.store.record_attribution(
            session_id=session_id,
            signature=sig,
            node_id=f"node_{i}",
            state_id=f"state_{i}",
        )

        # Simulate outcome (80% success rate, with some model variation)
        # Opus is better (90% success), Haiku is worse (70% success)
        if model_id == "claude-opus-4-5-20251101":
            success = (i % 10) < 9  # 90% success
        elif model_id == "claude-haiku-3-5-20241022":
            success = (i % 10) < 7  # 70% success
        else:
            success = (i % 10) < 8  # 80% success

        # Record outcome
        if success:
            manager.record_outcome(
                session_id=session_id,
                success=True,
                stats={"total_nodes": 10 + i, "total_iterations": 1, "total_tokens": 5000 + i * 100},
            )
        else:
            manager.record_outcome(
                session_id=session_id,
                success=False,
                failure_code=FailureCode.F2,
                failure_phase=phase,
                stats={"total_nodes": 10 + i, "total_iterations": 2, "total_tokens": 8000 + i * 100},
            )

    print(f"✓ Completed {num_sessions} sessions\n")


def demonstrate_study_mode():
    """Demonstrate Study Mode - data collection without biasing."""
    print_section("STUDY MODE - Collecting Clean Data")

    # Create temporary database for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "learning_demo.db"

    # Initialize in Study Mode
    manager = LearningManager(db_path=db_path, mode=LearningMode.STUDY)

    print("Initialized LearningManager in STUDY mode")
    print(f"Database: {db_path}")
    print(f"Mode: {manager.mode.value}")

    # Check for recommendations (should be None in STUDY mode)
    print("\n→ Requesting model recommendation in STUDY mode...")
    recommendation = manager.get_model_recommendation(phase=CyclePhase.BUILD)

    if recommendation is None:
        print("✓ No recommendation provided (STUDY mode doesn't bias)")
    else:
        print("✗ Unexpected recommendation in STUDY mode!")

    # Simulate some sessions
    print("\n→ Simulating 25 sessions to collect data...")
    simulate_sessions(manager, num_sessions=25)

    # Get learning stats
    stats = manager.get_learning_stats()
    print("Learning Statistics:")
    print(f"  Total Sessions: {stats.total_sessions}")
    print(f"  Successful: {stats.successful_sessions}")
    print(f"  Failed: {stats.failed_sessions}")
    print(f"  Success Rate: {stats.successful_sessions / stats.total_sessions * 100:.1f}%")
    print(f"  Ready for Production: {stats.ready_for_production}")
    print(f"\n  Recommendation: {stats.recommendation}")

    return manager


def demonstrate_transition(manager: LearningManager):
    """Demonstrate transition from STUDY to PRODUCTION mode."""
    print_section("TRANSITION ANALYSIS")

    # Check if ready to transition
    report = manager.should_transition_to_production()

    print(f"Ready for Production: {report.ready}")
    print(f"Session Count: {report.session_count} / {manager.MIN_SESSIONS_FOR_PRODUCTION}")
    print(f"Success Rate: {report.success_rate * 100:.1f}%")
    print(f"Divergence Rate: {report.divergence_rate * 100:.1f}%")
    print(f"\nRecommendation: {report.recommendation}")

    if not report.ready:
        print("\n→ Need more data. Simulating additional sessions...")
        remaining = manager.MIN_SESSIONS_FOR_PRODUCTION - report.session_count
        simulate_sessions(manager, num_sessions=remaining, start_index=report.session_count)

        # Check again
        report = manager.should_transition_to_production()
        print(f"\n✓ After additional sessions: {report.session_count} sessions collected")

    # Attempt transition
    print("\n→ Attempting transition to PRODUCTION mode...")
    success = manager.transition_to_production()

    if success:
        print("✓ Successfully transitioned to PRODUCTION mode")
        print(f"  Current mode: {manager.mode.value}")
    else:
        print("✗ Transition failed")

    return manager


def demonstrate_production_mode(manager: LearningManager):
    """Demonstrate Production Mode - using learned patterns."""
    print_section("PRODUCTION MODE - Using Learned Patterns")

    print(f"Current mode: {manager.mode.value}")

    # Get model performance summary
    print("\n→ Model Performance Summary:")
    summary = manager.get_model_performance_summary()

    for model_id, phases in summary.items():
        print(f"\n  {model_id}:")
        for phase, stats in phases.items():
            if stats["sample_count"] > 0:
                print(f"    {phase}: {stats['success_rate'] * 100:.1f}% success ({stats['sample_count']} samples)")

    # Request recommendations for different phases
    print("\n→ Requesting model recommendations in PRODUCTION mode:")

    for phase in [CyclePhase.BUILD, CyclePhase.TEST, CyclePhase.PLAN]:
        recommendation = manager.get_model_recommendation(phase=phase)

        if recommendation:
            print(f"\n  {phase.value.upper()}:")
            print(f"    Recommended Model: {recommendation.model_id}")
            print(f"    Success Rate: {recommendation.success_rate * 100:.1f}%")
            print(f"    Confidence: {recommendation.confidence * 100:.1f}%")
            print(f"    Based on {recommendation.based_on_samples} samples")
            print(f"    Reasoning: {recommendation.reasoning}")
        else:
            print(f"\n  {phase.value.upper()}: Exploring (epsilon-greedy) or no data")


def demonstrate_adaptive_questioning():
    """Demonstrate Adaptive Questioning - optimizing question asking."""
    print_section("ADAPTIVE QUESTIONING - Learning Question Patterns")

    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "questions_demo.db"

    questioner = AdaptiveQuestioner(db_path=db_path)

    print("Initialized AdaptiveQuestioner")

    # Create sample ambiguities
    ambiguities = [
        AmbiguityMarker(
            category="SUBJECTIVE",
            text="good performance",
            impact="BLOCKING",
            suggested_question="What specific metric defines 'good performance'?",
            suggested_answer="Response time < 200ms",
        ),
        AmbiguityMarker(
            category="UNDEFINED_TERM",
            text="the database",
            impact="CLARIFYING",
            suggested_question="Which database are you referring to?",
            suggested_answer="PostgreSQL",
        ),
        AmbiguityMarker(
            category="COMPARATIVE",
            text="faster than before",
            impact="CLARIFYING",
            suggested_question="How much faster? What's the baseline?",
            suggested_answer=None,
        ),
        AmbiguityMarker(
            category="CONTRADICTIONS",
            text="should be fast but also thorough",
            impact="BLOCKING",
            suggested_question="How to prioritize between speed and thoroughness?",
            suggested_answer=None,
        ),
    ]

    print(f"\n→ Original ambiguities: {len(ambiguities)}")
    for amb in ambiguities:
        print(f"  - {amb.category} ({amb.impact}): {amb.text}")

    # Prioritize questions with default priorities
    print("\n→ Prioritizing questions (default priorities)...")
    prioritized = questioner.prioritize_questions(ambiguities)

    print(f"  Prioritized order:")
    for i, amb in enumerate(prioritized, 1):
        print(f"  {i}. {amb.category} ({amb.impact}): {amb.text}")

    # Try with speed-optimized priorities
    print("\n→ Prioritizing with SPEED optimization (max 2 questions)...")
    speed_priorities = UserPriorities(
        speed_weight=0.8,
        cost_weight=0.1,
        control_weight=0.1,
        max_clarification_questions=2,
    )

    prioritized_speed = questioner.prioritize_questions(ambiguities, priorities=speed_priorities)

    print(f"  Top {len(prioritized_speed)} questions:")
    for i, amb in enumerate(prioritized_speed, 1):
        print(f"  {i}. {amb.category} ({amb.impact}): {amb.text}")

    # Simulate recording some question outcomes
    print("\n→ Simulating 20 question interactions to learn patterns...")

    for i in range(20):
        # Pick an ambiguity
        amb = ambiguities[i % len(ambiguities)]

        # Simulate user behavior
        # SUBJECTIVE: 90% answered, 80% use suggestion
        # UNDEFINED_TERM: 70% answered, 60% use suggestion
        # COMPARATIVE: 50% answered, 30% use suggestion
        # CONTRADICTIONS: 95% answered, 20% use suggestion

        if amb.category == "SUBJECTIVE":
            answered = (i % 10) < 9
            used_suggestion = answered and (i % 10) < 8
        elif amb.category == "UNDEFINED_TERM":
            answered = (i % 10) < 7
            used_suggestion = answered and (i % 10) < 6
        elif amb.category == "COMPARATIVE":
            answered = (i % 10) < 5
            used_suggestion = answered and (i % 10) < 3
        else:  # CONTRADICTIONS
            answered = (i % 10) < 9
            used_suggestion = answered and (i % 10) < 2

        question_id = questioner.record_question_outcome(
            session_id=f"session_{i}",
            ambiguity=amb,
            was_answered=answered,
            user_answer=amb.suggested_answer if used_suggestion else f"Custom answer {i}",
            used_suggestion=used_suggestion,
            answer_quality_score=1.0 if answered else 0.0,
        )

        # Update outcome (assume most sessions succeed)
        questioner.update_question_outcome(
            question_id=question_id,
            session_id=f"session_{i}",
            led_to_success=(i % 5) < 4,  # 80% success
        )

    print("✓ Recorded 20 question interactions")

    # Get statistics
    print("\n→ Question Statistics by Category:")

    for category in ["SUBJECTIVE", "UNDEFINED_TERM", "COMPARATIVE", "CONTRADICTIONS"]:
        stats = questioner.get_question_stats(category=category)

        if stats["total_questions"] > 0:
            print(f"\n  {category}:")
            print(f"    Total Asked: {stats['total_questions']}")
            print(f"    Answered: {stats['answered']} ({(1 - stats['skip_rate']) * 100:.0f}%)")
            print(f"    Skipped: {stats['skipped']} ({stats['skip_rate'] * 100:.0f}%)")
            print(f"    Used Suggestions: {stats['used_suggestions']} ({stats['suggestion_rate'] * 100:.0f}%)")
            print(f"    Avg Quality: {stats['avg_quality_score']:.2f}")


def main():
    """Run the full learning system demonstration."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PARAGON LEARNING SYSTEM DEMO" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")

    # Part 1: Study Mode
    manager = demonstrate_study_mode()

    # Part 2: Transition Analysis
    manager = demonstrate_transition(manager)

    # Part 3: Production Mode
    demonstrate_production_mode(manager)

    # Part 4: Adaptive Questioning
    demonstrate_adaptive_questioning()

    print_section("DEMO COMPLETE")
    print("Key Takeaways:")
    print("  1. STUDY mode collects data without biasing decisions")
    print("  2. System tracks success rates per model and phase")
    print("  3. Transition to PRODUCTION requires 100 sessions minimum")
    print("  4. PRODUCTION mode uses learned patterns to optimize")
    print("  5. Adaptive questioning learns from user behavior patterns")
    print("  6. Question prioritization respects user priorities")
    print("\nThe learning system enables Paragon to improve over time!")
    print()


if __name__ == "__main__":
    main()
