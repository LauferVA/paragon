"""
Example usage of the Paragon Reference System.

This script demonstrates how to use the reference system for:
1. Template matching
2. Gap analysis
3. Question generation
4. Integration with Socratic Engine
"""
from reference import TemplateMatcher, QuestionBank
from reference.matcher import match_templates_to_description
from reference.question_bank import get_questions_for_project_type, get_critical_questions


def example_1_template_matching():
    """Example 1: Match templates to project description."""
    print("=" * 70)
    print("EXAMPLE 1: Template Matching")
    print("=" * 70)

    descriptions = [
        "Build a web dashboard for visualizing service dependencies",
        "Create a CLI tool for checking microservice conflicts",
        "Implement a REST API for dependency management",
        "Build a data pipeline to process service metadata",
        "Train a machine learning model to predict conflicts",
    ]

    matcher = TemplateMatcher()

    for desc in descriptions:
        print(f"\nDescription: {desc}")
        matches = matcher.match_templates(desc, max_results=2)

        if matches:
            print(f"  Best match: {matches[0].template_name} ({matches[0].relevance_score:.0%})")
            print(f"  Keywords: {', '.join(matches[0].matched_keywords[:5])}")
        else:
            print("  No matches found")


def example_2_gap_analysis():
    """Example 2: Analyze gaps in a specification."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Gap Analysis")
    print("=" * 70)

    # User provides incomplete spec
    user_spec = """
    ## Overview
    A web application for dependency visualization.

    ## Features
    - View dependency graph
    - Check for conflicts
    - Export to SVG

    ## Technical Stack
    - React + TypeScript
    - D3.js for visualization
    - REST API backend
    """

    matcher = TemplateMatcher()

    # Analyze against PRD template
    prd_path = str(matcher.templates_dir / "generic" / "prd_template.md")
    prd_gaps = matcher.analyze_gaps(user_spec, prd_path)

    print(f"\nPRD Template Analysis:")
    print(f"  Coverage: {prd_gaps.coverage_score:.0%}")
    print(f"  Covered sections: {', '.join(prd_gaps.covered_sections[:3])}...")
    print(f"  Missing sections: {', '.join(prd_gaps.missing_sections[:3])}...")
    print(f"  Total questions to ask: {len(prd_gaps.questions_for_gaps)}")

    # Analyze against Web App template
    web_path = str(matcher.templates_dir / "domain" / "web_app.md")
    web_gaps = matcher.analyze_gaps(user_spec, web_path)

    print(f"\nWeb App Template Analysis:")
    print(f"  Coverage: {web_gaps.coverage_score:.0%}")
    print(f"  Missing sections: {len(web_gaps.missing_sections)}")
    print(f"  Total questions to ask: {len(web_gaps.questions_for_gaps)}")

    # Show sample questions
    print(f"\nSample questions to fill gaps:")
    for i, q in enumerate(prd_gaps.questions_for_gaps[:5], 1):
        print(f"  {i}. {q}")


def example_3_question_generation():
    """Example 3: Generate questions from templates."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Question Generation")
    print("=" * 70)

    bank = QuestionBank()

    # Get critical generic questions
    print("\nCritical generic questions:")
    critical = bank.get_generic_questions(min_priority="critical")
    for q in critical[:3]:
        print(f"  - [{q.category}] {q.text}")

    # Get high-priority API service questions
    print("\nHigh-priority API service questions:")
    api_questions = bank.get_domain_questions("api_service", min_priority="high")
    for q in api_questions[:5]:
        print(f"  - [{q.category}] {q.text}")

    # Search questions
    print("\nSearch for 'authentication' questions:")
    auth_questions = bank.search_questions("authentication")
    for q in auth_questions[:3]:
        domain = q.domain or "generic"
        print(f"  - [{domain}/{q.category}] {q.text}")


def example_4_question_sets():
    """Example 4: Organize questions into sets."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Question Sets")
    print("=" * 70)

    bank = QuestionBank()

    # Get question sets for web apps
    question_sets = bank.get_question_sets(domain="web_app")

    print(f"\nFound {len(question_sets)} question sets for web apps:")
    for qs in question_sets[:5]:
        print(f"  - {qs.name} ({qs.priority}): {len(qs.questions)} questions")


def example_5_convenience_functions():
    """Example 5: Use convenience functions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Convenience Functions")
    print("=" * 70)

    # Automatic template matching + question generation
    description = "Build a command-line tool for dependency analysis"

    print(f"Description: {description}")
    questions = get_questions_for_project_type(description)

    print(f"\nGenerated {len(questions)} relevant questions:")
    for q in questions[:5]:
        print(f"  - [{q.category}] {q.text}")

    # Get critical questions for CLI tools
    print("\nCritical CLI tool questions:")
    critical = get_critical_questions(domain="cli_tool")
    for q in critical[:3]:
        print(f"  - {q.text}")


def example_6_full_workflow():
    """Example 6: Complete workflow from description to questions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Full Workflow")
    print("=" * 70)

    # Step 1: User provides description
    user_input = """
    Build a REST API that manages service dependencies.
    Users can register services, define dependencies, and query for conflicts.
    """

    print(f"User input:\n{user_input}")

    # Step 2: Match templates
    print("\nStep 1: Matching templates...")
    matches = match_templates_to_description(user_input)

    if not matches:
        print("  No templates matched")
        return

    top_match = matches[0]
    print(f"  Detected project type: {top_match.template_name} ({top_match.relevance_score:.0%})")

    # Step 3: Get initial questions for this domain
    print("\nStep 2: Getting domain questions...")
    bank = QuestionBank()

    if top_match.template_type == "domain":
        questions = bank.get_domain_questions(
            top_match.template_name,
            min_priority="high"
        )
    else:
        questions = bank.get_generic_questions(min_priority="high")

    print(f"  Found {len(questions)} high-priority questions")

    # Step 4: Analyze gaps in user input
    print("\nStep 3: Analyzing gaps...")
    matcher = TemplateMatcher()
    gap_analysis = matcher.analyze_gaps(user_input, top_match.template_path)

    print(f"  Specification coverage: {gap_analysis.coverage_score:.0%}")
    print(f"  Missing sections: {len(gap_analysis.missing_sections)}")

    # Step 5: Generate targeted questions
    print("\nStep 4: Generating targeted questions...")
    gap_questions = bank.get_gap_questions(
        template=top_match.template_name,
        spec_content=user_input,
        template_type=top_match.template_type,
        min_priority="medium"
    )

    print(f"  Generated {len(gap_questions)} gap-filling questions")

    # Step 6: Show questions to ask
    print("\nQuestions to ask the user:")
    all_questions = gap_questions[:10]  # Limit to 10 for demo

    by_category = {}
    for q in all_questions:
        if q.category not in by_category:
            by_category[q.category] = []
        by_category[q.category].append(q)

    for category, cat_questions in list(by_category.items())[:3]:
        print(f"\n  [{category}]")
        for q in cat_questions[:3]:
            print(f"    - {q.text}")


def main():
    """Run all examples."""
    examples = [
        example_1_template_matching,
        example_2_gap_analysis,
        example_3_question_generation,
        example_4_question_sets,
        example_5_convenience_functions,
        example_6_full_workflow,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
