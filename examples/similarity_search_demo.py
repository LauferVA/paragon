"""
PARAGON SIMILARITY SEARCH DEMO

Demonstrates the three-level similarity search system:
1. Project-level: Find similar repositories
2. Module-level: Find reusable components
3. Implementation-level: Find specific code

Usage:
    python examples/similarity_search_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.repo_scanner import create_scanner
from agents.similarity_advisor import create_advisor


async def demo_project_search():
    """Demonstrate project-level similarity search."""
    print("\n" + "=" * 70)
    print("LEVEL 1: PROJECT SIMILARITY SEARCH")
    print("=" * 70)

    scanner = create_scanner()
    advisor = create_advisor()

    # Example: searching for a graph database project
    spec = """
    Build a graph database for storing code dependencies.
    Should support nodes, edges, and queries. Need to track
    relationships between code components and perform graph
    traversals like topological sort.
    """

    print(f"\nSearching for projects similar to:\n{spec.strip()}\n")

    projects = await scanner.search_similar_projects(
        description=spec,
        sources=["github"],
        max_results=5,
        min_similarity=0.5,
    )

    if projects:
        print(f"\nFound {len(projects)} similar projects:\n")
        for project in projects:
            print(advisor.format_project_suggestion(project, spec))
    else:
        print("\nNo similar projects found.")


async def demo_module_search():
    """Demonstrate module-level similarity search."""
    print("\n" + "=" * 70)
    print("LEVEL 2: MODULE SIMILARITY SEARCH")
    print("=" * 70)

    scanner = create_scanner()
    advisor = create_advisor()

    # Example: searching for a specific module
    module_spec = """
    A module for parsing Python code using tree-sitter.
    Should extract functions, classes, and docstrings.
    """

    print(f"\nSearching for modules similar to:\n{module_spec.strip()}\n")

    modules = await scanner.search_similar_modules(
        module_spec=module_spec,
        language="python",
        max_results=5,
        min_similarity=0.6,
    )

    if modules:
        print(f"\nFound {len(modules)} similar modules:\n")
        for module in modules:
            print(advisor.format_module_suggestion(module, module_spec))
    else:
        print("\nNo similar modules found.")


async def demo_implementation_search():
    """Demonstrate implementation-level similarity search."""
    print("\n" + "=" * 70)
    print("LEVEL 3: IMPLEMENTATION SIMILARITY SEARCH")
    print("=" * 70)

    scanner = create_scanner()
    advisor = create_advisor()

    # Example: searching for a specific algorithm
    task_description = "topological sort for directed acyclic graph"

    print(f"\nSearching for implementations of:\n{task_description}\n")

    implementations = await scanner.search_similar_implementations(
        task_description=task_description,
        language="python",
        max_results=3,
        min_similarity=0.7,
    )

    if implementations:
        print(f"\nFound {len(implementations)} similar implementations:\n")
        for impl in implementations:
            print(advisor.format_implementation_suggestion(impl, task_description))
    else:
        print("\nNo similar implementations found.")


async def demo_full_search():
    """Demonstrate all three levels together."""
    print("\n" + "=" * 70)
    print("FULL SIMILARITY SEARCH (ALL LEVELS)")
    print("=" * 70)

    scanner = create_scanner()
    advisor = create_advisor()

    spec = "Build a REST API for managing user authentication with JWT tokens"

    print(f"\nQuery: {spec}\n")

    # Search all levels
    projects = await scanner.search_similar_projects(spec, max_results=3)
    modules = await scanner.search_similar_modules(spec, max_results=3)
    implementations = await scanner.search_similar_implementations(spec, max_results=2)

    # Format summary
    summary = advisor.format_summary(
        projects=projects,
        modules=modules,
        implementations=implementations,
    )

    print(summary)


async def main():
    """Run all demos."""
    print("\n🔍 PARAGON SIMILARITY SEARCH DEMONSTRATION\n")
    print("This demo shows how Paragon can find existing code at three levels:")
    print("1. Whole projects that solve similar problems")
    print("2. Reusable modules/components")
    print("3. Specific code implementations")
    print("\nNote: Requires internet connection for GitHub search.")
    print("      Results depend on GitHub API availability and rate limits.")

    # Run demos
    try:
        await demo_project_search()
        await demo_module_search()
        await demo_implementation_search()
        await demo_full_search()

        print("\n" + "=" * 70)
        print("✅ DEMO COMPLETE")
        print("=" * 70)
        print("\nThe similarity search system can help you:")
        print("- Discover existing solutions before building from scratch")
        print("- Find reusable components to integrate")
        print("- Learn from similar implementations")
        print("\nIntegrate this into your research phase to save development time!")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nCommon issues:")
        print("- No internet connection (needed for GitHub search)")
        print("- GitHub API rate limit exceeded (60/hour without token)")
        print("- Missing dependencies (requests library)")
        print("\nTo increase rate limits, set GITHUB_TOKEN environment variable.")


if __name__ == "__main__":
    asyncio.run(main())
