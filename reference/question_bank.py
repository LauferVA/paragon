"""
PARAGON QUESTION BANK

Structured question library derived from templates.

This module provides:
- QuestionBank: Central repository of questions
- Question categorization by domain and priority
- Question dependencies (follow-up questions)
- Integration with template matcher for gap-based questions

Architecture:
- Questions are loaded from templates at runtime
- Support for both generic and domain-specific questions
- Extensible design for custom question sets
"""
from pathlib import Path
from typing import List, Dict, Optional, Set
import msgspec

from .matcher import TemplateMatcher, TemplateSection


# =============================================================================
# QUESTION SCHEMAS
# =============================================================================

class Question(msgspec.Struct, kw_only=True):
    """
    A single question in the question bank.

    Questions can be generic, domain-specific, or gap-specific.
    """
    id: str                         # Unique identifier (e.g., "web_app.auth.1")
    text: str                       # The question text
    category: str                   # Category (e.g., "Authentication", "Architecture")
    priority: str                   # "low", "medium", "high", "critical"
    domain: Optional[str] = None    # Domain this applies to (or None for generic)
    template_section: Optional[str] = None  # Which template section this comes from
    follow_ups: List[str] = []      # IDs of follow-up questions
    examples: List[str] = []        # Example answers
    rationale: Optional[str] = None # Why this question matters


class QuestionSet(msgspec.Struct, kw_only=True):
    """
    A set of related questions.

    Groups questions by category or section.
    """
    name: str                       # Set name (e.g., "Web App Authentication")
    description: str                # Brief description
    questions: List[Question]       # Questions in this set
    priority: str = "medium"        # Overall priority


# =============================================================================
# QUESTION BANK
# =============================================================================

class QuestionBank:
    """
    Central repository of questions derived from templates.

    Loads questions from template files and provides methods to retrieve
    questions based on domain, category, or gaps in a specification.

    Usage:
        bank = QuestionBank()

        # Get all questions for a domain
        web_questions = bank.get_domain_questions("web_app")

        # Get generic questions for all projects
        generic = bank.get_generic_questions()

        # Get questions to fill gaps in a spec
        gaps = bank.get_gap_questions(template="prd_template", spec_content="...")
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the question bank.

        Args:
            templates_dir: Path to templates directory. If None, uses default.
        """
        self.matcher = TemplateMatcher(templates_dir=templates_dir)
        self._questions_cache: Dict[str, List[Question]] = {}

    def get_generic_questions(
        self,
        category: Optional[str] = None,
        min_priority: str = "low",
    ) -> List[Question]:
        """
        Get generic questions applicable to all projects.

        Args:
            category: Filter by category (e.g., "Architecture", "Security")
            min_priority: Minimum priority ("low", "medium", "high", "critical")

        Returns:
            List of generic questions
        """
        questions: List[Question] = []

        # Load questions from all generic templates
        generic_templates = [
            "prd_template",
            "tech_spec_template",
            "api_design_template",
            "user_story_template",
        ]

        for template in generic_templates:
            template_questions = self._load_template_questions(
                template_name=template,
                template_type="generic",
            )
            questions.extend(template_questions)

        # Filter by category if specified
        if category:
            questions = [q for q in questions if q.category == category]

        # Filter by priority
        questions = self._filter_by_priority(questions, min_priority)

        return questions

    def get_domain_questions(
        self,
        domain: str,
        category: Optional[str] = None,
        min_priority: str = "low",
    ) -> List[Question]:
        """
        Get domain-specific questions.

        Args:
            domain: Domain name (e.g., "web_app", "cli_tool", "api_service")
            category: Filter by category
            min_priority: Minimum priority

        Returns:
            List of domain-specific questions
        """
        # Load questions from domain template
        questions = self._load_template_questions(
            template_name=domain,
            template_type="domain",
        )

        # Filter by category if specified
        if category:
            questions = [q for q in questions if q.category == category]

        # Filter by priority
        questions = self._filter_by_priority(questions, min_priority)

        return questions

    def get_gap_questions(
        self,
        template: str,
        spec_content: str,
        template_type: str = "generic",
        min_priority: str = "low",
    ) -> List[Question]:
        """
        Get questions to fill gaps in a specification.

        Compares the spec to a template and returns questions for missing sections.

        Args:
            template: Template name (e.g., "prd_template", "web_app")
            spec_content: The user's specification text
            template_type: "generic" or "domain"
            min_priority: Minimum priority

        Returns:
            List of questions to fill gaps
        """
        # Get template path
        if template_type == "generic":
            template_path = self.matcher.templates_dir / "generic" / f"{template}.md"
        else:
            template_path = self.matcher.templates_dir / "domain" / f"{template}.md"

        if not template_path.exists():
            return []

        # Analyze gaps
        gap_analysis = self.matcher.analyze_gaps(spec_content, str(template_path))

        # Load all questions from template
        all_questions = self._load_template_questions(template, template_type)

        # Filter to only questions from missing sections
        gap_questions = [
            q for q in all_questions
            if q.template_section in gap_analysis.missing_sections
        ]

        # Filter by priority
        gap_questions = self._filter_by_priority(gap_questions, min_priority)

        return gap_questions

    def get_question_sets(self, domain: Optional[str] = None) -> List[QuestionSet]:
        """
        Get organized question sets.

        Groups questions by category for easier navigation.

        Args:
            domain: Optional domain to filter by

        Returns:
            List of question sets
        """
        # Get questions
        if domain:
            questions = self.get_domain_questions(domain)
        else:
            questions = self.get_generic_questions()

        # Group by category
        by_category: Dict[str, List[Question]] = {}
        for q in questions:
            category = q.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(q)

        # Create question sets
        question_sets: List[QuestionSet] = []
        for category, cat_questions in by_category.items():
            # Determine overall priority (highest priority in the set)
            priorities = ["low", "medium", "high", "critical"]
            max_priority = "low"
            for q in cat_questions:
                if priorities.index(q.priority) > priorities.index(max_priority):
                    max_priority = q.priority

            question_sets.append(QuestionSet(
                name=category,
                description=f"Questions about {category.lower()}",
                questions=cat_questions,
                priority=max_priority,
            ))

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        question_sets.sort(key=lambda qs: priority_order.get(qs.priority, 4))

        return question_sets

    def search_questions(
        self,
        query: str,
        domain: Optional[str] = None,
    ) -> List[Question]:
        """
        Search for questions by keyword.

        Args:
            query: Search query
            domain: Optional domain to filter by

        Returns:
            List of matching questions
        """
        # Get questions
        if domain:
            questions = self.get_domain_questions(domain)
        else:
            questions = self.get_generic_questions()
            # Also include all domain questions
            for d in ["web_app", "cli_tool", "api_service", "data_pipeline", "ml_system"]:
                questions.extend(self.get_domain_questions(d))

        # Search in question text and category
        query_lower = query.lower()
        matches = [
            q for q in questions
            if query_lower in q.text.lower() or query_lower in q.category.lower()
        ]

        return matches

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _load_template_questions(
        self,
        template_name: str,
        template_type: str,
    ) -> List[Question]:
        """
        Load questions from a template file.

        Args:
            template_name: Template name
            template_type: "generic" or "domain"

        Returns:
            List of questions from the template
        """
        cache_key = f"{template_type}:{template_name}"

        # Check cache
        if cache_key in self._questions_cache:
            return self._questions_cache[cache_key]

        # Get template path
        if template_type == "generic":
            template_path = self.matcher.templates_dir / "generic" / f"{template_name}.md"
        else:
            template_path = self.matcher.templates_dir / "domain" / f"{template_name}.md"

        if not template_path.exists():
            return []

        # Parse template sections
        sections = self.matcher.get_template_sections(str(template_path))

        # Convert sections to questions
        questions: List[Question] = []
        for section in sections:
            for i, question_text in enumerate(section.questions):
                question_id = f"{template_name}.{section.title.lower().replace(' ', '_')}.{i}"

                questions.append(Question(
                    id=question_id,
                    text=question_text,
                    category=section.title,
                    priority=section.priority,
                    domain=template_name if template_type == "domain" else None,
                    template_section=section.title,
                ))

        # Cache for future use
        self._questions_cache[cache_key] = questions

        return questions

    def _filter_by_priority(
        self,
        questions: List[Question],
        min_priority: str,
    ) -> List[Question]:
        """
        Filter questions by minimum priority.

        Args:
            questions: Questions to filter
            min_priority: Minimum priority

        Returns:
            Filtered questions
        """
        priority_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = priority_levels.get(min_priority, 0)

        return [
            q for q in questions
            if priority_levels.get(q.priority, 0) >= min_level
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_questions_for_project_type(project_description: str) -> List[Question]:
    """
    Get relevant questions based on project description.

    Uses template matching to identify project type, then returns
    appropriate questions.

    Args:
        project_description: User's project description

    Returns:
        List of relevant questions
    """
    # Match templates
    matcher = TemplateMatcher()
    matches = matcher.match_templates(project_description, max_results=3)

    if not matches:
        # No matches, return generic questions
        bank = QuestionBank()
        return bank.get_generic_questions(min_priority="medium")

    # Get questions from top match
    top_match = matches[0]
    bank = QuestionBank()

    if top_match.template_type == "domain":
        return bank.get_domain_questions(top_match.template_name, min_priority="medium")
    else:
        return bank.get_generic_questions(min_priority="medium")


def get_critical_questions(domain: Optional[str] = None) -> List[Question]:
    """
    Get critical questions that must be answered.

    Args:
        domain: Optional domain to filter by

    Returns:
        List of critical questions
    """
    bank = QuestionBank()

    if domain:
        questions = bank.get_domain_questions(domain, min_priority="critical")
    else:
        questions = bank.get_generic_questions(min_priority="critical")

    return questions


def get_all_categories(domain: Optional[str] = None) -> List[str]:
    """
    Get all question categories.

    Args:
        domain: Optional domain to filter by

    Returns:
        List of category names
    """
    bank = QuestionBank()

    if domain:
        questions = bank.get_domain_questions(domain)
    else:
        questions = bank.get_generic_questions()

    categories: Set[str] = {q.category for q in questions}
    return sorted(list(categories))
