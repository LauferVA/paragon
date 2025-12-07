"""
PARAGON REFERENCE MATCHER

Template matching system that identifies relevant templates based on user descriptions.

This module provides:
- TemplateMatcher: Matches user descriptions to templates
- Gap analysis: Compares specs to templates to find missing sections
- Ranking: Orders templates by relevance

Architecture:
- Keyword-based matching for speed (no embeddings in v1)
- Support for both generic and domain-specific templates
- Extensible design for adding new templates
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import msgspec


# =============================================================================
# TEMPLATE MATCHING SCHEMAS
# =============================================================================

class MatchedTemplate(msgspec.Struct, kw_only=True):
    """
    A template that matches a user description.

    Contains the template metadata and relevance score.
    """
    template_name: str          # e.g., "prd_template", "web_app"
    template_type: str          # "generic" or "domain"
    template_path: str          # Full path to template file
    relevance_score: float      # 0.0 to 1.0 (higher = more relevant)
    matched_keywords: List[str] # Keywords that triggered this match
    description: str = ""       # Brief description of template


class TemplateSection(msgspec.Struct, kw_only=True):
    """
    A section within a template.

    Represents a major section with associated questions.
    """
    title: str                  # Section title (e.g., "Problem Statement")
    questions: List[str]        # Questions to ask if this section is missing
    keywords: List[str]         # Keywords that indicate this section is covered
    priority: str = "medium"    # "low", "medium", "high", "critical"


class GapAnalysis(msgspec.Struct, kw_only=True):
    """
    Result of comparing a spec to a template.

    Identifies missing sections and questions to ask.
    """
    template_name: str
    covered_sections: List[str]     # Sections present in the spec
    missing_sections: List[str]     # Sections absent from the spec
    questions_for_gaps: List[str]   # Questions to fill gaps
    coverage_score: float           # 0.0 to 1.0 (percentage covered)


# =============================================================================
# TEMPLATE MATCHER
# =============================================================================

class TemplateMatcher:
    """
    Matches user descriptions to relevant templates.

    Uses keyword-based matching to identify which templates are most
    relevant to a user's project description.

    Usage:
        matcher = TemplateMatcher()
        matches = matcher.match_templates("Build a REST API for user management")
        for match in matches:
            print(f"{match.template_name}: {match.relevance_score:.2f}")
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template matcher.

        Args:
            templates_dir: Path to templates directory. If None, uses default.
        """
        if templates_dir is None:
            # Default to reference/templates
            templates_dir = str(Path(__file__).parent / "templates")

        self.templates_dir = Path(templates_dir)

        # Domain-specific keyword patterns
        self._domain_keywords: Dict[str, List[str]] = {
            "web_app": [
                "web app", "webapp", "website", "frontend", "react", "vue",
                "angular", "ui", "user interface", "browser", "spa", "single page",
                "dashboard", "admin panel", "web portal", "responsive"
            ],
            "cli_tool": [
                "cli", "command line", "command-line", "terminal", "shell",
                "console", "script", "automation", "batch", "utility"
            ],
            "api_service": [
                "api", "rest", "graphql", "grpc", "backend", "service",
                "microservice", "endpoint", "web service", "http", "server",
                "restful", "json api"
            ],
            "data_pipeline": [
                "pipeline", "etl", "data processing", "batch job", "airflow",
                "data flow", "ingestion", "transformation", "streaming",
                "kafka", "spark", "extract transform load"
            ],
            "ml_system": [
                "machine learning", "ml", "model", "training", "prediction",
                "classification", "regression", "neural network", "ai",
                "deep learning", "tensorflow", "pytorch", "scikit"
            ],
        }

        # Generic template relevance keywords
        self._generic_keywords: Dict[str, List[str]] = {
            "prd_template": [
                "product", "requirements", "features", "problem statement",
                "user stories", "success metrics", "scope"
            ],
            "tech_spec_template": [
                "technical", "architecture", "design", "implementation",
                "system design", "technical specification", "data model"
            ],
            "api_design_template": [
                "api design", "endpoints", "api specification", "openapi",
                "swagger", "api documentation", "request response"
            ],
            "user_story_template": [
                "user story", "acceptance criteria", "persona", "workflow",
                "use case", "scenario"
            ],
        }

    def match_templates(
        self,
        description: str,
        min_score: float = 0.1,
        max_results: int = 5,
    ) -> List[MatchedTemplate]:
        """
        Find templates that match the description.

        Args:
            description: User's project description
            min_score: Minimum relevance score to include (0.0 to 1.0)
            max_results: Maximum number of results to return

        Returns:
            List of matched templates, ordered by relevance (highest first)
        """
        description_lower = description.lower()
        matches: List[MatchedTemplate] = []

        # Match domain-specific templates
        for domain, keywords in self._domain_keywords.items():
            score, matched_kw = self._calculate_score(description_lower, keywords)
            if score >= min_score:
                template_path = self.templates_dir / "domain" / f"{domain}.md"
                if template_path.exists():
                    matches.append(MatchedTemplate(
                        template_name=domain,
                        template_type="domain",
                        template_path=str(template_path),
                        relevance_score=score,
                        matched_keywords=matched_kw,
                        description=f"Domain-specific template for {domain.replace('_', ' ')}",
                    ))

        # Match generic templates
        for template, keywords in self._generic_keywords.items():
            score, matched_kw = self._calculate_score(description_lower, keywords)
            if score >= min_score:
                template_path = self.templates_dir / "generic" / f"{template}.md"
                if template_path.exists():
                    matches.append(MatchedTemplate(
                        template_name=template,
                        template_type="generic",
                        template_path=str(template_path),
                        relevance_score=score,
                        matched_keywords=matched_kw,
                        description=f"Generic {template.replace('_', ' ').replace(' template', '')}",
                    ))

        # Sort by relevance score (descending)
        matches.sort(key=lambda m: m.relevance_score, reverse=True)

        return matches[:max_results]

    def _calculate_score(
        self,
        description: str,
        keywords: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Calculate relevance score based on keyword matching.

        Args:
            description: Lowercased user description
            keywords: List of keywords to match

        Returns:
            Tuple of (score, matched_keywords)
        """
        matched: List[str] = []
        total_weight = 0.0

        for keyword in keywords:
            # Check if keyword appears in description
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, description):
                matched.append(keyword)
                # Longer keywords get higher weight (more specific)
                weight = len(keyword.split()) * 0.3 + 0.7
                total_weight += weight

        if not matched:
            return 0.0, []

        # Normalize score to 0-1 range
        # More matches and longer keywords = higher score
        score = min(1.0, total_weight / len(keywords))

        return score, matched

    def get_template_sections(self, template_path: str) -> List[TemplateSection]:
        """
        Parse a template file to extract sections and questions.

        Args:
            template_path: Path to the template markdown file

        Returns:
            List of template sections with questions
        """
        sections: List[TemplateSection] = []

        try:
            content = Path(template_path).read_text()
        except FileNotFoundError:
            return sections

        # Split by ## headers (sections)
        section_pattern = r'^## (.+)$'
        parts = re.split(section_pattern, content, flags=re.MULTILINE)

        # parts[0] is content before first section (title, description)
        # parts[1::2] are section titles
        # parts[2::2] are section contents

        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break

            title = parts[i].strip()
            content = parts[i + 1].strip()

            # Extract questions (lines starting with "- ")
            questions = self._extract_questions(content)

            # Extract keywords from the section content
            keywords = self._extract_keywords(title, content)

            # Determine priority based on title
            priority = self._determine_priority(title)

            sections.append(TemplateSection(
                title=title,
                questions=questions,
                keywords=keywords,
                priority=priority,
            ))

        return sections

    def _extract_questions(self, content: str) -> List[str]:
        """Extract questions from section content."""
        questions: List[str] = []

        # Look for "Questions if missing:" or "Questions to ask:" sections
        if "questions if missing:" in content.lower() or "questions to ask:" in content.lower():
            # Extract bullet points after the questions header
            lines = content.split('\n')
            in_questions = False

            for line in lines:
                line_lower = line.lower().strip()

                if 'questions if missing:' in line_lower or 'questions to ask:' in line_lower:
                    in_questions = True
                    continue

                if in_questions:
                    # End of questions section (new header or blank line followed by text)
                    if line.startswith('**') and '**' in line[2:]:
                        if 'example' in line.lower():
                            break

                    # Extract question bullet point
                    if line.strip().startswith('-'):
                        question = line.strip()[1:].strip()
                        if question and question.endswith('?'):
                            questions.append(question)

        return questions

    def _extract_keywords(self, title: str, content: str) -> List[str]:
        """Extract keywords from section title and content."""
        keywords: Set[str] = set()

        # Add title words
        title_words = re.findall(r'\b\w{4,}\b', title.lower())
        keywords.update(title_words)

        # Extract emphasized terms (between ** or `)
        emphasized = re.findall(r'\*\*([^*]+)\*\*|`([^`]+)`', content)
        for match in emphasized:
            for term in match:
                if term:
                    keywords.add(term.lower().strip())

        # Extract common technical terms
        tech_terms = re.findall(
            r'\b(?:api|database|authentication|authorization|testing|deployment|'
            r'performance|security|monitoring|caching|error|validation)\b',
            content.lower()
        )
        keywords.update(tech_terms)

        return sorted(list(keywords))

    def _determine_priority(self, title: str) -> str:
        """Determine section priority based on title."""
        title_lower = title.lower()

        # Critical sections
        if any(term in title_lower for term in [
            'overview', 'problem statement', 'scope', 'requirements',
            'authentication', 'security', 'error handling'
        ]):
            return "critical"

        # High priority sections
        if any(term in title_lower for term in [
            'architecture', 'data model', 'api', 'testing',
            'deployment', 'performance'
        ]):
            return "high"

        # Low priority sections
        if any(term in title_lower for term in [
            'open questions', 'future', 'nice to have', 'optional'
        ]):
            return "low"

        # Default to medium
        return "medium"

    def analyze_gaps(
        self,
        spec_content: str,
        template_path: str,
    ) -> GapAnalysis:
        """
        Compare a spec to a template and identify gaps.

        Args:
            spec_content: The user's specification text
            template_path: Path to template to compare against

        Returns:
            Gap analysis with missing sections and questions
        """
        sections = self.get_template_sections(template_path)

        spec_lower = spec_content.lower()
        covered: List[str] = []
        missing: List[str] = []
        questions: List[str] = []

        for section in sections:
            # Check if section is covered by looking for keywords
            is_covered = any(
                keyword in spec_lower
                for keyword in section.keywords
            )

            if is_covered:
                covered.append(section.title)
            else:
                missing.append(section.title)
                # Add questions for this gap
                questions.extend(section.questions)

        # Calculate coverage score
        total_sections = len(sections)
        coverage_score = len(covered) / total_sections if total_sections > 0 else 0.0

        template_name = Path(template_path).stem

        return GapAnalysis(
            template_name=template_name,
            covered_sections=covered,
            missing_sections=missing,
            questions_for_gaps=questions,
            coverage_score=coverage_score,
        )

    def get_questions_for_gaps(
        self,
        spec_content: str,
        template_match: MatchedTemplate,
    ) -> List[str]:
        """
        Get questions to fill gaps in a spec based on a matched template.

        This is a convenience method that combines gap analysis with
        question extraction.

        Args:
            spec_content: The user's specification text
            template_match: A matched template to compare against

        Returns:
            List of questions to ask
        """
        gap_analysis = self.analyze_gaps(spec_content, template_match.template_path)
        return gap_analysis.questions_for_gaps


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def match_templates_to_description(description: str) -> List[MatchedTemplate]:
    """
    Convenience function to match templates to a description.

    Args:
        description: User's project description

    Returns:
        List of matched templates
    """
    matcher = TemplateMatcher()
    return matcher.match_templates(description)


def get_questions_for_spec(
    spec_content: str,
    template_name: str,
) -> List[str]:
    """
    Convenience function to get questions for a spec.

    Args:
        spec_content: The user's specification
        template_name: Name of template to use (e.g., "prd_template", "web_app")

    Returns:
        List of questions to ask
    """
    matcher = TemplateMatcher()

    # Find template path
    generic_path = matcher.templates_dir / "generic" / f"{template_name}.md"
    domain_path = matcher.templates_dir / "domain" / f"{template_name}.md"

    if generic_path.exists():
        template_path = str(generic_path)
    elif domain_path.exists():
        template_path = str(domain_path)
    else:
        return []

    gap_analysis = matcher.analyze_gaps(spec_content, template_path)
    return gap_analysis.questions_for_gaps
