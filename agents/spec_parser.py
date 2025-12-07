"""
SPEC PARSER - Parse various spec file formats

Handles different spec file formats and extracts structured information:
- Markdown files (.md)
- Plain text files (.txt)
- JSON specs (.json)

Uses LLM to extract structure from unstructured text.
"""
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import msgspec

from core.llm import get_llm, StructuredLLM
from agents.schemas import ParsedSpec

logger = logging.getLogger(__name__)


# =============================================================================
# SPEC PARSER
# =============================================================================

class SpecParser:
    """
    Parser for various spec file formats.

    Handles:
    - Markdown (.md)
    - Plain text (.txt)
    - JSON (.json)

    Uses LLM to extract structured information from unstructured text.
    """

    def __init__(self, llm: Optional[StructuredLLM] = None):
        """
        Initialize the spec parser.

        Args:
            llm: Optional LLM instance (will use default if not provided)
        """
        self.llm = llm

    def _get_llm(self) -> StructuredLLM:
        """Get LLM instance, creating if needed."""
        if self.llm is None:
            self.llm = get_llm()
        return self.llm

    def parse_spec(self, filepath: str) -> ParsedSpec:
        """
        Parse a spec file and extract structured information.

        Args:
            filepath: Path to spec file (relative or absolute)

        Returns:
            ParsedSpec with extracted information

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(filepath)

        # Resolve to absolute path
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {filepath}")

        # Read file content
        try:
            content = path.read_text(encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read spec file: {e}")

        # Determine format
        suffix = path.suffix.lower()

        if suffix == '.json':
            return self._parse_json(content, suffix)
        elif suffix in ['.md', '.txt', '']:
            return self._parse_text(content, suffix or '.txt')
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _parse_json(self, content: str, file_format: str) -> ParsedSpec:
        """
        Parse JSON spec file.

        Expected format:
        {
          "title": "Project Name",
          "description": "...",
          "requirements": ["req1", "req2"],
          "technical_details": "...",
          "target_user": "...",
          "must_have_features": ["feature1", "feature2"],
          "constraints": ["constraint1"]
        }
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        return ParsedSpec(
            title=data.get('title', 'Untitled Project'),
            description=data.get('description', ''),
            requirements=data.get('requirements', []),
            technical_details=data.get('technical_details'),
            target_user=data.get('target_user'),
            must_have_features=data.get('must_have_features', []),
            constraints=data.get('constraints', []),
            raw_content=content,
            file_format=file_format,
        )

    def _parse_text(self, content: str, file_format: str) -> ParsedSpec:
        """
        Parse text/markdown spec file using LLM.

        Uses LLM to extract structured information from unstructured text.
        """
        llm = self._get_llm()

        system_prompt = """You are a technical specification parser. Extract structured information from the provided spec document.

Your task is to analyze the document and extract:
1. Title/name of the project
2. Overall description/overview
3. List of requirements (functional and non-functional)
4. Technical details (architecture, technologies, etc.)
5. Target user/audience
6. Must-have features
7. Constraints (technical, business, timeline)

Be thorough but concise. Extract all relevant information."""

        user_prompt = f"""# Specification Document

{content}

Parse this specification and extract structured information. If certain fields are not present in the document, leave them empty."""

        try:
            result = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=ParsedSpec,
            )

            # Add raw content and format
            # Note: msgspec Structs are immutable when frozen, so we need to create a new instance
            return ParsedSpec(
                title=result.title,
                description=result.description,
                requirements=result.requirements,
                technical_details=result.technical_details,
                target_user=result.target_user,
                must_have_features=result.must_have_features,
                constraints=result.constraints,
                raw_content=content,
                file_format=file_format,
            )
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}, using fallback")
            # Fallback: basic parsing
            return self._fallback_parse(content, file_format)

    def _fallback_parse(self, content: str, file_format: str) -> ParsedSpec:
        """
        Fallback parser when LLM is unavailable.

        Uses simple heuristics to extract information.
        """
        lines = content.split('\n')

        # Try to extract title from first non-empty line or first heading
        title = "Untitled Project"
        description_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check for markdown heading
            if stripped.startswith('# '):
                title = stripped[2:].strip()
                break
            elif not title or title == "Untitled Project":
                # Use first non-empty line as title
                title = stripped
                break

        # Use first paragraph as description
        in_description = False
        for line in lines:
            stripped = line.strip()

            if not in_description and stripped and not stripped.startswith('#'):
                in_description = True

            if in_description:
                if not stripped and description_lines:
                    break
                if stripped and not stripped.startswith('#'):
                    description_lines.append(stripped)

        description = ' '.join(description_lines)

        return ParsedSpec(
            title=title,
            description=description or "No description provided",
            requirements=[],
            technical_details=None,
            target_user=None,
            must_have_features=[],
            constraints=[],
            raw_content=content,
            file_format=file_format,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_spec_file(filepath: str, llm: Optional[StructuredLLM] = None) -> ParsedSpec:
    """
    Convenience function to parse a spec file.

    Args:
        filepath: Path to spec file
        llm: Optional LLM instance

    Returns:
        ParsedSpec with extracted information
    """
    parser = SpecParser(llm=llm)
    return parser.parse_spec(filepath)
