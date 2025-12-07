"""
PARAGON REFERENCE SYSTEM

Template-based requirement analysis and gap detection.

This module provides:
- Generic and domain-specific templates for specifications
- Template matching to identify relevant patterns
- Question banks derived from templates
- Gap analysis comparing specs to templates

Components:
- templates/: Markdown templates for different document types
- matcher.py: Template matching logic
- question_bank.py: Question generation from templates
"""

from .matcher import TemplateMatcher, MatchedTemplate
from .question_bank import QuestionBank, Question

__all__ = [
    "TemplateMatcher",
    "MatchedTemplate",
    "QuestionBank",
    "Question",
]
