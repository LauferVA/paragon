# Paragon Reference System

The Reference System provides template-based requirement analysis and gap detection for the Paragon agent framework.

## Overview

The reference system helps the architect agent ask better questions by:

1. **Matching user descriptions to known patterns** - Identifies relevant document templates based on keywords
2. **Detecting specification gaps** - Compares user input to templates to find missing information
3. **Generating targeted questions** - Provides questions derived from templates to fill gaps

## Architecture

```
reference/
├── __init__.py              # Package exports
├── matcher.py               # Template matching logic
├── question_bank.py         # Question repository
└── templates/
    ├── generic/             # Universal templates
    │   ├── prd_template.md
    │   ├── tech_spec_template.md
    │   ├── api_design_template.md
    │   └── user_story_template.md
    └── domain/              # Domain-specific templates
        ├── web_app.md
        ├── cli_tool.md
        ├── api_service.md
        ├── data_pipeline.md
        └── ml_system.md
```

## Components

### 1. Templates

Templates are markdown files with structured sections. Each section contains:
- **Purpose**: What the section is for
- **Questions if missing**: Questions to ask if this section is absent
- **Example**: Sample content

Templates come in two types:
- **Generic**: Universal templates (PRD, tech spec, API design, user stories)
- **Domain-specific**: Tailored to specific domains (web apps, CLIs, APIs, data pipelines, ML)

### 2. Template Matcher (`matcher.py`)

Matches user descriptions to relevant templates using keyword-based matching.

```python
from reference import TemplateMatcher

matcher = TemplateMatcher()

# Find relevant templates
matches = matcher.match_templates("Build a REST API for user management")

for match in matches:
    print(f"{match.template_name}: {match.relevance_score:.2f}")
    print(f"  Matched keywords: {', '.join(match.matched_keywords)}")
```

**Key Classes:**
- `TemplateMatcher`: Main matching engine
- `MatchedTemplate`: A matched template with relevance score
- `GapAnalysis`: Result of comparing a spec to a template

**Key Methods:**
- `match_templates(description)`: Find templates matching a description
- `analyze_gaps(spec_content, template_path)`: Compare spec to template
- `get_questions_for_gaps(spec_content, template_match)`: Get questions for missing sections

### 3. Question Bank (`question_bank.py`)

Central repository of questions derived from templates.

```python
from reference import QuestionBank

bank = QuestionBank()

# Get domain-specific questions
web_questions = bank.get_domain_questions("web_app", min_priority="high")

# Get generic questions
generic = bank.get_generic_questions(category="Architecture")

# Get questions to fill gaps
gaps = bank.get_gap_questions(
    template="prd_template",
    spec_content="We need to build a web app...",
)
```

**Key Classes:**
- `QuestionBank`: Question repository
- `Question`: A single question with metadata
- `QuestionSet`: Group of related questions

**Key Methods:**
- `get_generic_questions()`: Get universal questions
- `get_domain_questions(domain)`: Get domain-specific questions
- `get_gap_questions(template, spec_content)`: Get questions for gaps
- `search_questions(query)`: Search questions by keyword

## Usage Examples

### Example 1: Match Templates to Project Description

```python
from reference import TemplateMatcher

matcher = TemplateMatcher()

description = """
Build a web application with React for visualizing service dependencies.
Users can browse services, see dependency graphs, and check for conflicts.
"""

matches = matcher.match_templates(description, max_results=3)

print("Relevant templates:")
for match in matches:
    print(f"- {match.template_name} ({match.relevance_score:.2%})")
    # web_app (85%)
    # tech_spec_template (42%)
    # prd_template (35%)
```

### Example 2: Analyze Gaps in a Specification

```python
from reference import TemplateMatcher

matcher = TemplateMatcher()

spec_content = """
## Overview
A web app for dependency visualization.

## Features
- View dependency graph
- Check for conflicts
"""

gap_analysis = matcher.analyze_gaps(
    spec_content=spec_content,
    template_path="/path/to/templates/generic/prd_template.md",
)

print(f"Coverage: {gap_analysis.coverage_score:.0%}")
print(f"Missing sections: {', '.join(gap_analysis.missing_sections)}")
print(f"Questions to ask: {len(gap_analysis.questions_for_gaps)}")
```

### Example 3: Get Questions for a Domain

```python
from reference import QuestionBank

bank = QuestionBank()

# Get all high-priority questions for web apps
questions = bank.get_domain_questions(
    domain="web_app",
    min_priority="high",
)

print(f"Found {len(questions)} high-priority web app questions:")
for q in questions[:5]:
    print(f"- [{q.category}] {q.text}")
```

### Example 4: Fill Gaps in User Requirements

```python
from reference import TemplateMatcher, QuestionBank

# User provides initial description
user_input = "Build a CLI tool for checking dependencies"

# Step 1: Match templates
matcher = TemplateMatcher()
matches = matcher.match_templates(user_input)

if matches:
    top_match = matches[0]
    print(f"Detected project type: {top_match.template_name}")

    # Step 2: Get questions for gaps
    bank = QuestionBank()
    questions = bank.get_gap_questions(
        template=top_match.template_name,
        spec_content=user_input,
        template_type=top_match.template_type,
        min_priority="medium",
    )

    # Step 3: Ask questions
    print(f"\nI have {len(questions)} questions:")
    for q in questions[:10]:
        print(f"- {q.text}")
```

## Template Format

Templates use a standard markdown format:

```markdown
# Template Title

## Section Name
**Purpose:** What this section is for.

**Questions if missing:**
- Question 1?
- Question 2?
- Question 3?

**Example:**
> Example content here
```

**Important:**
- Questions must end with `?`
- Use `**Questions if missing:**` or `**Questions to ask:**` headers
- Each question should be a bullet point (`-`)
- Examples should be in blockquotes (`>`)

## Extending the System

### Adding a New Template

1. Create a markdown file in `templates/generic/` or `templates/domain/`
2. Follow the standard template format (see above)
3. Add keywords to the matcher (if creating a new domain template)

Example: Adding a "mobile_app" domain template

```python
# In matcher.py, add to _domain_keywords:
"mobile_app": [
    "mobile app", "ios", "android", "react native",
    "flutter", "swift", "kotlin", "mobile"
]
```

### Adding Custom Question Categories

Questions are automatically categorized by template section titles. To add custom categories:

1. Use meaningful section titles in your template
2. The section title becomes the question category
3. Priority is auto-detected from section title keywords

## Integration with Socratic Engine

The reference system complements the existing Socratic Engine (`requirements/socratic_engine.py`):

**Socratic Engine:**
- Provides L1/L2/L3 question hierarchy
- Canonical questions with ask_when conditions
- Gap analysis based on heuristics

**Reference System:**
- Template-based question generation
- Domain-specific knowledge
- Coverage analysis by template section

**Integration approach:**
```python
from requirements.socratic_engine import SocraticEngine
from reference import TemplateMatcher, QuestionBank

# Use Socratic Engine for structured Q&A flow
socratic = SocraticEngine()
session = socratic.create_session(req_id="REQ-123")

# Use Reference System for additional domain questions
bank = QuestionBank()
domain_questions = bank.get_domain_questions("web_app", min_priority="high")

# Combine: Ask Socratic L1 questions first, then domain questions
```

## Testing

Test the reference system:

```bash
# From paragon root directory
python -m pytest tests/test_reference.py -v
```

## Performance Considerations

**Caching:**
- Template questions are cached after first load
- Reuse `TemplateMatcher` and `QuestionBank` instances

**Scalability:**
- Keyword matching is O(n) where n = number of keywords
- Template parsing is done lazily (on first access)
- Consider using embeddings for semantic matching in future versions

## Future Enhancements

Potential improvements:

1. **Semantic Matching**: Use embeddings instead of keyword matching
2. **Question Dependencies**: Implement follow-up question chains
3. **Custom Templates**: Allow users to add their own templates
4. **Multi-language Support**: Templates in different languages
5. **LLM Integration**: Use LLM to generate questions from templates dynamically
6. **Version Control**: Track template versions and changes over time

## Design Philosophy

The reference system follows Paragon's core principles:

1. **No Pydantic**: Uses `msgspec.Struct` for all data schemas
2. **Extensibility**: Easy to add new templates and domains
3. **Performance**: Keyword-based matching for speed (no LLM calls)
4. **Clarity**: Human-readable markdown templates
5. **Integration**: Designed to work with existing Socratic Engine

## File Summary

### Templates Created

**Generic (4 templates):**
- `prd_template.md` - Product Requirements Document (10 sections, 40+ questions)
- `tech_spec_template.md` - Technical Specification (13 sections, 65+ questions)
- `api_design_template.md` - API Design (13 sections, 55+ questions)
- `user_story_template.md` - User Story (11 sections, 45+ questions)

**Domain-specific (5 templates):**
- `web_app.md` - Web Application (15 sections, 75+ questions)
- `cli_tool.md` - Command-Line Tool (17 sections, 85+ questions)
- `api_service.md` - API/Backend Service (19 sections, 95+ questions)
- `data_pipeline.md` - Data Pipeline (17 sections, 85+ questions)
- `ml_system.md` - Machine Learning System (18 sections, 90+ questions)

**Total: 9 templates, 535+ questions**

### Code Modules

- `__init__.py` - Package initialization and exports
- `matcher.py` - Template matching engine (350 lines)
- `question_bank.py` - Question repository (300 lines)
- `README.md` - Documentation (this file)

### Data Structures

**msgspec.Struct classes:**
- `MatchedTemplate` - Template match result
- `TemplateSection` - Section within a template
- `GapAnalysis` - Gap analysis result
- `Question` - Individual question
- `QuestionSet` - Group of related questions

## License

Part of the Paragon project. See main LICENSE file.
