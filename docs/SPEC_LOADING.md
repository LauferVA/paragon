# Spec Loading and Initial Conversation

This document describes the spec loading and initial conversation features in Paragon.

## Overview

Paragon supports two startup modes:

1. **Fresh Start**: Interactive conversation to understand your project
2. **Spec File**: Load a specification file and start immediately

## Fresh Start Mode

Start Paragon without a spec file to begin an interactive conversation:

```bash
python main.py --dev
```

The system will:
1. Show a welcome message
2. Ask standard questions about your project:
   - "What do you want to build today?"
   - "Who is the target user?"
   - "What are the must-have features?"
   - "Any technical constraints?"
   - "How will you know when this is successful?"
3. Build a structured specification from your answers
4. Create REQ nodes in the graph
5. Start the research phase

## Spec File Mode

Start Paragon with a spec file to skip the conversation:

```bash
python main.py --dev --spec path/to/spec.md
```

### Supported Formats

#### Markdown (.md)

Unstructured markdown files are parsed using LLM to extract:
- Project title
- Description
- Requirements
- Technical details
- Target user
- Features
- Constraints

Example: `/Users/lauferva/paragon/examples/sample_spec.md`

#### Plain Text (.txt)

Similar to markdown, parsed using LLM.

#### JSON (.json)

Structured JSON format with explicit fields:

```json
{
  "title": "Project Name",
  "description": "Project description",
  "requirements": ["req1", "req2"],
  "must_have_features": ["feature1", "feature2"],
  "technical_details": "Stack, architecture, etc.",
  "target_user": "Who will use this",
  "constraints": ["constraint1", "constraint2"]
}
```

Example: `/Users/lauferva/paragon/examples/sample_spec.json`

## Phase Determination

When loading a spec file, Paragon intelligently determines where to start:

- **Plan Phase**: Detailed spec with 3+ requirements and technical details
- **Research Phase**: Some requirements but missing details
- **Dialectic Phase**: Minimal information, needs ambiguity checking

## Implementation Details

### Key Files

1. `/Users/lauferva/paragon/agents/spec_parser.py`
   - `SpecParser` class for parsing various formats
   - `parse_spec_file()` convenience function
   - LLM-based extraction for unstructured text
   - Fallback parser for when LLM is unavailable

2. `/Users/lauferva/paragon/agents/initial_conversation.py`
   - `InitialConversation` class for fresh starts
   - Standard question bank
   - Answer recording and spec building
   - Graph persistence for REQ nodes

3. `/Users/lauferva/paragon/main.py`
   - `cmd_start_session()` handles both modes
   - Parses spec if provided
   - Shows appropriate greeting/message
   - Initializes orchestrator with correct phase

4. `/Users/lauferva/paragon/agents/orchestrator.py`
   - Updated `TDDOrchestrator.run()` to accept `initial_phase`
   - Skips init/dialectic if starting at a later phase
   - Pre-populates state appropriately

### Schemas

`ParsedSpec` (in `/Users/lauferva/paragon/agents/schemas.py`):
- `title: str`
- `description: str`
- `requirements: List[str]`
- `technical_details: Optional[str]`
- `target_user: Optional[str]`
- `must_have_features: List[str]`
- `constraints: List[str]`
- `raw_content: str`
- `file_format: str`

## Usage Examples

### Example 1: Fresh Start

```bash
python main.py --dev
```

Output:
```
Welcome to Paragon! I'm here to help you build high-quality software.

What would you like to build today?

(You can describe your project idea, or I can guide you through some questions to understand what you need.)
```

### Example 2: Load Markdown Spec

```bash
python main.py --dev --spec examples/sample_spec.md
```

Output:
```
Loading spec from: examples/sample_spec.md
Loaded spec: Fibonacci Calculator
Description: A simple web API that calculates Fibonacci numbers efficiently...
Starting phase: research

I've loaded your specification for 'Fibonacci Calculator'. I have 5 requirement(s). Let me conduct research to create a detailed implementation plan...
```

### Example 3: Load JSON Spec

```bash
python main.py --dev --spec examples/sample_spec.json
```

Output:
```
Loading spec from: examples/sample_spec.json
Loaded spec: Fibonacci Calculator API
Description: A simple web API that calculates Fibonacci numbers efficiently...
Starting phase: plan

I've loaded your detailed specification for 'Fibonacci Calculator API'. I have all the information I need. Let me create an implementation plan...
```

## Integration with Orchestrator

The orchestrator receives:
- `session_id`: Unique session ID
- `spec`: Full specification text (formatted)
- `requirements`: Extracted requirements list
- `initial_phase`: Where to start ("dialectic", "research", or "plan")

This allows the orchestrator to:
1. Skip irrelevant phases
2. Start with appropriate context
3. Avoid asking redundant questions
4. Begin work immediately if spec is complete

## Error Handling

- **File not found**: Clear error message, exits with code 1
- **Invalid format**: Unsupported file extension error
- **Parse failure**: Falls back to basic parsing
- **LLM unavailable**: Uses simple heuristic parser

## Future Enhancements

- Support for YAML specs
- Support for PRD templates
- Spec validation and completeness scoring
- Interactive spec refinement
- Multi-file spec support (references between files)
- Spec versioning and evolution tracking
