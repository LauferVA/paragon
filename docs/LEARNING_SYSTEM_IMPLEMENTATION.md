# Learning System Implementation Summary

**Date:** 2025-12-06
**Status:** Production Ready
**Branch:** G - Learning System (L4/L5) + Adaptive Questioner

---

## Overview

This document summarizes the implementation of Paragon's learning system components, which enable the system to improve its decision-making over time through data collection and pattern analysis.

---

## Components Implemented

### 1. Learning Manager (`infrastructure/learning.py`)

**Purpose:** Manages the two-mode learning system and applies learned patterns to optimize model selection.

**Key Features:**
- **Two-Mode Operation:**
  - STUDY Mode: Collects clean data without biasing decisions
  - PRODUCTION Mode: Uses learned patterns to optimize model routing

- **Model Recommendations:**
  - Tracks success rates per model and phase
  - Provides confidence-weighted recommendations
  - Implements epsilon-greedy exploration (10% random in production)

- **Transition Management:**
  - Requires 100 sessions minimum for production transition
  - Tracks success rates and divergence rates
  - Provides human-readable transition reports

- **Statistics Tracking:**
  - Session outcomes (success/failure)
  - Model performance per phase
  - Divergence detection integration

**Reference:** `docs/IMPLEMENTATION_PLAN_LEARNING.md` Sections 5, 9

### 2. Adaptive Questioner (`agents/adaptive_questioner.py`)

**Purpose:** Optimizes question-asking strategy based on learned user behavior patterns.

**Key Features:**
- **Question Prioritization:**
  - Ranks questions by expected information gain
  - Considers user priorities (speed, cost, control)
  - Respects max question limits

- **Suggested Answers:**
  - Uses historical acceptance rates
  - Returns suggestions only when confidence is high (≥70%)
  - Falls back to most common answers

- **Learning from Patterns:**
  - Tracks skip probabilities per ambiguity category
  - Records suggestion acceptance rates
  - Updates based on session outcomes

- **User Priority Integration:**
  - Speed optimization: fewer questions, penalize high skip probability
  - Cost optimization: boost questions with high-confidence suggestions
  - Control optimization: keep more questions

**Reference:** `docs/RESEARCH_ADAPTIVE_QUESTIONING.md`

---

## Database Schema Extensions

The learning system adds the following tables to the training database:

### Question Tracking Tables

```sql
-- Question attempts
CREATE TABLE question_attempts (
    question_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ambiguity_category TEXT NOT NULL,
    question_text TEXT NOT NULL,
    suggested_answer TEXT,
    user_answer TEXT,
    was_answered INTEGER NOT NULL,
    used_suggestion INTEGER DEFAULT 0,
    answer_quality_score REAL DEFAULT 0.5,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Question outcomes (linked to session success)
CREATE TABLE question_outcomes (
    question_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    led_to_success INTEGER,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## Test Coverage

### Learning Manager Tests (`tests/test_learning.py`)

**Coverage:** 21 tests passing

- Learning mode basics (STUDY vs PRODUCTION)
- Model recommendations with historical data
- Outcome recording (success, failure, with stats)
- Divergence detection integration
- Learning statistics generation
- Transition logic (readiness assessment)
- Model performance summaries

### Adaptive Questioner Tests (`tests/test_adaptive_questioner.py`)

**Coverage:** 23 tests passing

- Questioner initialization
- Question prioritization logic
- Skip probability calculation
- Suggested answer confidence
- Question outcome recording
- Historical pattern learning
- Question statistics
- User priority integration
- Priority score calculation

**Total:** 44 tests, 100% passing

---

## Demo Application (`examples/learning_demo.py`)

A comprehensive demonstration showing:

1. **Study Mode Operation**
   - Collects 25 initial sessions
   - Shows no recommendations in STUDY mode
   - Displays learning statistics

2. **Transition Analysis**
   - Checks readiness for production
   - Simulates additional sessions to reach 100
   - Performs transition to PRODUCTION mode

3. **Production Mode**
   - Shows model performance summary
   - Demonstrates model recommendations
   - Displays confidence-weighted decisions

4. **Adaptive Questioning**
   - Prioritizes questions by information gain
   - Applies user priorities (speed optimization)
   - Records and learns from 20 question interactions
   - Shows category-specific statistics

### Sample Output

```
PRODUCTION MODE - Model Performance Summary:

  claude-opus-4-5-20251101:
    dialectic: 100.0% success (10 samples)
    build: 100.0% success (8 samples)
    test: 87.5% success (8 samples)

  claude-sonnet-4-5-20250929:
    plan: 88.9% success (9 samples)
    build: 87.5% success (8 samples)

  BUILD:
    Recommended Model: claude-opus-4-5-20251101
    Success Rate: 100.0%
    Confidence: 44.4%
    Reasoning: Selected for build phase based on 8 historical samples
```

---

## Integration Points

### With Existing Infrastructure

1. **TrainingStore** (`infrastructure/training_store.py`)
   - Uses existing attribution and session outcome tables
   - Extends with question tracking tables
   - Shares database connection

2. **ForensicAnalyzer** (`infrastructure/attribution.py`)
   - LearningManager uses analyzer for failure attribution
   - Links question outcomes to session success

3. **DivergenceDetector** (`infrastructure/divergence.py`)
   - Integrated via `check_and_log_divergence()`
   - Affects transition readiness calculations

4. **Schemas** (`agents/schemas.py`)
   - Uses existing: `CyclePhase`, `FailureCode`, `NodeOutcome`, `AgentSignature`
   - Defines new: `AmbiguityMarker` (used by questioner)

---

## Usage Examples

### Learning Manager

```python
from infrastructure.learning import LearningManager, LearningMode

# Initialize in STUDY mode
manager = LearningManager(mode=LearningMode.STUDY)

# Record session outcomes
manager.record_outcome(
    session_id="session_1",
    success=True,
    stats={"total_nodes": 42, "total_tokens": 5000}
)

# Check transition readiness
report = manager.should_transition_to_production()
print(f"Ready: {report.ready}, Sessions: {report.session_count}")

# Transition to production
if report.ready:
    manager.transition_to_production()

# Get model recommendations (only in PRODUCTION mode)
recommendation = manager.get_model_recommendation(phase=CyclePhase.BUILD)
if recommendation:
    print(f"Use model: {recommendation.model_id}")
```

### Adaptive Questioner

```python
from agents.adaptive_questioner import AdaptiveQuestioner, UserPriorities

# Initialize questioner
questioner = AdaptiveQuestioner()

# Prioritize questions
priorities = UserPriorities(
    speed_weight=0.8,  # Optimize for speed
    max_clarification_questions=3
)

prioritized = questioner.prioritize_questions(
    ambiguities=detected_ambiguities,
    priorities=priorities
)

# Record question outcome
question_id = questioner.record_question_outcome(
    session_id="session_1",
    ambiguity=ambiguities[0],
    was_answered=True,
    user_answer="PostgreSQL",
    used_suggestion=True,
    answer_quality_score=1.0
)

# Update after session completes
questioner.update_question_outcome(
    question_id=question_id,
    session_id="session_1",
    led_to_success=True
)

# Get statistics
stats = questioner.get_question_stats(category="UNDEFINED_TERM")
print(f"Skip rate: {stats['skip_rate'] * 100:.1f}%")
```

---

## Key Design Decisions

### 1. Two-Mode Separation

**Rationale:** Clean separation between data collection (STUDY) and optimization (PRODUCTION) prevents biasing the training dataset.

**Implementation:** Mode is checked before every recommendation. STUDY mode always returns `None`.

### 2. Epsilon-Greedy Exploration

**Rationale:** Even in PRODUCTION mode, continue exploring (10% of the time) to discover new patterns and avoid getting stuck in local optima.

**Implementation:** Random 10% chance to return `None` even in PRODUCTION mode.

### 3. Confidence from Sample Size

**Rationale:** More samples = higher confidence, but with diminishing returns.

**Implementation:** Confidence = `min(0.9, sample_count / (sample_count + 10))`

### 4. Question Priority Score Calculation

**Rationale:** Balance information gain with user preferences.

**Implementation:**
- Base score = information gain
- Adjust based on user priorities (speed, cost, control)
- Blocking questions get 20% boost
- Cap at 1.0

### 5. Suggested Answer Threshold

**Rationale:** Only suggest answers when confidence is high enough to be useful.

**Implementation:** Require ≥70% historical acceptance rate before returning suggestions.

---

## Performance Characteristics

### Learning Manager

- **Database Operations:** O(1) per session recording
- **Recommendation Query:** O(M) where M = number of models (typically 4)
- **Memory:** Minimal - only active session data in memory

### Adaptive Questioner

- **Prioritization:** O(N log N) where N = number of ambiguities
- **Statistics Query:** O(1) per category (indexed)
- **Memory:** Minimal - only question data in memory

---

## Next Steps

### Recommended Enhancements

1. **Constraint-Based Filtering** (Future)
   - Currently planned but not implemented
   - Would allow filtering by token limits, cost caps, etc.

2. **Statistical Significance Testing** (Future)
   - Per IMPLEMENTATION_PLAN_LEARNING.md Section 2.2
   - Require p < 0.05 on at least 3 constraint dimensions

3. **GUI Integration** (Future)
   - Show learning stats in development mode
   - Allow manual review of learned patterns before transition

4. **A/B Testing Framework** (Future)
   - Test different question strategies
   - Compare STUDY vs PRODUCTION performance

---

## References

- [IMPLEMENTATION_PLAN_LEARNING.md](IMPLEMENTATION_PLAN_LEARNING.md) - Full learning system architecture
- [RESEARCH_ADAPTIVE_QUESTIONING.md](RESEARCH_ADAPTIVE_QUESTIONING.md) - Question optimization research
- [CLAUDE.md](../CLAUDE.md#6-learning-system-architecture) - Parent specification

---

## Files Created

### Production Code
- `/infrastructure/learning.py` - Learning manager (421 lines)
- `/agents/adaptive_questioner.py` - Adaptive questioner (531 lines)

### Tests
- `/tests/test_learning.py` - Learning manager tests (21 tests)
- `/tests/test_adaptive_questioner.py` - Questioner tests (23 tests)

### Documentation & Examples
- `/examples/learning_demo.py` - Full system demonstration (372 lines)
- `/docs/LEARNING_SYSTEM_IMPLEMENTATION.md` - This document

**Total:** ~1,745 lines of production code + tests + documentation
