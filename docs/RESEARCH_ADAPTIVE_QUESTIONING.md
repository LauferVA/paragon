# RESEARCH REPORT: Adaptive Question Learning for Paragon Dialectic System

**Date:** 2025-12-06
**Version:** 1.0
**Status:** Research Phase - No Implementation

---

## EXECUTIVE SUMMARY

This research investigates how to optimize the question-asking strategy in Paragon's dialectic phase to maximize widget (generated codebase) quality while respecting user experience and cost constraints.

**Core Problem:** What questions should Paragon ask to achieve the best outcomes?

**Key Finding:** Widget quality must serve as a **hard constraint floor**, with all other optimizations (speed, cost, satisfaction) occurring above that minimum threshold.

---

## 1. QUALITY FLOOR DEFINITION

### 1.1 The Primacy Principle

> **Widget quality metrics have PRIMACY. They are hard constraints, not tradeoffs.**

Any adaptive learning system that trades quality for speed/cost will degrade over time, rendering all other metrics useless.

### 1.2 Quality Metrics (Hard Constraints)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Test Pass Rate** | 100% | ALL tests must pass |
| **Static Analysis** | 0 critical issues | OWASP Top 10, CWE Top 25 |
| **Graph Invariants** | 100% compliance | Teleological integrity, no orphans |
| **Cyclomatic Complexity** | ≤ 15 per function | Maintainability |
| **Code Duplication** | ≤ 5% | DRY principle |
| **Security Posture** | No hardcoded secrets | Entropy analysis |

### 1.3 Implementation: Quality Gate

**New Module:** `agents/quality_gate.py`

```python
class QualityGate:
    def check_quality_floor(self, widget_nodes) -> QualityReport:
        # 1. Test pass rate
        # 2. Static analysis (ruff, mypy, bandit)
        # 3. Graph invariants
        # 4. Code quality metrics
        # 5. Security patterns
        return QualityReport(passed=..., violations=...)
```

**Integration:** If `passed=False`, orchestrator MUST NOT proceed to PASSED state.

---

## 2. USER PRIORITY SYSTEM

### 2.1 When to Ask About Priorities

**Timing:** Immediately after DIALECTIC phase, before RESEARCH

Rationale:
- User has seen their ambiguous prompt analyzed
- Low exit cost (haven't invested much time)
- Priorities can shape research depth

### 2.2 Priority Dimensions

1. **Speed vs. Thoroughness** - Fewer questions vs. deeper research
2. **Cost vs. Polish** - Cheaper models vs. best models
3. **Autonomy vs. Control** - Auto-proceed vs. ask everything
4. **Experimentation vs. Production** - Lower floor vs. full quality

### 2.3 Schema Addition

```python
class UserPriorities(msgspec.Struct, kw_only=True, frozen=True):
    speed_weight: float = 0.33
    cost_weight: float = 0.33
    control_weight: float = 0.34
    quality_mode: Literal["production", "experimental"] = "production"
    max_clarification_questions: int = 5
    auto_proceed_confidence: float = 0.85
```

### 2.4 How Priorities Influence Questions

1. Compute Expected Information Gain (EIG) for each question
2. Rank by EIG (highest first)
3. Apply priority filters:
   - `speed_weight > 0.6` → Keep only top 3 questions
   - `cost_weight > 0.6` → Skip web search, use LLM defaults
   - `control_weight > 0.6` → Ask all questions
4. Enforce `max_clarification_questions` limit

---

## 3. LEARNING STRATEGY

### 3.1 Cold Start Problem

Paragon has **ZERO historical data** initially. Pure supervised learning is not viable.

### 3.2 Recommended Approach: Contextual Bandits + RLHF

#### Phase 1: Cold Start (First 100 Sessions)
- **Strategy:** Thompson Sampling with Contextual Features
- **Features:** Ambiguity category, user domain, question complexity
- **Arms:** Ask with suggestion, ask open-ended, skip
- **Immediate Reward:** User response quality (+1 substantive, -0.5 "don't care")
- **Delayed Reward:** Widget passed quality floor (+10) or failed (-10)

#### Phase 2: Hybrid Learning (Sessions 100-1000)
- Add lightweight supervised model (BERT for ambiguity classification)
- Use model predictions to initialize Thompson Sampling priors

#### Phase 3: Full RLHF (Sessions 1000+)
- Policy gradient with human preference feedback
- Based on RLTHF framework (6-7% human effort for full alignment)

### 3.3 New Modules Needed

```
agents/adaptive_questioner.py
  - QuestionBandit: Contextual bandit for question selection
  - ContextExtractor: Extracts features from prompts
  - RewardCalculator: Computes rewards from outcomes

infrastructure/learning_store.py
  - QuestionAttempt: Stores (context, question, answer, outcome)
  - SessionOutcome: Stores widget quality metrics
  - BanditState: Persists Thompson Sampling parameters

infrastructure/rlhf_trainer.py (Phase 3 only)
  - RewardModel: Scores question quality
  - PolicyUpdater: Updates question generation strategy
```

---

## 4. METRICS FRAMEWORK

### 4.1 Question-Level Metrics

| Metric | Measurement | Target |
|--------|-------------|--------|
| User Response Quality | Substantive (1.0), Brief (0.5), Skip (0.0) | > 0.7 avg |
| Information Gain | Entropy reduction in implementation space | > 0.5 bits |
| Time to Answer | Seconds | < 60s |
| Answer Consistency | Semantic similarity to prior answers | > 0.8 |

### 4.2 Session-Level Metrics

| Category | Metrics |
|----------|---------|
| **Widget Quality** | Test pass rate, static analysis score, iterations to pass |
| **User Experience** | Completion rate (>90%), CSAT (1-5), NPS (0-10) |
| **Efficiency** | Question count, token cost (<$0.50), time to widget (<10min) |

### 4.3 Storage Schema

```sql
CREATE TABLE question_attempts (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    ambiguity_category TEXT,
    question_text TEXT,
    suggested_answer TEXT,
    user_answer TEXT,
    answer_quality_score REAL,
    information_gain_bits REAL
);

CREATE TABLE session_outcomes (
    session_id TEXT PRIMARY KEY,
    test_pass_rate REAL,
    iterations_to_pass INTEGER,
    token_cost_usd REAL,
    user_csat INTEGER
);
```

---

## 5. PREDICTIVE MODEL SKETCH

### 5.1 Features

**Prompt Features:**
- Length, domain, technical density, vagueness score

**Ambiguity Features:**
- Category, impact, has suggested answer, position

**Historical Features:**
- User's avg answer quality, completion rate, preferred weights

### 5.2 Model Architecture

| Phase | Model | Purpose |
|-------|-------|---------|
| Phase 1 | Logistic Regression | Bootstrap bandit priors |
| Phase 2 | XGBoost | Better contextual understanding |
| Phase 3 | BERT → MLP | Neural reward model |

---

## 6. ANTI-GAMING SAFEGUARDS

| Safeguard | Implementation |
|-----------|---------------|
| Multi-Metric Optimization | Quality floor is CONSTRAINT, not objective |
| Human-in-the-Loop Audits | 5% random sample for manual review |
| Adversarial Testing | Honeypot prompts (known-ambiguous) |
| Bounded Rewards | Clip rewards to prevent runaway optimization |
| Temporal Consistency | Alert if question count drops >30% week-over-week |
| Regularization | Entropy bonus to encourage exploration |

---

## 7. INTEGRATION PLAN

### 7.1 Files to Modify

| File | Changes |
|------|---------|
| `agents/orchestrator.py` | Add `priority_elicitation_node()`, integrate `QuestionBandit` |
| `agents/schemas.py` | Add `UserPriorities`, `QuestionAttempt`, `QualityReport` |
| `config/paragon.toml` | Add `[learning]` and `[quality_floor]` sections |

### 7.2 New Modules

| Module | Purpose |
|--------|---------|
| `agents/quality_gate.py` | Quality floor enforcement |
| `agents/adaptive_questioner.py` | Bandit-based question selection |
| `infrastructure/learning_store.py` | SQLite storage for learning data |
| `infrastructure/rlhf_trainer.py` | RLHF training (Phase 3) |

### 7.3 A/B Testing

- **Control:** Current static question strategy
- **Treatment A:** Bandit with speed optimization
- **Treatment B:** Bandit with control optimization
- **Treatment C:** Bandit with balanced optimization
- **Sample Size:** ~500 sessions per group

---

## 8. RESEARCH GAPS

### 8.1 Unanswered Questions

1. Is the ambiguity taxonomy complete? (IMPLICIT_ASSUMPTIONS, CONTRADICTIONS?)
2. Optimal question ordering? (Blocking first or easy first?)
3. Suggested answer acceptance rates in practice?
4. Cross-domain generalization of question strategies?

### 8.2 Ethical Considerations

1. **User Manipulation Risk:** Transparency about optimization goals
2. **Bias Amplification:** Stratified sampling by user expertise
3. **Data Privacy:** Local-only learning mode, opt-in for cloud
4. **Informed Consent:** Clear disclosure, opt-out option

---

## 9. IMPLEMENTATION PHASES

| Phase | Goal | Sessions |
|-------|------|----------|
| Phase 0 | Implement quality floor enforcement | - |
| Phase 1 | Add user priority elicitation | - |
| Phase 2 | Deploy contextual bandit with logging | 0-100 |
| Phase 3 | Analyze data, train supervised model | 100-500 |
| Phase 4 | Full RLHF | 1000+ |

---

## SOURCES

### Active Learning
- [Survey on Human-Centered Dialog Systems](https://dl.acm.org/doi/10.1145/3729220)
- [TO-GATE: Trajectory Optimization for Clarifying Questions](https://arxiv.org/html/2506.02827v1)

### RLHF
- [RLTHF: Targeted Human Feedback](https://arxiv.org/abs/2502.13417v2)
- [RLHF 101: Technical Tutorial](https://blog.ml.cmu.edu/2025/06/01/rlhf-101)

### Cold Start
- [Multi-Armed Bandits: Cold Start Problem](https://arxiv.org/abs/2502.01867)

### Code Quality
- [Code Quality in 2025](https://www.qodo.ai/blog/code-quality/)
- [AI-Generated Code Security](https://arxiv.org/html/2508.14727v1)

### Reward Hacking
- [Training on Reward Hacking Induces Reward Hacking](https://alignment.anthropic.com/2025/reward-hacking-ooc/)
