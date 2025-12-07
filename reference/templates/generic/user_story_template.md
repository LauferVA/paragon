# User Story Template

## Story Title
**Format:** As a [user type], I want to [action], so that [benefit].

**Questions if missing:**
- Who is the user?
- What do they want to do?
- Why do they want to do it (what value does it provide)?

**Example:**
> As a backend engineer, I want to check my service's dependencies before deploying, so that I can avoid breaking other services.

---

## User Persona
**Purpose:** Define who this user is.

**Questions if missing:**
- What is their role/job title?
- What is their technical skill level?
- What are their goals and motivations?
- What are their pain points?
- What tools do they currently use?

**Example:**
> **Role:** Senior Backend Engineer
> **Skills:** Expert in Python, Docker, Kubernetes; Comfortable with CLI tools
> **Goals:** Ship features quickly without breaking production
> **Pain Points:** Wasting time on rollbacks; Lack of visibility into service dependencies
> **Tools:** VS Code, GitLab, kubectl, Datadog

---

## Acceptance Criteria
**Purpose:** Define what "done" looks like.

**Format:** Use Given/When/Then scenarios.

**Questions if missing:**
- What are the specific conditions for acceptance?
- What scenarios must be supported?
- What edge cases must be handled?
- What is NOT required (out of scope)?

**Example:**
> **Scenario 1: Check dependencies with no conflicts**
> - **Given** I have a service with up-to-date dependencies
> - **When** I run `paragon-deps check my-service`
> - **Then** I see "No conflicts detected" and exit code 0
>
> **Scenario 2: Check dependencies with conflicts**
> - **Given** I have a service with conflicting dependencies
> - **When** I run `paragon-deps check my-service`
> - **Then** I see a list of conflicts with details and exit code 1
>
> **Out of Scope:**
> - Auto-resolution of conflicts (manual review required)

---

## Workflow / User Journey
**Purpose:** Describe the step-by-step user flow.

**Questions if missing:**
- What triggers this workflow?
- What are the steps the user takes?
- What does the user see at each step?
- What are the decision points?
- What is the expected outcome?

**Example:**
> 1. **Trigger:** Developer finishes coding a service update
> 2. **Action:** Developer runs `paragon-deps check my-service`
> 3. **System:** Analyzes dependencies and checks for conflicts
> 4. **Display:** Shows list of services that depend on mine
> 5. **Decision:** If conflicts found, developer reviews details
> 6. **Action:** Developer decides to proceed or fix conflicts
> 7. **Outcome:** Confident deployment or avoided incident

---

## Input / Output
**Purpose:** Define what goes in and what comes out.

**Questions if missing:**
- What inputs does the user provide?
- What format are the inputs?
- What are the input validation rules?
- What outputs does the user receive?
- What format are the outputs?

**Example:**
> **Input:**
> - Service name (string, required)
> - Optional flags: `--depth=N`, `--format=json|table`
>
> **Output:**
> - Console output showing dependencies and conflicts
> - Exit code: 0 (no conflicts), 1 (conflicts), 2 (error)
> - Optional JSON output for scripting

---

## Success Metrics
**Purpose:** Define how to measure if this story delivers value.

**Questions if missing:**
- How will you measure success?
- What user behavior changes do you expect?
- What metrics will improve?

**Example:**
> - 80% of deployments run dependency check first
> - 50% reduction in deployment rollbacks due to conflicts
> - < 30 seconds to check dependencies (fast enough to run frequently)

---

## Dependencies
**Purpose:** Identify what this story depends on.

**Questions if missing:**
- What other stories must be completed first?
- What external systems/APIs are required?
- What technical infrastructure is needed?

**Example:**
> **Story Dependencies:**
> - Story #42: "Analyze service dependencies" (must be complete)
>
> **System Dependencies:**
> - GitLab API access
> - Service registry with accurate metadata

---

## Technical Notes
**Purpose:** Capture technical details for implementation.

**Questions if missing:**
- What are the key technical considerations?
- What are the performance requirements?
- What are the security considerations?
- What error cases need to be handled?

**Example:**
> - Must analyze graph in < 5 seconds for responsive UX
> - Must handle case where service is not in registry (error message)
> - Must validate service name format before querying
> - Must cache results for 5 minutes to avoid redundant analysis

---

## UI / UX Details
**Purpose:** Specify interface details.

**Questions if missing:**
- What does the user see (screenshots, mockups)?
- What are the interaction patterns?
- What feedback does the user receive?
- What error messages are shown?

**Example:**
> **Console Output:**
> ```
> Checking dependencies for my-service...
>
> Dependencies (3):
>   ✓ auth-service@2.3.1
>   ✓ payment-service@1.5.0
>   ⚠ user-db@3.0.0 (conflict)
>
> Conflicts (1):
>   ⚠ user-db@3.0.0
>     Your service requires: ^3.0.0
>     customer-service requires: ^2.5.0
>     Conflict severity: MEDIUM
>     Suggestion: Coordinate with customer-service team
>
> Exit code: 1
> ```

---

## Test Scenarios
**Purpose:** Define how this will be tested.

**Questions if missing:**
- What are the key test cases?
- What edge cases must be tested?
- What error conditions must be tested?
- What are the performance test scenarios?

**Example:**
> **Test Cases:**
> 1. Service with no dependencies → Shows "No dependencies"
> 2. Service with compatible dependencies → Shows all dependencies, exit 0
> 3. Service with conflicting dependencies → Shows conflicts, exit 1
> 4. Service not in registry → Error message, exit 2
> 5. Invalid service name format → Validation error, exit 2
> 6. GitLab API timeout → Graceful error message, exit 2
>
> **Performance Test:**
> - Graph with 100 services → Check completes in < 5 seconds

---

## Edge Cases & Error Handling
**Purpose:** Define how edge cases are handled.

**Questions if missing:**
- What edge cases exist?
- How should each edge case be handled?
- What error messages should be shown?
- What should happen on partial failures?

**Example:**
> **Edge Case: Circular Dependencies**
> - Detected during analysis
> - Show error: "Circular dependency detected: A → B → C → A"
> - Exit code: 1
>
> **Edge Case: Stale Dependency Data**
> - Show warning: "Data is 24h old, run 'paragon-deps analyze' to refresh"
> - Proceed with stale data (don't block user)
>
> **Edge Case: Partial GitLab API Failure**
> - Show warning: "Could not fetch metadata for 2 services"
> - Show results for available services
> - Include warning in output

---

## Open Questions
**Purpose:** Track unresolved questions about this story.

**Questions if missing:**
- What questions remain unanswered?
- What design decisions are pending?
- What needs clarification from stakeholders?

**Example:**
> - Should we auto-run this check in pre-commit hooks, or only on-demand?
> - What should we do if multiple conflicting versions are equally valid?
> - Should we integrate with Slack to notify affected teams?
