**Input Format:** Technical Specification

# Tech Spec: Biohazard Machinery Activation Protocol

## System Architecture
A centralized API that determines if a specific `UserID` is authorized to activate a specific `MachineID`. This decision logic must happen in real-time before the hardware powers on.

## Evaluation Logic
The system must evaluate rules in this strict order:
1.  **Blocklist:** Is `UserID` explicitly banned in the `global_blocklist` table? -> Return `FALSE`.
2.  **Allowlist:** Is `UserID` explicitly allowed in the `machine_access` table? -> Return `TRUE`.
3.  **Percentage Rollout:** (For beta testing new machines). If `hash(UserID + MachineID) % 100 < rollout_percentage` -> Return `TRUE`.

## Performance Constraint
This check happens every time a centrifuge starts. Latency must be < 10ms. The rule definitions (Blocklists/Rollouts) should be cached in memory but must be updateable via a separate Admin API without restarting the service.
