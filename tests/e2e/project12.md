**Input Format:** Product Requirement Document (PRD)

# PRD: Museum Provenance & Location System

## 1. Core Entity: The Artifact
Every item must have a unique `AccessionID`.
An artifact can have "Children." For example, if a statue breaks, the main body is `ID-01`, and the broken hand is `ID-01-A`. We need to track them separately but know they are related in the hierarchy.

## 2. Chain of Custody (Immutable)
We need a legal log of every movement. If `ID-01` moves from "Display Hall A" to "Restoration Lab B", that action must be signed by a user and timestamped.
* **Constraint:** This log *cannot* be deleted or edited, even by admins. It must be append-only.

## 3. Workflow Enforcement
An artifact cannot be moved to "Public Display" unless it has a `Status: Cleared` flag. This flag can only be set if the "Condition Report" form has been filled out and approved by a Conservator.
