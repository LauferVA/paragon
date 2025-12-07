**Input Format:** Product Requirement Document (PRD)

# PRD: The Standardized Treaty Translator

## 1. Problem
Our diplomats write treaties in a high-level, human-readable format called "Protocol-A." However, the execution machines (the bureaucracy) only understand a low-level, rigid instruction set called "Assembly-B."

## 2. The Solution
We need a translation pipeline that converts "Protocol-A" text into "Assembly-B" instructions.

## 3. Phases
* **Tokenization:** Break the sentence `IF Treaty_Active THEN Pay_Gold` into `[IF, ID, THEN, ID]`.
* **Grammar Check:** Ensure the sentence structure is valid according to the rulebook.
* **Optimization:** If a treaty says "Pay 5 Gold" and later "Pay 0 Gold", simplify it to just "Pay 5 Gold" before finalizing.
* **Generation:** Output the final machine codes.
