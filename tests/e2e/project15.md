**Input Format:** Product Requirement Document (PRD)

# PRD: The "League Time Machine"

## 1. Concept
Users want to test if a specific fantasy sports drafting strategy (e.g., "Always pick a Quarterback in Round 1") would have won the league in 2010, 2011, or 2012.

## 2. Data Ingest
The system must load a massive CSV of "Historical Game Events" (e.g., `PlayerID: 55, Event: Touchdown, Timestamp: 2010-09-12 14:05:00`).

## 3. The Simulation Engine
The engine must "replay" a season from start to finish.
* **Input:** A User Strategy Function (logic defined in code).
* **Process:**
    * Iterate through time day-by-day.
    * At the start of the week, the Strategy Function selects a "Lineup."
    * The Engine checks the Historical Data to see how those players performed.
    * Calculate points and update the User's "League Rank."
* **Output:** A report showing the final Win/Loss record for that strategy over the season.
