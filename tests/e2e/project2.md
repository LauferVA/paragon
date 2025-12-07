**Input Format:** Product Requirement Document (PRD)

# PRD: Intelligent Flow Control System (IFCS)

## 1. Executive Summary

The IFCS is a software module designed to protect the integrity of the Main Trunk conveyor system in our distribution center. It acts as a gatekeeper, analyzing incoming flow rates from distinct upstream sources and enforcing pre-defined throughput limits to prevent system-wide congestion.

## 2. User Stories

  * **As a Site Admin**, I want to define a specific "Maximum Flow Rate" (items per minute) for each unique Feeder Source ID, so that high-volume lines get more capacity than low-volume lines.
  * **As the System**, I must make a go/no-go decision for every item within 2ms of scanning to ensure the physical line speed is maintained.
  * **As the System**, I must "reset" the allowance for a Feeder Source automatically over time, so that a penalized line can resume operation once its burst of traffic has subsided.
  * **As an Operator**, I want to see a log or metric of how many items were "Diverted" per Feeder Source so I can identify malfunctioning equipment.

## 3. Functional Requirements

1.  **Identification:** The system receives a `ScanEvent` containing the `SourceID` (string) and `Timestamp` (UTC).
2.  **Decision Engine:**
      * The engine must check the recent history of that `SourceID`.
      * If the count of items in the current sliding window exceeds the `Limit`, return `Action: DIVERT`.
      * Otherwise, return `Action: PASS` and increment the count.
3.  **Concurrency:** The system must handle up to 10,000 `ScanEvents` per second arriving from multiple parallel scanners simultaneously without race conditions.
