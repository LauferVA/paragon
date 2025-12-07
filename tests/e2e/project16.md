**Input Format:** Technical Specification

# Tech Spec: Environmental Services Dispatcher (ESD)

## Components
1.  **The Producer (The API):** Nurses click "Discharge Patient" → Sends a `CleanRequest` payload (Room #, Hazard Level) to the Broker.
2.  **The Broker (Redis-backed):** Holds the queue of dirty rooms. Priority is determined by `Hazard Level` (Level 1 = Immediate, Level 3 = Routine).
3.  **The Consumers (The Janitors):** We have 50 active worker tablets. When a worker marks "Ready," they pull the highest-priority job from the Broker.

## Resilience Requirements
* **The "Dead Zone" Problem:** The hospital basement has bad Wi-Fi. A worker might accept a job and then lose connection.
* **Heartbeat Logic:** If a worker tablet stops sending a heartbeat for > 5 minutes while a job is "In Progress," the system must assume the worker is unavailable. It must **re-queue** that specific room so another janitor can take it. We cannot have a room blocked forever by a disconnected tablet.
