**Input Format:** Technical Specification

# Tech Spec: Fleet Resource Scheduler

## Entities
* **The Node:** A physical drone with limited CPU and Battery (RAM).
* **The Pod:** A mission software package (e.g., "Camera_V1") that needs 500MB RAM.

## The Controller Manager
A central loop that compares "Desired State" vs "Actual State."
* **Desired:** "Run 5 copies of Camera_V1."
* **Actual:** "Only 3 are running."
* **Action:** Select 2 available Nodes that have enough free RAM and schedule the missing Pods on them.

## Self-Healing
The Scheduler must heartbeat every Node. If Node 4 stops responding (crashes), the Scheduler must declare it "Dead" and immediately reschedule its Pods onto Node 5 to maintain the Desired State.
