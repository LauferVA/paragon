**Input Format:** Verbal Description via Chat

**User:** The Master Control Program for the building.
**Context:** Think of the entire skyscraper as one computer.

**Request:**
I need you to write the low-level code that manages these resources and offers a safe API for the apps (HVAC, Security) to run on top of.
* **Memory Management:** We have limited power (RAM). You need to allocate power to different floors (Processes). If a floor isn't being used, page it out (turn off lights) to save resources.
* **Scheduling:** We have 1,000 requests for the elevators (CPU). You need a scheduler (Round Robin or Priority) to decide who moves next so nobody waits forever.
* **Interrupts:** If the Fire Alarm is pulled (Hardware Interrupt), the CPU must stop whatever it's doing immediately and run the "Emergency Handler" code.
