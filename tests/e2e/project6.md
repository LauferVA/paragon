**Input Format:** Product Requirement Document (PRD)

# PRD: Ground-to-Air Text Link (GATL)

## 1. Objective
A low-bandwidth, real-time messaging server to allow Tower Controllers to send text instructions to Pilots when radio frequency is congested.

## 2. Key Features
* **Channels:** Pilots enter a specific "Frequency Room" (e.g., `LAX_Tower_1`) based on their location.
* **Real-Time Delivery:** Messages must appear instantly on all connected screens in that room. No page refreshes allowed.
* **Persistence:** If a pilot disconnects and reconnects (e.g., due to temporary signal loss), they should automatically receive the last 50 messages to regain context.
* **Status Indicators:** The Tower interface needs to see a live "Connected" indicator next to the Pilot's callsign.
* **Broadcast:** The Tower controller must be able to send a "Global Alert" that goes to all rooms simultaneously.

## 3. Technical Constraints
* Must support 5,000 concurrent open connections.
* Message delivery latency must be under 200ms.
