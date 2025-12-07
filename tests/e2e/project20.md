**Input Format:** Verbal Description via Chat

**User:** Email for the submarine fleet.
**Context:** Our subs are deep underwater and offline 99% of the time. When they surface, they have a 5-minute window to blast all their data to a satellite before diving again.

**Problem:** Standard TCP/IP fails because it expects an instant "ACK" from the destination. If the destination is on the other side of the earth, the sub dives before the confirmation comes back, and the connection times out.

**Request:**
I need a "Store-and-Forward" protocol:
1.  The Submarine creates a "Bundle" of emails and stores them locally.
2.  When the link is up, it blasts the bundle to the Satellite.
3.  **Custody Transfer:** The Satellite accepts the bundle and takes responsibility for it. The Sub deletes its copy.
4.  The Satellite holds the data until it sees a Ground Station, then blasts it down.
5.  If a hop fails, the current holder keeps the data. No data is ever dropped, just delayed.
