**Input Format:** Technical Specification

# Tech Spec: Reliable Message Transport Protocol (RMTP)

## Overview
We have an unreliable wire that drops 10% of packets and reorders them randomly. We need a software layer on top of this that guarantees perfect, ordered delivery of a data stream.

## Packet Structure
Every packet must have a `SequenceNumber` (32-bit) and a `Checksum`.

## State Machine
* **The Sender:** Must keep a "Sliding Window" of unacknowledged packets. If it doesn't receive an `ACK` for Packet 5 within 200ms, it must resend Packet 5.
* **The Receiver:** Must buffer out-of-order packets. If it gets Packet 1 and Packet 3, it must hold Packet 3 in a "Reassembly Queue" and wait for Packet 2 before handing the data to the user.
* **Congestion:** If packets are dropping, the Sender must automatically slow down its transmission rate (halve the window size) to let the network recover.
