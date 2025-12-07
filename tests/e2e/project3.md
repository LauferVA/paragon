**Input Format:** Technical Specification

# Tech Spec: Uplink Heartbeat Monitor

## Overview

A daemon process required to maintain active sessions with our orbital assets. The system must execute specific shell commands at precise time intervals defined in a configuration file.

## Configuration Schema (JSON)

The system must ingest a config file defining tasks:

```json
{
  "uplink_alpha": { "frequency": "*/5 * * * *", "command": "/bin/ping_alpha" },
  "uplink_beta":  { "frequency": "0 12 * * *",  "command": "/bin/reset_beta" }
}
```

## Core Logic

1.  **Parser:** Parse standard Unix-style time expressions (Minute, Hour, Day of Month, Month, Day of Week).
2.  **Event Loop:** Maintain a main loop that checks the current system time against the schedule every minute.
3.  **Execution:** Spawn a child process to execute the `command` when the time matches.
4.  **Logging:** Capture `stdout` and `stderr` of the child process to a daily log file.
