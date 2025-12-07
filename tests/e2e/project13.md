**Input Format:** Technical Specification

# Tech Spec: Stadium Egress Safety Sim

## Overview
A physics-based engine to simulate 10,000 agents ("People") trying to exit a room through a single door. This is to test building code compliance.

## Agent Logic (Per 'Tick')
The simulation runs in a loop with a time step ($dt$) of 0.1 seconds.
1.  **Goal Vector:** Every agent has a desired velocity vector pointing toward the coordinate $(X_{door}, Y_{door})$.
2.  **Repulsion Force:** If Agent A is within 0.5 meters of Agent B, calculate a repulsive force vector inversely proportional to the distance (to prevent overlap).
3.  **Wall Collision:** Agents cannot pass through coordinates defined as "Walls" in the `map.json` file.
4.  **Integration:** Update Position = Position + (Velocity * $dt$).

## Output
The system must output a `.frame` file every 100ms containing the $(X,Y)$ coordinates of every agent, so we can render a movie of the crowd flow.
