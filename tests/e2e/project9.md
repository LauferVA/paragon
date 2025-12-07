**Input Format:** Product Requirement Document (PRD)

# PRD: Proximity Alert System (PAS)

## 1. Overview
We track animals in a national park using GPS collars. We need a system to process these coordinate streams and detect "Conflict Events" in real-time.

## 2. Actors
* **Type A (Predators):** Moving points, updated every 5 seconds.
* **Type B (Herds):** Moving points, updated every 5 seconds.

## 3. Logic
* For every location update from a Type A animal, calculate the distance to all Type B animals.
* If distance < 1.0 km, trigger an `AlertEvent` (send webhook).

## 4. Optimization Requirement
Brute force calculation (comparing every A against every B) is too slow because we have thousands of animals. You need a spatial index strategy to quickly discard distant animals and only calculate the math for nearby ones.
