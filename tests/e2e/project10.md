**Input Format:** Technical Specification

# Tech Spec: Laboratory Sample Inventory System (LSIS)

## 1. Core Data Model
We need a flexible schema system. Unlike a standard database with fixed columns, our "Sample Types" change frequently.
* **Schema Definition:** Admins can define a `SampleType` (e.g., "Blood Vial") and its required fields (e.g., "Volume", "DonorID").
* **Entry Creation:** Users create `Entries` that match a specific `SampleType`. The system must validate that the Entry has the correct fields for that Type.

## 2. API Requirements
* `POST /schema`: Create a new definition.
* `POST /entry`: Create a new data point. Must return 400 Error if the JSON body doesn't match the active Schema.
* `GET /entries`: Return JSON list.

## 3. Architecture
This is effectively a content management backend. It needs to serve JSON to our frontend React app.
