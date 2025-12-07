**Input Format:** Technical Specification

# Tech Spec: High-Rise Tenant Portal

## Data Isolation (Crucial)
We have 50 different buildings (Tenants) using this system.
* **Requirement:** Building A's admin *cannot* see Building B's maintenance requests.
* **Implementation:** All data lives in one Postgres DB for cost reasons, but every query *must* automatically append `WHERE building_id = X`.

## Role-Based Access Control (RBAC)
* **Super Admin:** Can create new Buildings and set subscription limits.
* **Building Manager:** Can add Residents to their specific building.
* **Resident:** Can only see their own payment history and tickets.

## Scalability
We need to be able to spin up a new Building in seconds without running database migrations or restarting the server.
