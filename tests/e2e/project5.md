**Input Format:** Product Requirement Document (PRD)

# PRD: Inventory Sticker System (ISS)

## 1. Problem Statement

Our internal SKUs are 64 characters long (e.g., `US-MI-AA-SECTION-4-ROW-9-BIN-22`). These don't fit on the small 1x1 inch stickers we use for sorting. We need a system to generate a unique, random 6-character alphanumeric code that maps to the long SKU.

## 2. Requirements

  * **Generation:** Input a Long SKU, receive a Short Code (e.g., `aX9j2L`).
  * **Collision Avoidance:** The system must ensure no two SKUs get the same Short Code.
  * **Lookup:** Scanning the Short Code must instantly return the Long SKU.
  * **Tracking:** We need to count how many times each Short Code is scanned to identify high-traffic items.

## 3. Interface

A simple HTTP API with:

  * `POST /register` (Input: Long SKU -> Output: Short Code)
  * `GET /lookup/{code}` (Input: Short Code -> Output: Long SKU + Increment Count)
