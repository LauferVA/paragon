**Input Format:** Technical Specification

# Tech Spec: The "Iron-Clad" Filing System

## 1. Core Requirement
We need a single-file storage engine to hold millions of census records on an air-gapped machine. We cannot use an external server (like MySQL). The software must manage the raw bytes on the disk directly.

## 2. The Storage Format (Page Structure)
* The file is divided into fixed-size 4KB "Pages."
* **Page 0:** The Header (Metadata).
* **Interior Pages:** Contain pointers to other pages (The Navigation Tree).
* **Leaf Pages:** Contain the actual data rows.

## 3. Query Logic
The system must support a query language that allows us to `SELECT` records where `Age > 18` AND `City = 'Detroit'`. This cannot be a full table scan; you must implement an indexing algorithm (balanced tree) to find the records in $O(\log n)$ time.

## 4. ACID Compliance
If the power plug is pulled while writing a record, the file cannot be corrupted. You must implement a "Write-Ahead Log" (WAL) that records the intent before modifying the Page.
