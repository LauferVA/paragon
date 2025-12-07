**Input Format:** Technical Specification

# Tech Spec: The Indestructible Data Store

## Goal
Store 10 Petabytes of genetic sequencing data across 5,000 cheap, unreliable hard drives scattered across the globe.

## Content Addressing
Files are not retrieved by location (`server1/file.txt`). They are retrieved by their content hash (`CID`). If I ask for `Hash_ABC`, the network finds who has it.

## Block Management
* **Splitting:** Files are split into 256MB blocks.
* **Replication Factor:** Every block must be stored on at least 3 different physical drives.
* **Rebalancing:** A background daemon monitors drive health. If Drive A fails, the system detects the under-replicated blocks and copies them to Drive B to restore the replication factor to 3.
