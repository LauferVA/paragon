**Input Format:** Product Requirement Document (PRD)

# PRD: Star Pattern Identification System

## 1. The Goal
Astronomers upload a photo of a random patch of sky. The system converts the star positions into a "Feature Vector" (a list of 512 floating-point numbers representing the geometry).

## 2. The Challenge
We have a database of 1 billion known star clusters (vectors). We need to find the "Nearest Neighbor"—the known cluster that is mathematically closest to the uploaded vector.

## 3. Performance
A standard linear search (checking all 1 billion) takes too long. You need to build a specialized graph index where similar vectors are linked.
* **Ingest:** When a new cluster is discovered, navigate the graph to find its "friends" and link them.
* **Search:** Traverse the graph greedily to find the closest match in milliseconds.
