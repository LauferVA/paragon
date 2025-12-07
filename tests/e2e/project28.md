**Input Format:** Product Requirement Document (PRD)

# PRD: The "Shell Company" Analyzer

## 1. Data Model
We are not storing rows and columns. We are storing `Entities` (People, Companies) and `Relationships` (Owns, Married_To, Director_Of).

## 2. Storage Requirement
We need to store these connections on disk in a way that makes "hopping" instant. If I have a Person, I should find their Company immediately without scanning a table index. The pointers should be physical byte offsets on the disk.

## 3. The Query Language
We need a way to ask deep recursive questions:
*"Find all Companies that are 5 hops away from Person X via 'Director_Of' relationships."*

The system must traverse these millions of relationships efficiently to flag circular ownership loops (e.g., Company A owns B, B owns C, C owns A).
