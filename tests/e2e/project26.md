**Input Format:** Verbal Description via Email

**Subject:** Distributing heavy maps without a central server

When a hurricane hits, the cell towers go down, but we set up a local mesh network of relief trucks. We need to distribute a massive 50GB "Damage Map" file to all 100 trucks.

We can't have everyone download from the HQ truck—it'll crash the HQ's bandwidth.

I need a system where:
1.  **Chunking:** The HQ truck breaks the file into 2MB pieces.
2.  **Swarming:** Truck A downloads Chunk 1 from HQ. Truck B sees that Truck A has Chunk 1, so it downloads it from Truck A (not HQ), while simultaneously downloading Chunk 2 from HQ.
3.  Eventually, all trucks are swapping chunks with each other.
4.  **Verification:** You need to verify every chunk with a hash (SHA-256) so nobody gets a corrupted map.
