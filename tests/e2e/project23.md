**Input Format:** Verbal Description via Email

**Subject:** Need a simulator for vehicle impact testing

We're spending too much money crashing real cars. I need a software simulator where I can define 3D objects (cars, walls, dummies) as collections of "Rigid Bodies."

The software needs a main loop that runs 60 times a second. In every frame, it needs to:
1.  **Integration:** Apply gravity and velocity to every object.
2.  **Detection:** Check for collisions. If Car A's bounding box overlaps with Wall B, you need to calculate the exact point of impact.
3.  **Resolution:** Calculate the "bounce" vector based on their mass and restitution (bounciness) and push them apart so they don't sink into each other.

It needs to be accurate. You'll likely need to implement something called "Broad Phase" (to find potential hits quickly) and "Narrow Phase" (to calculate the exact math).
