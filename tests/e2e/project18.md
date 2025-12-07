**Input Format:** Product Requirement Document (PRD)

# PRD: Geo-Layer Visualizer

## 1. The Viewport
The user sees a 2D top-down map of a geological survey. They need to scroll through "Depth" (Z-axis) to see different layers of the earth.

## 2. Tools
* **Windowing:** The user drags the mouse right/left to change the **Contrast** of the thermal layer.
* **Measurement:** User clicks two points; system calculates real-world distance based on the file's metadata tags.
* **Cine Mode:** A button to auto-play through the depth layers like a movie loop.

## 3. Architecture
The frontend must be a "dumb" canvas for performance. The backend handles the heavy lifting of parsing the proprietary `.geo` files (which are massive 500MB blocks) and sending simple JPEG tiles to the client on demand.
