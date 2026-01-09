### Comprehensive evaluation table for an end-to-end autonomous driving system. 

These metrics assess how well the system perceives its surroundings, predicts the future, and plans a safe path.

### 1. Tracking (Dynamic Perception)

These metrics measure how well the system detects and follows moving objects over time.

* **AMOTA↑ (Average Multi-Object Tracking Accuracy):** The primary metric for 3D tracking. It accounts for errors in detection, misidentification, and localization, averaged over different confidence thresholds. Higher is better.
* **AMOTP↓ (Average Multi-Object Tracking Precision):** Measures the average error in the position (localization) of the tracked objects. Lower is better.
* **IDS↓ (Identity Switches):** The number of times the system "loses" an object and assigns it a new ID, or swaps IDs between two different cars. Lower is better.

### 2. Mapping (Static Perception)

These metrics use **Intersection over Union (IoU)** to see how well the predicted map matches the ground truth.

* **IoU-lane↑:** How accurately the model identifies the geometry of driving lanes.
* **IoU-road↑:** How accurately the model identifies the entire drivable area (the "drivable surface").

### 3. Motion Forecasting (Prediction)

These measure the accuracy of the predicted future paths  of other agents (like pedestrians or cars).

* **minADE↓ (minimum Average Displacement Error):** The average distance between the predicted trajectory and the actual trajectory over all future time steps. "Min" usually refers to picking the best prediction out of  guesses.
* **minFDE↓ (minimum Final Displacement Error):** The distance between the predicted *final* position (e.g., where the car will be in 5 seconds) and the actual final position.
* **MR↓ (Miss Rate):** The percentage of cases where the predicted trajectory was further than a certain threshold (e.g., 2 meters) from the actual ground truth.

### 4. Occupancy Prediction

This evaluates the "Occupancy Grid" or "Voxel" map, often separated into **near (n.)** and **far (f.)** ranges.

* **IoU-n.↑ / IoU-f.↑:** The spatial accuracy of the occupancy grid for nearby and far-away areas.
* **VPQ-n.↑ / VPQ-f.↑ (Video Panoptic Quality):** A more advanced metric that evaluates both "what" is in the grid (classification) and "how" it moves over time (temporal consistency). It’s essentially a measure of how clean and stable the occupancy "blobs" are.

### 5. Planning

These metrics evaluate the final output: the path the autonomous car itself intends to take.

* **avg.L2↓ (Average L2 distance):** The average Euclidean distance between the path the model planned and the path a human expert driver actually took in that same situation.
* **avg.Col.↓ (Average Collision Rate):** Perhaps the most important safety metric. It measures how often the planned path would result in a collision with either a predicted object or a static obstacle.

---

### Summary Table

| Category | Primary Metric | What it "Cares" about |
| --- | --- | --- |
| **Tracking** | AMOTA | Not losing track of the car next to you. |
| **Mapping** | IoU-road | Knowing where the pavement ends. |
| **Motion** | minADE | Guessing correctly where the cyclist is going. |
| **Occupancy** | VPQ | Identifying a "wall" of space that is blocked. |
| **Planning** | avg.Col. | Not hitting anything. |

