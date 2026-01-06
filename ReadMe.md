### Autonomous Driving Perception System 


This project shows complete autonomous driving perception stack from classical methods to sota deep learning

--- 

## OVERVIEW

A modular perception system for self-driving cars with:

1. Classical Computer Vision (feature-based methods)
2. Deep Learning (YOLO, U-Net, Monodepth2)
3. Sensor Fusion (Camera + LiDAR)
4. Tracking (Kalman filter + DeepSORT)

---



## FEATURES 

### 1. Classical Computer Vision:

* Lane detection (Canny + Hough Transform)
* Optical flow (Lucas-Kanade, Farneback)
* Stereo depth estimation
* Feature-based object detection

### 2. Deep Learning:

* Object detection (YOLOv5)
* Semantic segmentation (U-Net)
* Depth estimation (Monodepth2)
* Multi-task learning architecture

### 3. Sensor Fusion:

* Camera-LiDAR fusion
* Kalman filtering
* Particle filtering
* BEV transformation

### 4. Tracking:

* Hungarian algorithm matching
* DeepSORT with appearance features
* Trajectory prediction
* Multi-object tracking

### 5. Planning:

* Bird's Eye View representation
* A* path planning
* Obstacle avoidance



### TASKS TO BE DONE:

Tasks to be dine:
1. Add more sensors (Radar, Ultrasonic)
2. Implement SLAM for localization
3. Add behavior prediction using RNNs
4. Real-time optimization with TensorRT
5. Simulation testing with CARLA
6. Visualizing the results

### DATASETS TO USE:

1. KITTI (Autonomous driving benchmark)
2. nuScenes (Multi-modal autonomous driving)
3. Waymo Open Dataset
4. Cityscapes (Urban scene understanding)
5. BDD100K (Diverse driving scenarios)

