import numpy as np

class SensorFusion:
    def __init__(self):
        self.camera_intrinsic = np.array([
            [718.0, 0, 607.2],
            [0, 718.0, 185.2],
            [0, 0, 1]
        ])  # KITTI camera parameters
        
        self.camera_to_lidar_extrinsic = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, -0.08],
            [1, 0, 0, -0.27],
            [0, 0, 0, 1]
        ])  # Example extrinsic matrix
    
    def project_lidar_to_camera(self, lidar_points):
        """
        Project LiDAR points to camera image
        """
        # Remove points behind camera
        mask = lidar_points[:, 0] > 0
        lidar_points = lidar_points[mask]
        
        # Transform LiDAR points to camera coordinates
        points_cam = (self.camera_to_lidar_extrinsic[:3, :3] @ lidar_points.T).T + \
                    self.camera_to_lidar_extrinsic[:3, 3]
        
        # Project to image plane
        points_2d = (self.camera_intrinsic @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        return points_2d, points_cam
    
    def fuse_camera_lidar(self, camera_detections, lidar_points, image_shape):
        """
        Fuse camera detections with LiDAR points
        """
        # Project LiDAR to camera
        points_2d, points_cam = self.project_lidar_to_camera(lidar_points)
        
        # Filter points within image bounds
        height, width = image_shape[:2]
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
        points_2d = points_2d[valid_mask]
        points_cam = points_cam[valid_mask]
        
        # Associate LiDAR points with camera detections
        fused_detections = []
        
        for det in camera_detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Find LiDAR points within detection bounding box
            in_bbox_mask = (points_2d[:, 0] >= x1) & (points_2d[:, 0] <= x2) & \
                          (points_2d[:, 1] >= y1) & (points_2d[:, 1] <= y2)
            
            bbox_points = points_cam[in_bbox_mask]
            
            if len(bbox_points) > 0:
                # Calculate 3D bounding box from LiDAR points
                min_vals = bbox_points.min(axis=0)
                max_vals = bbox_points.max(axis=0)
                
                # Estimate object dimensions
                dimensions = max_vals - min_vals
                
                # Calculate center
                center = (min_vals + max_vals) / 2
                
                # Calculate distance
                distance = np.linalg.norm(center)
                
                # Update detection with 3D information
                fused_det = det.copy()
                fused_det.update({
                    '3d_bbox': [min_vals[0], min_vals[1], min_vals[2],
                               max_vals[0], max_vals[1], max_vals[2]],
                    'dimensions': dimensions,
                    'center_3d': center,
                    'distance': distance,
                    'num_lidar_points': len(bbox_points),
                    'points_3d': bbox_points
                })
                
                fused_detections.append(fused_det)
        
        return fused_detections
    
    def kalman_filter_fusion(self, detections, dt=0.1):
        """
        Kalman filter for object tracking and state estimation
        """
        class KalmanFilter:
            def __init__(self, dt=0.1):
                # State: [x, y, z, vx, vy, vz, width, height, length]
                self.n_states = 9
                self.n_measurements = 6  # x, y, z, width, height, length
                
                # State transition matrix
                self.F = np.eye(self.n_states)
                self.F[0, 3] = dt
                self.F[1, 4] = dt
                self.F[2, 5] = dt
                
                # Measurement matrix
                self.H = np.zeros((self.n_measurements, self.n_states))
                self.H[:6, :6] = np.eye(6)
                self.H[3, 6] = 1
                self.H[4, 7] = 1
                self.H[5, 8] = 1
                
                # Process noise covariance
                self.Q = np.eye(self.n_states) * 0.1
                
                # Measurement noise covariance
                self.R = np.eye(self.n_measurements) * 0.5
                
                # State covariance
                self.P = np.eye(self.n_states) * 10
                
                # State
                self.x = np.zeros(self.n_states)
            
            def predict(self):
                self.x = self.F @ self.x
                self.P = self.F @ self.P @ self.F.T + self.Q
                return self.x
            
            def update(self, z):
                # Kalman gain
                S = self.H @ self.P @ self.H.T + self.R
                K = self.P @ self.H.T @ np.linalg.inv(S)
                
                # Update state
                y = z - self.H @ self.x
                self.x = self.x + K @ y
                
                # Update covariance
                I = np.eye(self.n_states)
                self.P = (I - K @ self.H) @ self.P
                
                return self.x
        
        # Create or update Kalman filters for each detection
        filtered_detections = []
        
        for det in detections:
            if 'kf' not in det:
                # Initialize new Kalman filter
                det['kf'] = KalmanFilter(dt)
                
                # Initial state from detection
                center = det.get('center_3d', [0, 0, 0])
                dims = det.get('dimensions', [1, 1, 1])
                initial_state = np.array([
                    center[0], center[1], center[2],  # Position
                    0, 0, 0,                         # Velocity (initial 0)
                    dims[0], dims[1], dims[2]        # Dimensions
                ])
                det['kf'].x = initial_state
            
            # Predict
            predicted_state = det['kf'].predict()
            
            # Prepare measurement
            if 'center_3d' in det and 'dimensions' in det:
                z = np.array([
                    det['center_3d'][0],
                    det['center_3d'][1],
                    det['center_3d'][2],
                    det['dimensions'][0],
                    det['dimensions'][1],
                    det['dimensions'][2]
                ])
                
                # Update
                updated_state = det['kf'].update(z)
                
                # Update detection with filtered state
                det['filtered_center'] = updated_state[:3]
                det['filtered_velocity'] = updated_state[3:6]
                det['filtered_dimensions'] = updated_state[6:9]
            
            filtered_detections.append(det)
        
        return filtered_detections
    
    def particle_filter_for_nonlinear(self, detections, n_particles=100):
        """
        Particle filter for nonlinear motion models
        """
        class ParticleFilter:
            def __init__(self, n_particles=100, state_dim=6):
                self.n_particles = n_particles
                self.state_dim = state_dim
                
                # Initialize particles
                self.particles = np.random.randn(n_particles, state_dim) * 0.1
                self.weights = np.ones(n_particles) / n_particles
                
            def predict(self, dt=0.1):
                # Add process noise
                process_noise = np.random.randn(self.n_particles, self.state_dim) * 0.1
                self.particles += process_noise
                
                # Simple motion model
                self.particles[:, 0] += self.particles[:, 3] * dt  # x += vx * dt
                self.particles[:, 1] += self.particles[:, 4] * dt  # y += vy * dt
                self.particles[:, 2] += self.particles[:, 5] * dt  # z += vz * dt
            
            def update(self, measurement, measurement_noise=0.1):
                # Calculate likelihood
                residuals = self.particles - measurement
                likelihoods = np.exp(-0.5 * np.sum(residuals**2, axis=1) / measurement_noise**2)
                
                # Update weights
                self.weights *= likelihoods
                self.weights /= np.sum(self.weights) + 1e-10
                
                # Resample if needed
                effective_n = 1.0 / np.sum(self.weights**2)
                if effective_n < self.n_particles / 2:
                    self._resample()
            
            def _resample(self):
                # Systematic resampling
                indices = np.zeros(self.n_particles, dtype=int)
                cumulative_sum = np.cumsum(self.weights)
                step = 1.0 / self.n_particles
                u = np.random.rand() * step
                
                i = 0
                for j in range(self.n_particles):
                    while u > cumulative_sum[i]:
                        i += 1
                    indices[j] = i
                    u += step
                
                self.particles = self.particles[indices]
                self.weights = np.ones(self.n_particles) / self.n_particles
            
            def estimate(self):
                # Weighted mean
                return np.average(self.particles, weights=self.weights, axis=0)
        
        # Apply particle filter to each detection
        for det in detections:
            if 'pf' not in det:
                det['pf'] = ParticleFilter(n_particles=n_particles)
            
            # Predict
            det['pf'].predict()
            
            # Update if measurement available
            if 'center_3d' in det:
                measurement = np.array([
                    det['center_3d'][0],
                    det['center_3d'][1],
                    det['center_3d'][2],
                    0, 0, 0  # Velocity unknown
                ])
                det['pf'].update(measurement)
                
                # Get estimate
                det['pf_estimate'] = det['pf'].estimate()
        
        return detections

