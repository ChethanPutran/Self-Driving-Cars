import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class ClassicalPerception:
    def __init__(self):
        self.feature_detector = cv2.SIFT()
        self.matcher = cv2.BFMatcher()
        
    def lane_detection(self, image):
        """
        Traditional lane detection using Canny + Hough Transform
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Create mask for region of interest (road area)
        height, width = edges.shape
        mask = np.zeros_like(edges, dtype=np.uint8)
        ROI_polygon = np.array([[
            (width * 0.1, height),
            (width * 0.45, height * 0.6),
            (width * 0.55, height * 0.6),
            (width * 0.9, height)
        ]], dtype=np.int32)

        mask = cv2.fillPoly(mask, ROI_polygon, 255)

        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough Transform for line detection
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=100
        )
        
        # Separate left and right lanes
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter by slope and position
                if abs(slope) > 0.3:  # Avoid horizontal lines
                    if slope < 0 and x1 < width/2 and x2 < width/2:
                        left_lines.append(line[0])
                    elif slope > 0 and x1 > width/2 and x2 > width/2:
                        right_lines.append(line[0])
        
        return left_lines, right_lines, masked_edges
    
    def optical_flow(self, prev_frame, curr_frame, method='lk'):
        """
        Compute optical flow for motion estimation
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if method == 'lk':
            # Lucas-Kanade optical flow
            # Detect good features to track
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30
            )
            
            if prev_pts is None:
                return None, None
            
            # Calculate optical flow
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                curr_gray,
                prev_pts,
                None,
                winSize=(15, 15),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Filter only good points
            good_prev = prev_pts[status == 1]
            good_curr = curr_pts[status == 1]
            
            # Calculate flow vectors
            flow_vectors = good_curr - good_prev
            
            return good_prev, flow_vectors
            
        elif method == 'farneback':
            # Farneback dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            
            # Convert to magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            return flow, magnitude, angle
    
    def stereo_depth(self, left_image, right_image):
        """
        Stereo vision for depth estimation
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Stereo matching
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,  # Must be divisible by 16
            blockSize=11,
            P1=8 * 3 * 11 ** 2,
            P2=32 * 3 * 11 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Convert disparity to depth
        # Assuming baseline = 0.12m and focal length = 718 pixels (KITTI params)
        baseline = 0.12
        focal_length = 718.0
        depth = (baseline * focal_length) / (disparity + 1e-6)
        
        # Filter invalid depths
        depth[depth > 100] = 100  # Max depth 100m
        depth[disparity <= 0] = 0
        
        return depth, disparity
    
    def feature_based_object_detection(self, image):
        """
        Traditional feature-based object detection
        """
        # Detect keypoints and descriptors
        keypoints, descriptors = self.feature_detector.detectAndCompute(image, None)
        
        # Load template features for different objects
        templates = {
            'car': self._load_template_features('car'),
            'pedestrian': self._load_template_features('pedestrian'),
            'traffic_light': self._load_template_features('traffic_light')
        }
        
        detected_objects = []
        
        for obj_name, template_desc in templates.items():
            if descriptors is not None and template_desc is not None:
                # Match features
                matches = self.matcher.knnMatch(descriptors, template_desc, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                # If enough good matches, consider object detected
                if len(good_matches) > 10:
                    # Get matched keypoints
                    src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
                    dst_pts = np.float32([template_desc[m.trainIdx] for m in good_matches])
                    
                    # Find bounding box using clustering
                    if len(src_pts) > 4:
                        clustering = DBSCAN(eps=50, min_samples=3).fit(src_pts)
                        
                        for cluster_id in set(clustering.labels_):
                            if cluster_id != -1:  # Ignore noise
                                cluster_points = src_pts[clustering.labels_ == cluster_id]
                                if len(cluster_points) > 5:
                                    x_min, y_min = cluster_points.min(axis=0)
                                    x_max, y_max = cluster_points.max(axis=0)
                                    
                                    detected_objects.append({
                                        'class': obj_name,
                                        'bbox': [x_min, y_min, x_max, y_max],
                                        'confidence': len(cluster_points) / len(src_pts),
                                        'keypoints': cluster_points
                                    })
        
        return detected_objects
    
    def _load_template_features(self, object_type):
        """Load pre-computed template features"""
        # In practice, load from file
        # For demo, return None
        return None
    
    def motion_segmentation(self, prev_frame, curr_frame):
        """
        Segment moving objects using frame differencing
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        moving_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                moving_objects.append({
                    'bbox': [x, y, x+w, y+h],
                    'area': area
                })
        
        return moving_objects, thresh