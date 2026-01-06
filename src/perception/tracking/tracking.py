import numpy as np
from scipy.optimize import linear_sum_assignment

class ObjectTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_age = 5  # Frames to keep lost tracks
        
    def hungarian_matching(self, detections, tracks, cost_threshold=0.7):
        """
        Match detections to tracks using Hungarian algorithm
        """
        if len(detections) == 0 or len(tracks) == 0:
            return [], list(range(len(detections))), list(tracks.keys())
        
        # Build cost matrix
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track_id in enumerate(tracks.keys()):
                track = tracks[track_id]
                
                # Calculate IoU between detection and last track position
                if 'bbox' in det and 'bbox_history' in track and len(track['bbox_history']) > 0:
                    last_bbox = track['bbox_history'][-1]
                    iou = self._calculate_iou(det['bbox'], last_bbox)
                    cost_matrix[i, j] = 1 - iou  # Convert similarity to cost
                else:
                    cost_matrix[i, j] = 1.0  # Max cost
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches by cost threshold
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(tracks.keys())
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < cost_threshold:
                matches.append((i, list(tracks.keys())[j]))
                unmatched_detections.remove(i)
                unmatched_tracks.remove(list(tracks.keys())[j])
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update_tracks(self, detections, frame_id):
        """
        Update tracks with new detections
        """
        # Match detections to existing tracks
        matches, unmatched_dets, unmatched_tracks = self.hungarian_matching(
            detections, self.tracks
        )
        
        # Update matched tracks
        for det_idx, track_id in matches:
            det = detections[det_idx]
            
            if track_id in self.tracks:
                track = self.tracks[track_id]
                
                # Update track
                track['bbox'] = det['bbox']
                track['bbox_history'].append(det['bbox'])
                track['frame_history'].append(frame_id)
                track['age'] = 0  # Reset age
                
                # Update 3D info if available
                if 'center_3d' in det:
                    track['center_3d'] = det['center_3d']
                    track['center_3d_history'].append(det['center_3d'])
                
                # Update class
                if 'class' in det:
                    track['class'] = det['class']
                
                # Update other info
                track.update({k: v for k, v in det.items() 
                            if k not in ['bbox', 'center_3d']})
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'track_id': track_id,
                'bbox': det['bbox'],
                'bbox_history': [det['bbox']],
                'frame_history': [frame_id],
                'age': 0,
                'center_3d_history': [det.get('center_3d', [0, 0, 0])],
                'class': det.get('class', 'unknown'),
                **{k: v for k, v in det.items() 
                  if k not in ['bbox', 'center_3d']}
            }
        
        # Increment age for unmatched tracks and remove old ones
        tracks_to_remove = []
        for track_id in unmatched_tracks:
            if track_id in self.tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return self.tracks
    
    def predict_trajectories(self, tracks, prediction_horizon=10):
        """
        Predict future trajectories using simple motion models
        """
        predictions = {}
        
        for track_id, track in tracks.items():
            if len(track['bbox_history']) < 2:
                continue
            
            # Get recent positions
            positions = np.array(track['bbox_history'][-5:])  # Last 5 frames
            
            # Calculate center points
            centers = np.array([[(x1+x2)/2, (y1+y2)/2] 
                              for x1, y1, x2, y2 in positions])
            
            # Simple linear prediction
            if len(centers) >= 2:
                # Calculate velocity
                velocity = centers[-1] - centers[-2]
                
                # Predict future positions
                future_positions = []
                for t in range(1, prediction_horizon + 1):
                    pred_center = centers[-1] + velocity * t
                    
                    # Assume constant size
                    bbox_size = positions[-1][2:] - positions[-1][:2]
                    pred_bbox = np.concatenate([
                        pred_center - bbox_size/2,
                        pred_center + bbox_size/2
                    ])
                    
                    future_positions.append(pred_bbox)
                
                predictions[track_id] = {
                    'current_bbox': track['bbox'],
                    'future_bboxes': future_positions,
                    'velocity': velocity,
                    'prediction_confidence': min(1.0, len(positions) / 10.0)
                }
        
        return predictions
    
    def deep_sort_tracking(self, detections, frame_id):
        """
        Simplified DeepSORT-like tracking with appearance features
        """
        # Extract appearance features (simplified - would use CNN in practice)
        appearance_features = []
        for det in detections:
            # In practice, use ReID model to extract features
            # For demo, use random features
            feature = np.random.randn(128)
            feature = feature / np.linalg.norm(feature)  # Normalize
            det['appearance'] = feature
            appearance_features.append(feature)
        
        # Match using appearance + motion
        if len(self.tracks) > 0 and len(detections) > 0:
            # Build cost matrix with appearance similarity
            cost_matrix = np.zeros((len(detections), len(self.tracks)))
            
            for i, det in enumerate(detections):
                for j, (track_id, track) in enumerate(self.tracks.items()):
                    # Appearance cost
                    if 'appearance' in det and 'appearance_history' in track:
                        # Cosine similarity
                        last_appearance = track['appearance_history'][-1]
                        cos_sim = np.dot(det['appearance'], last_appearance)
                        appearance_cost = 1 - (cos_sim + 1) / 2  # Convert to [0, 1]
                    else:
                        appearance_cost = 0.5
                    
                    # Motion cost (IoU)
                    motion_cost = 1
                    if 'bbox' in det and 'bbox_history' in track and len(track['bbox_history']) > 0:
                        iou = self._calculate_iou(det['bbox'], track['bbox_history'][-1])
                        motion_cost = 1 - iou
                    
                    # Combined cost
                    cost_matrix[i, j] = 0.7 * appearance_cost + 0.3 * motion_cost
            
            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Update tracks
            matched_pairs = []
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < 0.5:  # Threshold
                    matched_pairs.append((i, list(self.tracks.keys())[j]))
        
        # Rest of tracking logic similar to update_tracks
        return self.update_tracks(detections, frame_id)

