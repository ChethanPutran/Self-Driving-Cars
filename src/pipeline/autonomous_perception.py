import cv2
import numpy as np
from collections import deque
from src.perception.classical.cv_methods import ClassicalPerception
from src.perception.dl.models import PerceptionDLModels
from src.perception.tracking.tracking import ObjectTracker
from src.state_estimation.sensor_fusion import SensorFusion
from src.planning.path_planning import BEVTransformer

class AutonomousPerceptionPipeline:
    def __init__(self):
        self.classical = ClassicalPerception()
        self.dl_models = PerceptionDLModels()
        self.fusion = SensorFusion()
        self.tracker = ObjectTracker()
        self.bev = BEVTransformer()
        
        # Load models
        print("Loading models...")
        self.yolo_model = self.dl_models.load_yolo('yolov5s')
        self.unet_model = self.dl_models.UNet().to(self.dl_models.device)
        # Note: In practice, load pre-trained weights
        
        # Initialize
        self.frame_buffer = deque(maxlen=10)
        self.detection_history = []
        
    def process_frame(self, frame, frame_id, lidar_data=None):
        """
        Process single frame through complete pipeline
        """
        print(f"\nProcessing frame {frame_id}...")
        
        # Store frame
        self.frame_buffer.append(frame)
        
        # 1. Classical Computer Vision
        print("  Running classical CV...")
        lanes, _, edges = self.classical.lane_detection(frame)
        moving_objects, motion_mask = self.classical.motion_segmentation(
            self.frame_buffer[-2] if len(self.frame_buffer) >= 2 else frame,
            frame
        )
        
        # 2. Deep Learning Detection
        print("  Running deep learning detection...")
        yolo_detections, _ = self.dl_models.detect_objects_yolo(frame, self.yolo_model)
        
        # 3. Semantic Segmentation
        print("  Running semantic segmentation...")
        seg_map, _ = self.dl_models.segment_road(frame, self.unet_model)
        
        # 4. Depth Estimation (simplified)
        print("  Estimating depth...")
        depth_map = self._simulate_depth(frame)
        
        # 5. Combine detections
        all_detections = []
        for det in yolo_detections:
            all_detections.append({
                'bbox': det['bbox'],
                'class': det['class_name'],
                'confidence': det['confidence'],
                'source': 'yolo'
            })
        
        for obj in moving_objects:
            all_detections.append({
                'bbox': obj['bbox'],
                'class': 'moving',
                'confidence': 0.7,
                'source': 'motion'
            })
        
        # 6. Sensor Fusion (if LiDAR available)
        if lidar_data is not None:
            print("  Fusing with LiDAR...")
            fused_detections = self.fusion.fuse_camera_lidar(
                all_detections, lidar_data, frame.shape
            )
            all_detections = fused_detections
        
        # 7. Tracking
        print("  Tracking objects...")
        tracks = self.tracker.deep_sort_tracking(all_detections, frame_id)
        
        # 8. Kalman Filter
        filtered_tracks = self.fusion.kalman_filter_fusion(
            list(tracks.values())
        )
        
        # 9. Trajectory Prediction
        predictions = self.tracker.predict_trajectories(
            {k: v for k, v in tracks.items() if 'bbox_history' in v}
        )
        
        # 10. BEV Representation
        print("  Creating BEV map...")
        bev_map = self.bev.create_bev_map(filtered_tracks)
        
        # 11. Path Planning
        path, occupancy = self.bev.plan_path(bev_map)
        
        # Compile results
        results = {
            'frame_id': frame_id,
            'original_frame': frame,
            'lanes': lanes,
            'edges': edges,
            'detections': all_detections,
            'tracks': tracks,
            'filtered_tracks': filtered_tracks,
            'predictions': predictions,
            'segmentation': seg_map,
            'depth': depth_map,
            'bev': bev_map,
            'path': path,
            'occupancy': occupancy
        }
        
        self.detection_history.append(results)
        
        return results
    
    def _simulate_depth(self, frame):
        """Simulate depth map for demo"""
        # In practice, use Monodepth2 or stereo
        height, width = frame.shape[:2]
        
        # Create simple depth gradient
        y_coords = np.arange(height).reshape(-1, 1)
        depth = 1.0 + (y_coords / height) * 50  # Closer at top, farther at bottom
        
        # Add some noise
        depth += np.random.randn(height, width) * 0.5
        
        # Normalize for visualization
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
        
        return depth_colormap
    
    def visualize_results(self, results):
        """
        Create visualization of all pipeline outputs
        """
        frame = results['original_frame'].copy()
        
        # 1. Draw lanes
        for lane in results['lanes'][0]:  # Left lanes
            cv2.line(frame, (int(lane[0]), int(lane[1])), 
                    (int(lane[2]), int(lane[3])), (0, 0, 255), 3)
        
        for lane in results['lanes'][1]:  # Right lanes
            cv2.line(frame, (int(lane[0]), int(lane[1])), 
                    (int(lane[2]), int(lane[3])), (255, 0, 0), 3)
        
        # 2. Draw detections
        for det in results['detections']:
            bbox = det['bbox']
            color = (0, 255, 0) if det['source'] == 'yolo' else (255, 255, 0)
            cv2.rectangle(frame, 
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])),
                         color, 2)
            
            # Add label
            label = f"{det.get('class', 'obj')}: {det.get('confidence', 0):.2f}"
            cv2.putText(frame, label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 3. Draw tracks
        for track_id, track in results['tracks'].items():
            if 'bbox' in track:
                bbox = track['bbox']
                cv2.rectangle(frame,
                             (int(bbox[0]), int(bbox[1])),
                             (int(bbox[2]), int(bbox[3])),
                             (255, 0, 255), 2)
                
                # Draw track ID
                cv2.putText(frame, f"ID: {track_id}",
                           (int(bbox[0]), int(bbox[1]) - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # 4. Draw predicted trajectories
        for track_id, pred in results['predictions'].items():
            if track_id in results['tracks']:
                current_bbox = results['tracks'][track_id]['bbox']
                cx = (current_bbox[0] + current_bbox[2]) / 2
                cy = (current_bbox[1] + current_bbox[3]) / 2
                
                # Draw future positions
                for i, future_bbox in enumerate(pred['future_bboxes']):
                    fx = (future_bbox[0] + future_bbox[2]) / 2
                    fy = (future_bbox[1] + future_bbox[3]) / 2
                    
                    # Color fades with time
                    alpha = 1.0 - (i / len(pred['future_bboxes']))
                    color = (int(255 * alpha), 0, int(255 * (1 - alpha)))
                    
                    cv2.circle(frame, (int(fx), int(fy)), 3, color, -1)
        
        # Create composite visualization
        composite = self._create_composite_display(results, frame)
        
        return composite
    
    def _create_composite_display(self, results, annotated_frame):
        """
        Create 2x2 grid display of different outputs
        """
        # Resize all images to same size
        size = (640, 480)
        
        # Original frame with annotations
        frame_display = cv2.resize(annotated_frame, size)
        
        # Segmentation map
        seg_display = cv2.resize(results['segmentation'], size)
        
        # Depth map
        depth_display = cv2.resize(results['depth'], size)
        
        # BEV map
        bev_display = cv2.resize(results['bev'], size)
        
        # Create 2x2 grid
        top_row = np.hstack([frame_display, seg_display])
        bottom_row = np.hstack([depth_display, bev_display])
        composite = np.vstack([top_row, bottom_row])
        
        # Add titles
        titles = ['Detection & Tracking', 'Semantic Segmentation',
                 'Depth Estimation', 'Bird\'s Eye View']
        
        positions = [(10, 30), (650, 30), (10, 510), (650, 510)]
        for title, pos in zip(titles, positions):
            cv2.putText(composite, title, pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return composite
    
    def run_on_video(self, video_path, output_path='output.mp4', max_frames=100):
        """
        Run pipeline on video
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        out_size = (1280, 960)  # 2x2 grid size
        
        # Define the codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, out_size)

        
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            print(f"\n=== Processing Frame {frame_count} ===")
            
            # Process frame
            results = self.process_frame(frame, frame_count)
            
            # Visualize
            composite = self.visualize_results(results)
            
            # Write to output
            out.write(composite)
            
            # Display
            cv2.imshow('Autonomous Perception', composite)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete. Output saved to {output_path}")
        print(f"Processed {frame_count} frames")
        
        return frame_count
