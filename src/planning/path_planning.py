import numpy as np
import cv2

class BEVTransformer:
    def __init__(self, grid_size=0.2, grid_width=100, grid_height=100):
        """
        Bird's Eye View transformation
        """
        self.grid_size = grid_size  # meters per cell
        self.grid_width = grid_width  # cells
        self.grid_height = grid_height  # cells
        self.bev_center = (grid_width // 2, grid_height // 2)
        
    def create_bev_map(self, detections, ego_pose=None):
        """
        Create BEV occupancy grid from detections
        """
        # Initialize BEV grid
        bev_grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        
        # Draw ego vehicle at center
        ego_x, ego_y = self.bev_center
        cv2.circle(bev_grid, (ego_x, ego_y), 5, (0, 255, 0), -1)  # Green
        
        for det in detections:
            if 'center_3d' in det:
                x, y, z = det['center_3d']
                
                # Convert to BEV coordinates
                bev_x = int(ego_x + x / self.grid_size)
                bev_y = int(ego_y - y / self.grid_size)  # Note: y is forward
                
                # Check bounds
                if 0 <= bev_x < self.grid_width and 0 <= bev_y < self.grid_height:
                    # Get dimensions
                    if 'dimensions' in det:
                        width, height, length = det['dimensions']
                        bev_width = int(width / self.grid_size)
                        bev_length = int(length / self.grid_size)
                        
                        # Draw rectangle
                        pt1 = (bev_x - bev_width//2, bev_y - bev_length//2)
                        pt2 = (bev_x + bev_width//2, bev_y + bev_length//2)
                        
                        # Color by class
                        color = self._get_class_color(det.get('class', 'unknown'))
                        cv2.rectangle(bev_grid, pt1, pt2, color, 2)
                        
                        # Add ID if available
                        if 'track_id' in det:
                            cv2.putText(bev_grid, str(det['track_id']),
                                      (bev_x, bev_y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add grid lines
        for i in range(0, self.grid_width, 10):
            cv2.line(bev_grid, (i, 0), (i, self.grid_height), (50, 50, 50), 1)
        for i in range(0, self.grid_height, 10):
            cv2.line(bev_grid, (0, i), (self.grid_width, i), (50, 50, 50), 1)
        
        return bev_grid
    
    def _get_class_color(self, class_name):
        """Get color for different object classes"""
        colors = {
            'car': (255, 0, 0),      # Blue
            'pedestrian': (0, 255, 0),  # Green
            'cyclist': (0, 0, 255),    # Red
            'truck': (255, 255, 0),    # Cyan
            'traffic_light': (0, 255, 255)  # Yellow
        }
        return colors.get(class_name, (255, 255, 255))  # White for unknown
    
    def plan_path(self, bev_grid, start=None, goal=None):
        """
        Simple path planning using A* algorithm
        """
        if start is None:
            start = self.bev_center
        if goal is None:
            goal = (self.bev_center[0], 10)  # 2 meters ahead
        
        # Convert BEV to binary occupancy grid
        occupancy = np.mean(bev_grid, axis=2) > 100
        occupancy = occupancy.astype(np.uint8)
        
        # Dilate obstacles for safety margin
        kernel = np.ones((3, 3), np.uint8)
        occupancy = cv2.dilate(occupancy, kernel, iterations=2)
        
        # A* path planning
        path = self._astar(occupancy, start, goal)
        
        return path, occupancy
    
    def _astar(self, grid, start, goal):
        """A* path planning algorithm"""
        import heapq
        
        # Directions: 8-connected
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # Cost for diagonal moves
        sqrt2 = np.sqrt(2)
        
        # Initialize
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < grid.shape[1] and 
                        0 <= neighbor[1] < grid.shape[0]):
                    continue
                
                # Check obstacle
                if grid[neighbor[1], neighbor[0]] == 1:
                    continue
                
                # Calculate cost
                cost = sqrt2 if dx != 0 and dy != 0 else 1
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def _heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

