import cv2
import numpy as np
from ultralytics import YOLO

class Camera:
    def __init__(self, camera_index=None):
        if camera_index is None:
            camera_index = self._find_usb_camera()
        
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source at index {camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def _find_usb_camera(self):
        for i in range(1, 10):  # Skip 0 (laptop cam)
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return i
        raise ValueError("No working USB camera found (tried indices 1–9).")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        self.cap.release()

class LaneDetector:
    def __init__(self):
        self.lane_center = None
        self.lane_width = None
        self.is_two_lanes = False
        self.road_polygon = None

    def detect_lanes(self, image):
        height, width = image.shape[:2]
        
        # Define a larger region of interest for road detection
        roi_vertices = [
            (0, height),
            (width/2 - width/4, height/2 + height/4),  # Wider base
            (width/2 + width/4, height/2 + height/4),  # Wider base
            (width, height)
        ]
        
        # Create a green polygon overlay for road detection visualization
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(roi_vertices, np.int32)], (0, 255, 0))
        alpha = 0.3  # Transparency factor
        self.road_polygon = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding for better lane detection
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(gray, 50, 150)  # Adjusted thresholds
        masked = self._region_of_interest(edges, np.array([roi_vertices], np.int32))

        # Detect lines with adjusted parameters
        lines = cv2.HoughLinesP(
            masked, rho=2, theta=np.pi/180, threshold=100,
            minLineLength=50, maxLineGap=30
        )

        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2-y1)/(x2-x1) if (x2-x1) != 0 else 0
                if abs(slope) < 0.3: continue  # Less strict slope threshold
                if slope < 0 and x1 < width/2 and x2 < width/2:
                    left_lines.append(line[0])
                elif slope > 0 and x1 > width/2 and x2 > width/2:
                    right_lines.append(line[0])

        min_y = int(height * 0.6)
        max_y = height

        left_line = self._fit_line(left_lines, min_y, max_y)
        right_line = self._fit_line(right_lines, min_y, max_y)

        if left_line and right_line:
            self.lane_center = (left_line[0] + right_line[0]) // 2
            self.lane_width = right_line[0] - left_line[0]
            self.is_two_lanes = True if self.lane_width > 200 else False  # Adjusted width threshold
        elif left_line:
            self.lane_center = left_line[0] + 300  # Increased offset for single lane
            self.is_two_lanes = False
        elif right_line:
            self.lane_center = right_line[0] - 300  # Increased offset for single lane
            self.is_two_lanes = False
        else:
            self.lane_center = None
            self.is_two_lanes = False

        return left_line, right_line

    def _fit_line(self, lines, min_y, max_y):
        if not lines: return None
        xs, ys = [], []
        for x1, y1, x2, y2 in lines:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        poly = np.poly1d(np.polyfit(ys, xs, deg=1))
        return [int(poly(max_y)), max_y, int(poly(min_y)), min_y]

    def _region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(img, mask)

    def get_lane_status(self):
        if self.lane_center is None:
            return "no lane"
        return "double lane" if self.is_two_lanes else "single lane"

class ObjectDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.classes_of_interest = ['car', 'person', 'stop sign']
    
    def detect_objects(self, frame):
        results = self.model(frame)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = self.model.names[int(box.cls[0])]
                if cls in self.classes_of_interest and box.conf[0] >= 0.5:
                    detections.append({
                        'class': cls,
                        'bbox': (x1, y1, x2, y2),
                        'distance': self._estimate_distance(x2-x1, cls)
                    })
        return detections
    
    def _estimate_distance(self, width, obj_class):
        ref_sizes = {'car': 2.0, 'person': 0.5, 'stop sign': 0.6}
        return (ref_sizes.get(obj_class, 2.0) * 1000) / width

class DecisionMaker:
    def __init__(self):
        self.current_action = "forward"
        self.obstacle_distance_threshold = 5.0  # meters
        self.safe_overtake_distance = 10.0
    
    def make_decision(self, lane_info, lane_detector, objects):
        left_line, right_line = lane_info
        center = lane_detector.lane_center
        frame_center = 640  # Assuming 1280 width

        # If no center line, brake
        if center is None:
            return "brake"

        close_obstacles = [obj for obj in objects if obj['distance'] < self.obstacle_distance_threshold]
        
        if close_obstacles:
            if lane_detector.is_two_lanes:
                if all(obj['bbox'][0] > center for obj in close_obstacles):
                    return "right"
                elif all(obj['bbox'][2] < center for obj in close_obstacles):
                    return "left"
            return "brake"
        
        if center < frame_center - 50:
            return "right"
        elif center > frame_center + 50:
            return "left"
        return "forward"

def main():
    camera = Camera()
    lane_detector = LaneDetector()
    object_detector = ObjectDetector()
    decision_maker = DecisionMaker()
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None: break
            
            lanes = lane_detector.detect_lanes(frame)
            objects = object_detector.detect_objects(frame)
            action = decision_maker.make_decision(lanes, lane_detector, objects)
            
            # Display lane status
            lane_status = lane_detector.get_lane_status()
            cv2.putText(frame, f"Lane: {lane_status}", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display action
            cv2.putText(frame, f"Action: {action}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw lane center if detected
            if lane_detector.lane_center:
                cv2.line(frame, (lane_detector.lane_center, 720), (lane_detector.lane_center, 600), 
                         (0, 255, 255), 3)
            
            # Draw detected objects
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{obj['class']} {obj['distance']:.1f}m", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            # Combine the frame with the road polygon overlay
            frame = cv2.addWeighted(frame, 0.7, lane_detector.road_polygon, 0.3, 0)
            
            cv2.imshow('Autonomous Driving', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()