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

class ObjectDetector:
    def __init__(self):
        self.model = YOLO('models/yolov8n.pt')
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

def main():
    camera = Camera(0)
    object_detector = ObjectDetector()
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None: break
            
            objects = object_detector.detect_objects(frame)

            # Draw detected objects
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{obj['class']} {obj['distance']:.1f}m", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
