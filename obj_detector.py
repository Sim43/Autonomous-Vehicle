from ultralytics import YOLO
import torch
import logging

# Suppress all YOLO-related logging
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

class ObjectDetector:
    def __init__(self):
        # Automatically select GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device.upper()}")

        self.model = YOLO('models/yolov8n.pt')
        self.model.to(self.device)
        self.classes_of_interest = ['car', 'person', 'stop sign']
    
    def detect_objects(self, frame):
        results = self.model(frame, device=self.device)
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
