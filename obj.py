from ultralytics import YOLO

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