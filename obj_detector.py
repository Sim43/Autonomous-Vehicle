from ultralytics import YOLO
import torch
import logging

# Suppress all YOLO-related logging
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

class ObjectDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device.upper()}")

        self.model = YOLO('models/yolov8n.pt')
        self.class_name_to_id = {'car': 2, 'person': 0, 'stop sign': 11}
        self.model.to(self.device)

        self.ref_sizes = {'car': 2.0, 'person': 0.5, 'stop sign': 0.6}

    def detect_objects(self, frame):
        class_ids = list(self.class_name_to_id.values())
        results = self.model.predict(frame, device=self.device, classes=class_ids, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            if class_name in self.ref_sizes:
                width = x2 - x1
                distance = (self.ref_sizes[class_name] * 1000) / width
                detections.append({
                    'class': class_name,
                    'bbox': (x1, y1, x2, y2),
                    'distance': distance
                })
        return detections
