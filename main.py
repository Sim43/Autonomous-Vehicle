import cv2
from camera import Camera
from obj import ObjectDetector

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
