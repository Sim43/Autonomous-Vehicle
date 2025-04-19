import cv2
import argparse
from lane_detector import LaneDetector
from obj_detector import ObjectDetector

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} ({int(det['distance'])}m)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame

def main(source):
    lane_detector = LaneDetector()
    object_detector = ObjectDetector()

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame_with_lanes = lane_detector.process_image(frame)
        except AssertionError as e:
            print("Lane detection skipped:", e)
            frame_with_lanes = frame

        detections = object_detector.detect_objects(frame)
        frame_with_all = draw_detections(frame_with_lanes, detections)

        cv2.imshow("Lane + Object Detection", frame_with_all)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane and Object Detection System")
    parser.add_argument('--video', type=str, help="Path to video file (overrides camera)", default=None)
    parser.add_argument('--camera-index', type=int, help="Camera index to use (e.g., 0 for default, 1 for USB camera)", default=0)

    args = parser.parse_args()

    if args.video:
        source = args.video
    else:
        source = args.camera_index

    main(source)
