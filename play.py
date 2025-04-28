import cv2
import argparse
from lane_detector import LaneDetector
from obj_detector import ObjectDetector
from controller import PIDController, ESPController

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class']} ({int(det['distance'])}m)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame

def draw_driving_info(frame, steering_angle, accel):
    brake_or_accel = "Accel" if accel else "Brake"
    text = f"Steering: {int(steering_angle)} | {brake_or_accel}"
    org = (frame.shape[1] - 320, 40)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return frame

def find_available_camera(max_index=10):
    for index in range(1, max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
    return None

def main(source, use_esp):
    lane_detector = LaneDetector()
    object_detector = ObjectDetector()
    pid = PIDController(Kp=500, Ki=0.1, Kd=100)
    esp_controller = ESPController() if use_esp else None

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame_with_lanes, offset = lane_detector.process_image(frame)
            except AssertionError as e:
                print("Lane detection skipped:", e)
                frame_with_lanes = frame
                offset = 0

            steering_angle = pid.compute(offset)
            steering_angle = int(max(min(steering_angle, 1800), -1800))

            if use_esp:
                esp_controller.set_steering(steering_angle)

            detections = object_detector.detect_objects(frame)

            accel = True
            object_too_close = any(det['distance'] < 2.0 for det in detections)

            if use_esp:
                if object_too_close:
                    accel = False
                    esp_controller.set_brake(True)
                    esp_controller.set_acceleration(False)
                else:
                    esp_controller.set_acceleration(True)
                    esp_controller.set_brake(False)

            frame_with_all = draw_detections(frame_with_lanes, detections)
            frame_with_all = draw_driving_info(frame_with_all, steering_angle, accel)
            cv2.imshow("Lane + Object Detection", frame_with_all)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if use_esp and esp_controller:
            esp_controller.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane and Object Detection System")
    parser.add_argument('--video', type=str, help="Path to video file (overrides camera)", default=None)
    parser.add_argument('--use-esp', action='store_true', help="Enable ESP controller for sending commands")
    args = parser.parse_args()

    source = args.video if args.video else find_available_camera()
    if source is None:
        print("No available camera found.")
        exit(1)

    main(source, use_esp=args.use_esp)
