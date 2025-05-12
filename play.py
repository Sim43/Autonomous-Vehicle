import cv2
import argparse
from lane_finder import LaneFinder
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

def main(source, use_esp, reverse):
    lane_detector = LaneFinder()
    object_detector = ObjectDetector()
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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_with_lanes, offset, found = lane_detector.process_image(frame_rgb, reset=False, show_period=0)
            except AssertionError as e:
                print("Lane detection skipped:", e)
                frame_with_lanes = frame
                offset = 0
                found = False

            # Get detections
            detections = object_detector.detect_objects(frame)
            object_too_close = any(det['distance'] <= 4.0 for det in detections) if detections else False

            # Default driving behavior
            steering_angle = offset * 300
            if reverse:
                steering_angle *= -1
            accel = not object_too_close and found

            # Apply controls to ESP if enabled
            if use_esp:
                esp_controller.set_steering(steering_angle)
                if not accel:
                    esp_controller.set_brake(True)
                    esp_controller.set_acceleration(False, reverse)
                else:
                    esp_controller.set_acceleration(True, reverse)
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
    parser.add_argument('--video', type=str, help="Path to video file or camera index (e.g. 0, 1, etc.)", required=True)
    parser.add_argument('--esp', action='store_true', help="Enable ESP controller")
    parser.add_argument('--reverse', action='store_true', help="Reverse the steering angle direction")

    args = parser.parse_args()

    # Convert to integer if camera index was passed
    source = int(args.video) if args.video.isdigit() else args.video

    main(source, use_esp=args.esp, reverse=args.reverse)
