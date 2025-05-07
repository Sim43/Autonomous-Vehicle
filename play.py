import cv2
import argparse
import sys
import csv
import os
from datetime import datetime
sys.path.append('lane_finder_v1')
sys.path.append('lane_finder_v2')
from lane_finder_v1.lane_detector import LaneDetector
from lane_finder_v2.lane_finder import LaneFinder
from obj_detector import ObjectDetector
from controller import PIDController, ESPController

class DriveLogger:
    def __init__(self, enable_logging=False):
        self.enable_logging = enable_logging
        self.log_file = None
        self.csv_writer = None
        self.log_dir = "realtime_logs"
        
        if self.enable_logging:
            self._initialize_logging()
    
    def _initialize_logging(self):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self.log_dir, f"drive_log_{timestamp}.csv")
            
            self.log_file = open(log_path, 'w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            
            # Write header with additional ESP state information
            header = [
                "timestamp",
                "steering_angle",
                "accel_active",
                "brake_active",
                "esp_steering",
                "esp_accel",
                "esp_brake",
                "lane_offset",
                "object_detected",
                "closest_object_distance"
            ]
            self.csv_writer.writerow(header)
            self.log_file.flush()
        except Exception as e:
            print(f"Failed to initialize logging: {e}")
            self.enable_logging = False
    
    def log_drive_state(self, steering_angle, accel, brake, esp_controller=None, 
                      lane_offset=None, detections=None):
        if not self.enable_logging:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            # Get ESP state if available
            esp_steering = esp_controller.current_steering if esp_controller else None
            esp_accel = esp_controller.accel_active if esp_controller else None
            esp_brake = esp_controller.brake_active if esp_controller else None
            
            # Process object detection info
            object_detected = bool(detections) if detections is not None else None
            closest_distance = min([d['distance'] for d in detections]) if detections else None
            
            row = [
                timestamp,
                int(steering_angle),
                int(accel),
                int(brake),
                esp_steering,
                esp_accel,
                esp_brake,
                lane_offset,
                object_detected,
                closest_distance
            ]
            
            self.csv_writer.writerow(row)
            self.log_file.flush()  # Ensure data is written immediately
        except Exception as e:
            print(f"Logging error: {e}")
    
    def close(self):
        if self.log_file:
            try:
                self.log_file.close()
            except:
                pass

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

def main(source, use_esp, ldv, reverse):
    if ldv == 1:
        lane_detector = LaneDetector()
    else: 
        lane_detector = LaneFinder()
    object_detector = ObjectDetector()
    esp_controller = ESPController() if use_esp else None
    drive_logger = DriveLogger(enable_logging=use_esp)

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
                if ldv == 1: 
                    frame_with_lanes, offset = lane_detector.process_image(frame)
                else: 
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_with_lanes, offset, found = lane_detector.process_image(frame_rgb, reset=False, show_period=0)
            except AssertionError as e:
                print("Lane detection skipped:", e)
                frame_with_lanes = frame
                offset = 0
                found = False

            steering_angle = offset * 300
            if reverse:
                steering_angle *= -1

            if use_esp:
                esp_controller.set_steering(steering_angle)

            detections = object_detector.detect_objects(frame)

            accel = True
            object_too_close = any(det['distance'] <= 4.0 for det in detections) if detections else False

            if use_esp:
                if object_too_close or (ldv == 2 and not found):
                    accel = False
                    esp_controller.set_brake(True)
                    esp_controller.set_acceleration(False, reverse)
                else:
                    esp_controller.set_acceleration(True, reverse)
                    esp_controller.set_brake(False)

            frame_with_all = draw_detections(frame_with_lanes, detections)
            frame_with_all = draw_driving_info(frame_with_all, steering_angle, accel)

            # Log the driving state with additional context
            drive_logger.log_drive_state(
                steering_angle=steering_angle,
                accel=accel,
                brake=not accel,
                esp_controller=esp_controller,
                lane_offset=offset,
                detections=detections
            )

            cv2.imshow("Lane + Object Detection", frame_with_all)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        drive_logger.close()
        
        if use_esp and esp_controller:
            esp_controller.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane and Object Detection System")
    parser.add_argument('--video', type=str, help="Path to video file or camera index (e.g. 0, 1, etc.)", required=True)
    parser.add_argument('--esp', action='store_true', help="Enable ESP controller and logging")
    parser.add_argument('--ldv', type=int, help="Use Lane version v1 or v2 (default v2)", default=2)
    parser.add_argument('--reverse', action='store_true', help="Reverse the steering angle direction")

    args = parser.parse_args()

    # Convert to integer if camera index was passed
    source = int(args.video) if args.video.isdigit() else args.video

    main(source, use_esp=args.esp, ldv=args.ldv, reverse=args.reverse)