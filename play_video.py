import cv2
from lane_detector import LaneDetector
from video_processor import VideoProcessor

def main():
    # Initialize lane detector
    lane_detector = LaneDetector()
    
    # Initialize video processor with lane detector
    video_processor = VideoProcessor(lane_detector)
    
    # Process video file
    input_path = "videos/car.mp4"
    output_path = "output/processed_video.mp4"  # Set to None if you don't want to save
    
    try:
        # Open and process video
        video_processor.open_video(input_path)
        
        # Process with display and optional saving
        video_processor.process_frames(display=True, output_path=output_path)
        
        # Alternative: Process single frame from camera
        # cap = cv2.VideoCapture(0)  # Webcam
        # ret, frame = cap.read()
        # if ret:
        #     processed_frame = video_processor.process_frame(frame)
        #     cv2.imshow('Processed Frame', processed_frame)
        #     cv2.waitKey(0)
        # cap.release()
        
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()