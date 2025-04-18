import cv2

class VideoProcessor:
    def __init__(self, lane_detector):
        self.lane_detector = lane_detector
        self.cap = None
    
    def open_video(self, input_path):
        """Open a video file for processing"""
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file")
        return self
    
    def process_frames(self, display=True, output_path=None):
        """Process each frame of the opened video"""
        if self.cap is None:
            raise RuntimeError("No video opened. Call open_video() first")
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            writer = cv2.VideoWriter(output_path, 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   fps, 
                                   (frame_width, frame_height))
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame = self.lane_detector.process_image(frame)
            
            # Write to output file if specified
            if writer:
                writer.write(processed_frame)
            
            # Display the frame if requested
            if display:
                cv2.imshow('Lane Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Clean up
        if writer:
            writer.release()
        self.cap.release()
        cv2.destroyAllWindows()
    
    def process_frame(self, frame):
        """Process a single frame"""
        return self.lane_detector.process_image(frame)