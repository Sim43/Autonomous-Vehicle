import cv2 

class Camera:
    def __init__(self, camera_index=None):
        if camera_index is None:
            camera_index = self._find_usb_camera()
        
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source at index {camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def _find_usb_camera(self):
        for i in range(1, 10):  # Skip 0 (laptop cam)
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return i
        raise ValueError("No working USB camera found (tried indices 1–9).")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        self.cap.release()