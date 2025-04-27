import numpy as np
import cv2
import pickle

class LaneDetector:
    def __init__(self):
        self.window_search = True
        self.frame_count = 0
        self.left_fit_prev = None
        self.right_fit_prev = None
        self.curve_radius = 0
        self.offset = 0
        
        self.camera = pickle.load(open("models/camera_matrix.pkl", "rb"))
        self.mtx = self.camera['mtx']
        self.dist = self.camera['dist']
        self.camera_img_size = self.camera['imagesize']
        
        # Perspective matrices precomputed for speed
        self.perspective_transform, self.inverse_perspective_transform = self._compute_perspective()

    def _compute_perspective(self):
        x, y = self.camera_img_size
        src = np.float32([
            [0.117 * x, y],
            [(0.5 * x) - (x*0.078), (2/3)*y],
            [(0.5 * x) + (x*0.078), (2/3)*y],
            [x - (0.117 * x), y]
        ])
        dst = np.float32([
            [0.25 * x, y],
            [0.25 * x, 0],
            [x - 0.25 * x, 0],
            [x - 0.25 * x, y]
        ])
        return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)

    def distort_correct(self, img):
        img_size = (img.shape[1], img.shape[0])
        assert img_size == self.camera_img_size, 'image size mismatch'
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
    def sobel_operations(self, gray, sobel_kernel=3):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        return sobelx, sobely

    def abs_sobel_thresh(self, sobelx, sobely, orient='x', thresh=(0, 255)):
        abs_sobel = np.absolute(sobelx if orient == 'x' else sobely)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output
    
    def mag_threshold(self, sobelx, sobely, thresh=(0, 255)):
        mag = np.sqrt(sobelx**2 + sobely**2)
        scale = np.max(mag)/255 if np.max(mag) > 0 else 1
        mag_scaled = (mag/scale).astype(np.uint8)
        binary_output = np.zeros_like(mag_scaled)
        binary_output[(mag_scaled >= thresh[0]) & (mag_scaled <= thresh[1])] = 1
        return binary_output

    def dir_threshold(self, sobelx, sobely, thresh=(0, np.pi/2)):
        abs_grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(abs_grad_dir)
        binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1
        return binary_output
    
    def hls_select(self, img, sthresh=(0, 255), lthresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1]) & (l_channel >= lthresh[0]) & (l_channel <= lthresh[1])] = 1
        return binary_output
    
    def binary_pipeline(self, img):
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
        sobelx, sobely = self.sobel_operations(gray)

        s_binary = self.hls_select(img_blur, sthresh=(140, 255), lthresh=(120, 255))
        x_binary = self.abs_sobel_thresh(sobelx, sobely, orient='x', thresh=(25, 200))
        y_binary = self.abs_sobel_thresh(sobelx, sobely, orient='y', thresh=(25, 200))
        mag_binary = self.mag_threshold(sobelx, sobely, thresh=(30, 100))
        dir_binary = self.dir_threshold(sobelx, sobely, thresh=(0.8, 1.2))
        
        combined = np.zeros_like(s_binary)
        combined[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        
        final_binary = cv2.bitwise_or(s_binary, combined)
        return final_binary

    def warp_image(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.perspective_transform, img_size), self.inverse_perspective_transform

    def process_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        undist = self.distort_correct(img)
        binary_img = self.binary_pipeline(undist)
        birdseye, _ = self.warp_image(binary_img)

        if self.window_search:
            self.left_fit_prev, self.right_fit_prev = self.track_lanes_initialize(birdseye)
            self.window_search = False
        else:
            self.left_fit_prev, self.right_fit_prev, _, _, _, _ = self.track_lanes_update(birdseye, self.left_fit_prev, self.right_fit_prev)

        processed = self.lane_fill_poly(birdseye, undist, self.left_fit_prev, self.right_fit_prev)

        if self.frame_count % 15 == 0:
            self.offset = self.vehicle_offset(undist, self.left_fit_prev, self.right_fit_prev)

        self.frame_count += 1
        return cv2.cvtColor(processed, cv2.COLOR_RGB2BGR), self.offset
