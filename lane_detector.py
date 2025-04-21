import numpy as np
import cv2
import pickle

class LaneDetector:
    def __init__(self):
        # Initialize parameters
        self.window_search = True
        self.frame_count = 0
        self.left_fit_prev = None
        self.right_fit_prev = None
        
        # Load camera calibration
        self.camera = pickle.load(open("models/camera_matrix.pkl", "rb"))
        self.mtx = self.camera['mtx']
        self.dist = self.camera['dist']
        self.camera_img_size = self.camera['imagesize']
        
    def distort_correct(self, img):
        img_size = (img.shape[1], img.shape[0])
        assert (img_size == self.camera_img_size), 'image size is not compatible'
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist
    
    def abs_sobel_thresh(self, img, orient='x', thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output
    
    def mag_threshold(self, img, sobel_kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        mag = np.sqrt(x**2 + y**2)
        scale = np.max(mag)/255
        eightbit = (mag/scale).astype(np.uint8)
        binary_output = np.zeros_like(eightbit)
        binary_output[(eightbit > thresh[0]) & (eightbit < thresh[1])] = 1
        return binary_output
    
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        direction = np.arctan2(y, x)
        binary_output = np.zeros_like(direction)
        binary_output[(direction > thresh[0]) & (direction < thresh[1])] = 1
        return binary_output
    
    def hls_select(self, img, sthresh=(0, 255), lthresh=()):
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        L = hls_img[:,:,1]
        S = hls_img[:,:,2]
        binary_output = np.zeros_like(S)
        binary_output[(S >= sthresh[0]) & (S <= sthresh[1]) & (L > lthresh[0]) & (L <= lthresh[1])] = 1
        return binary_output
    
    def binary_pipeline(self, img):
        img_copy = cv2.GaussianBlur(img, (3, 3), 0)
        s_binary = self.hls_select(img_copy, sthresh=(140, 255), lthresh=(120, 255))
        x_binary = self.abs_sobel_thresh(img_copy, thresh=(25, 200))
        y_binary = self.abs_sobel_thresh(img_copy, thresh=(25, 200), orient='y')
        xy = cv2.bitwise_and(x_binary, y_binary)
        mag_binary = self.mag_threshold(img_copy, sobel_kernel=3, thresh=(30,100))
        dir_binary = self.dir_threshold(img_copy, sobel_kernel=3, thresh=(0.8, 1.2))
        gradient = np.zeros_like(s_binary)
        gradient[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        final_binary = cv2.bitwise_or(s_binary, gradient)
        return final_binary
    
    def warp_image(self, img):
        image_size = (img.shape[1], img.shape[0])
        x = img.shape[1]
        y = img.shape[0]
        
        source_points = np.float32([
            [0.117 * x, y],
            [(0.5 * x) - (x*0.078), (2/3)*y],
            [(0.5 * x) + (x*0.078), (2/3)*y],
            [x - (0.117 * x), y]
        ])
        
        destination_points = np.float32([
            [0.25 * x, y],
            [0.25 * x, 0],
            [x - (0.25 * x), 0],
            [x - (0.25 * x), y]
        ])
        
        perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
        inverse_perspective_transform = cv2.getPerspectiveTransform(destination_points, source_points)
        warped_img = cv2.warpPerspective(img, perspective_transform, image_size, flags=cv2.INTER_LINEAR)
        
        return warped_img, inverse_perspective_transform
    
    def track_lanes_initialize(self, binary_warped):
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        midpoint = int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 9
        window_height = int(binary_warped.shape[0]/nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
            win_y_high = int(binary_warped.shape[0] - window*window_height)
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        return left_fit, right_fit
    
    def track_lanes_update(self, binary_warped, left_fit, right_fit):
        if self.frame_count % 10 == 0:
            self.window_search = True
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                          (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        return left_fit, right_fit, leftx, lefty, rightx, righty
    
    def get_val(self, y, poly_coeff):
        return poly_coeff[0]*y**2 + poly_coeff[1]*y + poly_coeff[2]
    
    def lane_fill_poly(self, binary_warped, undist, left_fit, right_fit):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = self.get_val(ploty, left_fit)
        right_fitx = self.get_val(ploty, right_fit)
        
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        newwarp = cv2.warpPerspective(color_warp, self.inverse_perspective_transform, 
                                     (binary_warped.shape[1], binary_warped.shape[0]))
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result
    
    def process_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        undist = self.distort_correct(img)
        # Get binary image
        binary_img = self.binary_pipeline(undist)
        # Perspective transform
        birdseye, self.inverse_perspective_transform = self.warp_image(binary_img)
        
        if self.window_search:
            self.window_search = False
            left_fit, right_fit = self.track_lanes_initialize(birdseye)
            self.left_fit_prev = left_fit
            self.right_fit_prev = right_fit
        else:
            left_fit = self.left_fit_prev
            right_fit = self.right_fit_prev
            left_fit, right_fit, _, _, _, _ = self.track_lanes_update(birdseye, left_fit, right_fit)
        
        self.left_fit_prev = left_fit
        self.right_fit_prev = right_fit
        
        # Draw polygon
        processed_frame = self.lane_fill_poly(birdseye, undist, left_fit, right_fit)
        
        self.frame_count += 1
        # Convert back to BGR for OpenCV compatibility
        return cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)