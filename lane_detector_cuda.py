import numpy as np
import cv2
import pickle

class LaneDetector:
    def __init__(self):
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("CUDA is available. Device Name:", cv2.cuda.getDevice())
        else:
            print("CUDA is not available in OpenCV.")

        self.window_search = True
        self.frame_count = 0
        self.left_fit_prev = None
        self.right_fit_prev = None
        self.curve_radius = 0
        self.offset = 0
        
        # Load camera calibration
        self.camera = pickle.load(open("models/camera_matrix.pkl", "rb"))
        self.mtx = self.camera['mtx']
        self.dist = self.camera['dist']
        self.camera_img_size = self.camera['imagesize']

        # Perspective matrices
        self.perspective_transform, self.inverse_perspective_transform = self._compute_perspective()

        # Pre-create filters
        self.gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (3, 3), 0)
        self.sobel_filter_x = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0, ksize=3)
        self.sobel_filter_y = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1, ksize=3)

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

    def sobel_operations(self, gray_gpu):
        sobelx = self.sobel_filter_x.apply(gray_gpu)
        sobely = self.sobel_filter_y.apply(gray_gpu)
        return sobelx, sobely
    
    def abs_sobel_thresh(self, sobel_gpu, thresh=(0, 255)):
        sobel_abs_gpu = cv2.cuda.abs(sobel_gpu)
        sobel_abs = sobel_abs_gpu.download()
        scaled_sobel = np.uint8(255 * sobel_abs / np.max(sobel_abs))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output
    
    def mag_threshold(self, sobelx_gpu, sobely_gpu, thresh=(0, 255)):
        sobelx = sobelx_gpu.download()
        sobely = sobely_gpu.download()
        mag = np.sqrt(sobelx**2 + sobely**2)
        scale = np.max(mag)/255 if np.max(mag) > 0 else 1
        mag_scaled = (mag/scale).astype(np.uint8)
        binary_output = np.zeros_like(mag_scaled)
        binary_output[(mag_scaled >= thresh[0]) & (mag_scaled <= thresh[1])] = 1
        return binary_output
    
    def dir_threshold(self, sobelx_gpu, sobely_gpu, thresh=(0, np.pi/2)):
        sobelx = sobelx_gpu.download()
        sobely = sobely_gpu.download()
        abs_grad_dir = np.arctan2(np.abs(sobely), np.abs(sobelx))
        binary_output = np.zeros_like(abs_grad_dir)
        binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1
        return binary_output

    def hls_select(self, img_gpu, sthresh=(0, 255), lthresh=(0, 255)):
        hls_gpu = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_RGB2HLS)
        hls = hls_gpu.download()
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1]) &
                      (l_channel >= lthresh[0]) & (l_channel <= lthresh[1])] = 1
        return binary_output

    def binary_pipeline(self, img):
        img_gpu = cv2.cuda_GpuMat()
        img_gpu.upload(img)

        img_blur_gpu = self.gaussian_filter.apply(img_gpu)

        # Create multiple CUDA streams
        stream1 = cv2.cuda.Stream()
        stream2 = cv2.cuda.Stream()
        stream3 = cv2.cuda.Stream()
        stream4 = cv2.cuda.Stream()
        stream5 = cv2.cuda.Stream()

        # Launch operations in parallel streams
        gray_gpu = cv2.cuda.cvtColor(img_blur_gpu, cv2.COLOR_RGB2GRAY, stream=stream1)

        sobelx = self.sobel_filter_x.apply(gray_gpu, stream=stream2)
        sobely = self.sobel_filter_y.apply(gray_gpu, stream=stream3)
        
        hls_gpu = cv2.cuda.cvtColor(img_blur_gpu, cv2.COLOR_RGB2HLS, stream=stream4)

        # Wait for operations to finish
        stream1.waitForCompletion()
        stream2.waitForCompletion()
        stream3.waitForCompletion()
        stream4.waitForCompletion()

        # Continue after synchronization
        s_binary = self.hls_select(hls_gpu.download(), sthresh=(140, 255), lthresh=(120, 255))
        x_binary = self.abs_sobel_thresh(sobelx)
        y_binary = self.abs_sobel_thresh(sobely)
        mag_binary = self.mag_threshold(sobelx, sobely)
        dir_binary = self.dir_threshold(sobelx, sobely)

        combined = np.zeros_like(s_binary)
        combined[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        s_binary_gpu = cv2.cuda_GpuMat()
        combined_gpu = cv2.cuda_GpuMat()
        s_binary_gpu.upload(s_binary)
        combined_gpu.upload(combined)

        final_binary_gpu = cv2.cuda.bitwise_or(s_binary_gpu, combined_gpu)
        final_binary = final_binary_gpu.download()

        return final_binary

    
    def warp_image(self, img):
        img_gpu = cv2.cuda_GpuMat()
        img_gpu.upload(img)
        warped_gpu = cv2.cuda.warpPerspective(img_gpu, self.perspective_transform, (img.shape[1], img.shape[0]))
        return warped_gpu.download(), self.inverse_perspective_transform

    def track_lanes_initialize(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 9
        window_height = binary_warped.shape[0] // nwindows
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
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
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
    
    def vehicle_offset(self, img, left_fit, right_fit):
        xm_per_pix = 3.7/700
        image_center = img.shape[1]/2
        
        left_low = self.get_val(img.shape[0], left_fit)
        right_low = self.get_val(img.shape[0], right_fit)
        
        lane_center = (left_low + right_low)/2.0
        distance = image_center - lane_center
        
        return round(distance*xm_per_pix, 5)
    
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
