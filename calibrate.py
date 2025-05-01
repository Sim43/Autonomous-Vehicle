"""
This script is used to calibrate the camera based on the provided images
The distortion coefficients and camera matrix are saved for later reuse
"""
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle




def calibrate(filename, silent = False):
    images_path = os.path.abspath("/home/zero/Downloads/xD/camera_cal")
    n_x = 8
    n_y = 5

    # setup object points
    objp = np.zeros((n_y*n_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_x, 0:n_y].T.reshape(-1, 2)
    image_points = []
    object_points = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # loop through provided images
    for image_file in os.listdir(images_path):
        if image_file.endswith(".jpg"):
            img_path = os.path.join(images_path, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Failed to load image: {image_file}")
                continue

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(img_gray, (n_x, n_y), None)

            if found:
                corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners2)
                object_points.append(objp)

                # Use a writable copy for drawing
                img_draw = img.copy()
                cv2.drawChessboardCorners(img_draw, (n_x, n_y), corners2, found)

                if not silent:
                    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
                    plt.title(f"Corners in {image_file}")
                    plt.axis('off')
                    plt.show()
            else:
                print(f"❌ No corners found in {image_file}")


    # pefrorm the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_gray.shape[::-1], None, None)
    img_size  = (1280, 720)
    # pickle the data and save it
    calib_data = {'mtx':mtx,
                  'dist':dist,
                  'imagesize':img_size}
    with open(filename, 'wb') as f:
        pickle.dump(calib_data, f)

    if not silent:
        for image_file in os.listdir(images_path):
            if image_file.endswith("jpg"):
                # show distorted images
                img = mpimg.imread(os.path.join(images_path, image_file))
                plt.imshow(cv2.undistort(img, mtx, dist))
                plt.show()

    return mtx, dist

if __name__ == '__main__':
    calibrate('/home/zero/Downloads/xD/models/real_cam.pkl', False)







