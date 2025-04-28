import pickle
import numpy as np

def main():
    # Your calibration data
    calib_data = {
        'cam_matrix': [
            [1.15693581e+03, 0.00000000e+00, 6.65950782e+02],
            [0.00000000e+00, 1.15213336e+03, 3.88784060e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ],
        'dist_coeffs': [
            [-2.37640816e-01, -8.53716908e-02, -7.90993807e-04,
             -1.15659392e-04,  1.05673884e-01]
        ],
        'img_size': (720, 1280, 3)
    }

    # Create new dictionary in the correct format
    converted_data = {
        'mtx': np.array(calib_data['cam_matrix']),    # ensure it's a numpy array
        'dist': np.array(calib_data['dist_coeffs']),  # ensure it's a numpy array
        'imagesize': (1280, 720)                      # fixed correctly
    }

    # Save to pkl file
    with open('own_camera.pkl', 'wb') as f:
        pickle.dump(converted_data, f)

    print("Calibration data converted and saved to 'own_camera.pkl'.")

if __name__ == "__main__":
    main()
