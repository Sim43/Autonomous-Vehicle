import cv2
import numpy as np

def calculate_curvature(poly_fit, y_eval):
    A = poly_fit[0]
    B = poly_fit[1]
    return ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / abs(2 * A + 1e-5)  # avoid div by 0

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(img):
    h, w = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (int(0.1 * w), h),
        (int(0.45 * w), int(0.6 * h)),
        (int(0.55 * w), int(0.6 * h)),
        (int(0.9 * w), h),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def get_lane_lines(edges, frame_shape):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=50)
    left_pts, right_pts = [], []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1 + 1e-5)
                if slope < -0.5:
                    left_pts += [(x1, y1), (x2, y2)]
                elif slope > 0.5:
                    right_pts += [(x1, y1), (x2, y2)]
    return np.array(left_pts), np.array(right_pts)

def draw_lane(frame, poly_fit, color):
    y_vals = np.linspace(0, frame.shape[0]-1, frame.shape[0])
    x_vals = poly_fit[0]*y_vals**2 + poly_fit[1]*y_vals + poly_fit[2]
    pts = np.array([np.stack([x_vals, y_vals], axis=1)], dtype=np.int32)
    cv2.polylines(frame, pts, isClosed=False, color=color, thickness=3)

cap = cv2.VideoCapture(0)  # Use correct camera index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess(frame)
    roi_edges = region_of_interest(edges)

    left_pts, right_pts = get_lane_lines(roi_edges, frame.shape)

    if len(left_pts) > 0 and len(right_pts) > 0:
        # Fit 2nd-degree polynomial
        left_fit = np.polyfit(left_pts[:,1], left_pts[:,0], 2)
        right_fit = np.polyfit(right_pts[:,1], right_pts[:,0], 2)
        
        # Draw lanes
        draw_lane(frame, left_fit, (255, 0, 0))
        draw_lane(frame, right_fit, (0, 0, 255))

        # Calculate curvature
        y_eval = frame.shape[0]  # bottom of image
        curv_left = calculate_curvature(left_fit, y_eval)
        curv_right = calculate_curvature(right_fit, y_eval)
        avg_curvature = (curv_left + curv_right) / 2

        # Display result
        cv2.putText(frame, f"Curvature: {int(avg_curvature)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if avg_curvature < 300:
            cv2.putText(frame, "⚠️ SHARP TURN AHEAD!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Sharp Turn Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
