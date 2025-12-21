# 🚗 Real-Time Autonomous Vehicle System

A real-time autonomous vehicle control system that combines computer vision (lane detection and object detection) with ESP32-based motor control for autonomous driving capabilities.

---

## 📁 Directory Structure

```
realtime_autonomous_vehicle/
├── play.py                    # Main entry point - runs the autonomous driving system
├── requirements.txt           # Python dependencies
├── warning.png               # Warning icon displayed when lanes are lost
│
├── camera_pkls/              # Camera calibration data (REQUIRED)
│   ├── calib.p              # Camera calibration matrix and distortion coefficients
│   └── maps.p               # Perspective transform matrix and pixel-to-meter conversion
│
└── Hardware/                 # ESP32 motor control components
    ├── Arduino_Codes/        # Firmware for ESP32 modules
    │   ├── accel.ino        # Acceleration module firmware
    │   ├── brake.ino        # Braking module firmware
    │   ├── steer.ino        # Steering module firmware
    │   └── speed.ino        # Speed sensor firmware
    │
    ├── tests/                # Hardware testing utilities
    │   ├── detect_ports.py  # Auto-detect and create ESP32 port symlinks
    │   ├── test_accel.py    # Test acceleration module
    │   ├── test_brake.py    # Test braking module
    │   └── test_steer.py    # Test steering module
    │
    └── notes/                # Hardware documentation
        ├── ports.txt        # Port configuration notes
        └── other.txt        # Additional hardware notes
```

---

## 🔄 System Workflow

```
Camera/Video Input
    ↓
[1] Undistort Image (using calib.p)
    ↓
[2] Perspective Transform (using maps.p)
    ↓
[3] Lane Detection → Compute offset from center
    ↓
[4] Object Detection (YOLO) → Calculate distances to obstacles
    ↓
[5] Decision Logic → Determine steering, acceleration, braking
    ↓
[6] ESP32 Control (UART) → Send commands to motor controllers
    ↓
Motor Control Output
```

### Key Components:

1. **Lane Detection (`LaneFinder` class)**
   - Undistorts camera feed using calibration data
   - Applies perspective transform to bird's-eye view
   - Detects left and right lane lines using color thresholding and polynomial fitting
   - Calculates vehicle's offset from lane center

2. **Object Detection (`ObjectDetector` class)**
   - Uses YOLOv8 model to detect cars, pedestrians, signs, and traffic lights
   - Estimates distance to detected objects using bounding box width
   - Flags objects too close (within 4 meters) for braking decisions

3. **Decision Logic**
   - Steering angle: proportional to lane offset
   - Acceleration: enabled when lanes are detected AND no objects are too close
   - Braking: activated when obstacles are detected within safety distance

4. **ESP32 Control (`ESPController` class)**
   - Communicates with ESP32 modules via serial UART
   - Sends steering, acceleration, and braking commands
   - Uses symlinks at `/tmp/steer`, `/tmp/accel`, `/tmp/brake`

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Camera Calibration (Required)

**If you are using a camera**, you **must** calibrate it first to generate the required calibration files:

- `camera_pkls/calib.p` - Contains camera matrix and distortion coefficients
- `camera_pkls/maps.p` - Contains perspective transform matrix and pixel-to-meter conversion

**How to calibrate:**
1. Follow an online camera calibration tutorial (e.g., OpenCV camera calibration tutorials)
2. Use a checkerboard pattern to capture calibration images
3. Generate and save the calibration data to `camera_pkls/calib.p`
4. Generate the perspective transform data and save to `camera_pkls/maps.p`

The system requires these files to function properly. Without them, lane detection will fail.

### 3. Testing with Video Files

To test the system with a video file:

1. **Create a `videos` folder** in the project root:
   ```bash
   mkdir videos
   ```

2. **Add your video file** to the `videos` folder (e.g., `videos/test_video.mp4`)

3. Run the system with the video path:
   ```bash
   python play.py --video videos/test_video.mp4
   ```

### 4. Hardware Setup (Optional)

If you want to control physical motors via ESP32:

1. **Upload firmware** to your ESP32 modules:
   - Upload `Hardware/Arduino_Codes/accel.ino` to acceleration ESP32
   - Upload `Hardware/Arduino_Codes/brake.ino` to braking ESP32
   - Upload `Hardware/Arduino_Codes/steer.ino` to steering ESP32

2. **Detect and configure ESP32 ports**:
   ```bash
   python Hardware/tests/detect_ports.py
   ```
   This creates symlinks at `/tmp/steer`, `/tmp/accel`, and `/tmp/brake` for consistent port access.

3. **Test individual modules** (optional):
   ```bash
   python Hardware/tests/test_accel.py
   python Hardware/tests/test_brake.py
   python Hardware/tests/test_steer.py
   ```

---

## ▶️ Running the System

### Basic Usage

**With camera:**
```bash
python play.py --video 0
```
(Replace `0` with your camera index, e.g., `1`, `2`, etc.)

**With video file:**
```bash
python play.py --video videos/your_video.mp4
```

**With ESP32 motor control enabled:**
```bash
python play.py --video 0 --esp
```

**Reverse steering direction** (if needed):
```bash
python play.py --video 0 --esp --reverse
```

### Command-Line Arguments

- `--video` (required): Camera index (integer like `0`, `1`) or path to video file
- `--esp`: Enable ESP32 motor controller communication
- `--reverse`: Reverse the steering angle direction

---

## 📊 Features

- **Advanced Lane Detection**
  - Real-time lane line detection using polynomial fitting
  - Handles various lighting and road conditions
  - Visual feedback with lane overlays and warning indicators

- **Object Detection & Safety**
  - YOLOv8-based detection of vehicles, pedestrians, signs, and traffic lights
  - Distance estimation for collision avoidance
  - Automatic braking when objects are too close

- **Autonomous Decision Making**
  - Dynamic steering based on lane position
  - Intelligent acceleration/braking based on obstacles
  - Smooth control transitions

- **Hardware Integration**
  - UART communication with ESP32 microcontrollers
  - Modular control system for steering, acceleration, and braking

---

## ⚠️ Important Notes

- **Camera calibration is mandatory** when using a physical camera. The system will not work properly without `camera_pkls/calib.p` and `camera_pkls/maps.p`.
- Video files can be placed in a `videos/` folder for organized testing.
- When using ESP32 modules, ensure they are properly connected and the port detection script has been run.
- The system uses CUDA for object detection if available, otherwise falls back to CPU.
- Press `q` or `ESC` to exit the application safely.

---

## 🔧 Troubleshooting

- **"FileNotFoundError: camera_pkls/calib.p"**: Calibrate your camera and generate the required calibration files.
- **ESP32 connection errors**: Run `Hardware/tests/detect_ports.py` to re-detect and configure ports.
- **Poor lane detection**: Ensure good lighting conditions and that calibration data matches your camera setup.
- **Video file not found**: Check that the video path is correct and the file exists.
