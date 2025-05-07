import serial
import glob
import time
import os

# Expected responses from each ESP
DEVICE_IDS = {
    "ACCELERATION_MODULE": "accel",
    "BRAKE_MODULE": "brake",
    "STEERING_MODULE": "steer"
}

# Maximum time to wait for a valid response
MAX_READ_TIME = 2.0  # seconds

def detect_devices():
    ports = glob.glob("/dev/ttyUSB*")
    detected = {}

    for port in ports:
        try:
            with serial.Serial(port, 115200, timeout=0.5) as ser:
                time.sleep(1)
                ser.reset_input_buffer()
                ser.write(b"f\n")

                start_time = time.time()
                while time.time() - start_time < MAX_READ_TIME:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"{port} -> '{line}'")
                        if line in DEVICE_IDS:
                            detected[DEVICE_IDS[line]] = port
                            break  # Found valid device, stop reading this port
        except Exception as e:
            print(f"Error on {port}: {e}")

    return detected

def create_symlinks(mapping):
    for name, port in mapping.items():
        link_path = f"/tmp/{name}"
        try:
            if os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(port, link_path)
            print(f"Created symlink: {link_path} -> {port}")
        except Exception as e:
            print(f"Failed to create symlink for {name}: {e}")

if __name__ == "__main__":
    mapping = detect_devices()
    print("\n=== Device Mapping ===")
    for name, port in mapping.items():
        print(f"{name}: {port}")
    create_symlinks(mapping)
