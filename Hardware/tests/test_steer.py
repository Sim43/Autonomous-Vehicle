# test_steer.py
import serial
import time

ser = serial.Serial('/dev/ttyUSB1', 115200, timeout=1)  # Adjust COM port if needed
time.sleep(2)  # Give some time for ESP32 to reset

try:
    while True:
        cmd = input("Enter steering angle (e.g., 1800, -1800, 0): ").strip()
        try:
            angle = int(cmd)  # Try to convert to integer
            ser.write(f"{angle}\n".encode())
        except ValueError:
            print("Invalid input. Enter a valid integer angle.")

except KeyboardInterrupt:
    print("\nExiting...")
    ser.close()
