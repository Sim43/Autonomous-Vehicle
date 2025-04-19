# test_steer.py
import serial
import time

ser = serial.Serial('/dev/steer', 115200)  # Change COM port as needed
time.sleep(2)  # Wait for connection

angles = [1800, -1800, 0]
for angle in angles:
    print(f"Sending angle: {angle}")
    ser.write(f"{angle}\n".encode())
    time.sleep(1)  # Wait for action

ser.close()
