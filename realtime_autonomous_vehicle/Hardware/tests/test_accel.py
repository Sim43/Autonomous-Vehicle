# test_accel.py
import serial
import time

ser = serial.Serial('/tmp/accel', 115200, timeout=1)
time.sleep(2)  # Give some time for ESP32 to reset

try:
    while True:
        cmd = input("Enter 1 to start motor, 0 to stop or 2 to reverse: ").strip()
        if cmd in ['1', '0', '2']:
            ser.write(cmd.encode())
        else:
            print("Invalid input. Enter only 1 or 0 or 2.")

except KeyboardInterrupt:
    print("\nExiting...")
    ser.close()
