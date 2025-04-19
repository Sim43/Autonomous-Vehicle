# test_brake.py
import serial
import time

ser = serial.Serial('COM3', 115200)  # Change COM port
time.sleep(2)

commands = ['1', '0', '1', '0']
for cmd in commands:
    print(f"Sending command: {cmd}")
    ser.write(cmd.encode())
    time.sleep(1.5)  # Simulate brake timing

ser.close()
