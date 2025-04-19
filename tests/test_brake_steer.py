import serial
import time

# Replace with your actual port
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # Wait for ESP32 to reset

# Example maneuver function
def perform_maneuver():
    angles = [30, -30, 0, 45, -45, 0]
    
    for angle in angles:
        msg = f"S:{angle}\n"
        print(f"Sending steering: {msg.strip()}")
        ser.write(msg.encode())
        time.sleep(1)

    print("Applying Brake")
    ser.write(b"B:1\n")
    time.sleep(2)

    print("Releasing Brake")
    ser.write(b"B:0\n")
    time.sleep(1)

try:
    perform_maneuver()

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    ser.close()
