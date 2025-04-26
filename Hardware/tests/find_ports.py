import serial
import time
import glob

# Find all ttyUSB devices
ports = glob.glob('/dev/ttyUSB*')

detected = {}

for port in ports:
    try:
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(1)  # Wait for ESP to boot and send message

        if ser.in_waiting:
            message = ser.readline().decode('utf-8', errors='ignore').strip()
            detected[port] = message
        else:
            detected[port] = 'No message received'

        ser.close()

    except Exception as e:
        detected[port] = f"Error: {e}"

# Print nicely
print("\nDetected Devices:")
for port, id_message in detected.items():
    print(f"{port}: {id_message}")
