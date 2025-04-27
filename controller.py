import serial
import time

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
    

class ESPController:
    def __init__(self):
        self.accel_serial = None
        self.steer_serial = None
        self.brake_serial = None

        self.current_steering = None
        self.accel_active = None
        self.brake_active = None

        self._initialize_connections()

    def _initialize_connections(self):
        try:
            self.accel_serial = serial.Serial('/dev/ttyUSB2', 115200, timeout=1)
            time.sleep(2)
        except Exception as e:
            print(f"Failed to initialize acceleration ESP: {e}")

        try:
            self.steer_serial = serial.Serial('/dev/ttyUSB1', 115200, timeout=1)
            time.sleep(2)
        except Exception as e:
            print(f"Failed to initialize steering ESP: {e}")

        try:
            self.brake_serial = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            time.sleep(2)
        except Exception as e:
            print(f"Failed to initialize brake ESP: {e}")

    def set_steering(self, angle):
        if angle != self.current_steering:
            self.current_steering = angle
            if self.steer_serial:
                try:
                    self.steer_serial.write(f"{angle}\n".encode())
                except Exception as e:
                    print(f"Steering control error: {e}")

    def set_acceleration(self, active):
        if active != self.accel_active:
            self.accel_active = active
            if self.accel_serial:
                try:
                    self.accel_serial.write(b'1' if active else b'0')
                except Exception as e:
                    print(f"Acceleration control error: {e}")

    def set_brake(self, active):
        if active != self.brake_active:
            self.brake_active = active
            if self.brake_serial:
                try:
                    self.brake_serial.write(b'1' if active else b'0')
                except Exception as e:
                    print(f"Brake control error: {e}")

    def emergency_stop(self):
        self.set_brake(True)
        self.set_acceleration(False)

    def shutdown(self):
        self.set_acceleration(False)
        self.set_brake(False)
        self.set_steering(0)

        if self.accel_serial:
            self.accel_serial.close()
        if self.steer_serial:
            self.steer_serial.close()
        if self.brake_serial:
            self.brake_serial.close()
