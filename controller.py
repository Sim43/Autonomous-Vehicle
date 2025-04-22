import threading
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
        # Initialize serial connections
        self.accel_serial = None
        self.steer_serial = None
        self.brake_serial = None
        
        # State variables
        self.current_steering = 0
        self.accel_active = False
        self.brake_active = False
        self.running = True
        
        # Initialize connections
        self._initialize_connections()
        
        # Start control threads
        self.accel_thread = threading.Thread(target=self._accel_control_loop)
        self.steer_thread = threading.Thread(target=self._steer_control_loop)
        self.brake_thread = threading.Thread(target=self._brake_control_loop)
        
        self.accel_thread.start()
        self.steer_thread.start()
        self.brake_thread.start()
    
    def _initialize_connections(self):
        try:
            self.accel_serial = serial.Serial('/dev/accel', 115200, timeout=1)
            time.sleep(2)
        except Exception as e:
            print(f"Failed to initialize acceleration ESP: {e}")
        
        try:
            self.steer_serial = serial.Serial('/dev/steer', 115200, timeout=1)
            time.sleep(2)
        except Exception as e:
            print(f"Failed to initialize steering ESP: {e}")
        
        try:
            self.brake_serial = serial.Serial('/dev/brake', 115200, timeout=1)
            time.sleep(2)
        except Exception as e:
            print(f"Failed to initialize brake ESP: {e}")
    
    def set_steering(self, angle):
        """Set the steering angle (-1800 to 1800)"""
        self.current_steering = max(min(angle, 1800), -1800)
    
    def set_acceleration(self, active):
        """Enable or disable acceleration"""
        if active and not self.brake_active:  # Can't accelerate while braking
            self.accel_active = active
            self.brake_active = False
    
    def set_brake(self, active):
        """Enable or disable braking"""
        if active and not self.accel_active:  # Can't brake while accelerating
            self.brake_active = active
            self.accel_active = False
    
    def _accel_control_loop(self):
        while self.running:
            if self.accel_serial:
                try:
                    cmd = '1' if self.accel_active else '0'
                    self.accel_serial.write(cmd.encode())
                except Exception as e:
                    print(f"Acceleration control error: {e}")
            time.sleep(0.1)
    
    def _steer_control_loop(self):
        while self.running:
            if self.steer_serial:
                try:
                    self.steer_serial.write(f"{self.current_steering}\n".encode())
                except Exception as e:
                    print(f"Steering control error: {e}")
            time.sleep(0.1)
    
    def _brake_control_loop(self):
        while self.running:
            if self.brake_serial:
                try:
                    cmd = '1' if self.brake_active else '0'
                    self.brake_serial.write(cmd.encode())
                except Exception as e:
                    print(f"Brake control error: {e}")
            time.sleep(0.1)
    
    def emergency_stop(self):
        """Immediately stop the vehicle"""
        self.brake_active = True
        self.accel_active = False
    
    def shutdown(self):
        """Clean shutdown of all controllers"""
        self.running = False
        self.accel_active = False
        self.brake_active = False
        
        # Wait for threads to finish
        self.accel_thread.join()
        self.steer_thread.join()
        self.brake_thread.join()
        
        # Close serial connections
        if self.accel_serial:
            self.accel_serial.close()
        if self.steer_serial:
            self.steer_serial.close()
        if self.brake_serial:
            self.brake_serial.close()