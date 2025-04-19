#include <ESP32Servo.h>

// Pin Definitions
const int stepPin = 5;        // Stepper motor step pin (GPIO 5)
const int dirPin = 18;         // Stepper motor direction pin (GPIO 4)
const int servoPin = 13;      // Servo control pin (GPIO 13)
// Constants
const int stepsPerRevolution = 800;  // Full steps per revolution for stepper motor
const int stepDelayMicroseconds = 400;  // Delay between steps for speed control

// Objects
Servo brakeServo;              // Servo object for braking/acceleration control


void setup() {
  // Initialize Serial Communication
  Serial.begin(115200);       // UART baud rate

  // Set motor pins as outputs
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);

  // Initialize servo
  brakeServo.attach(servoPin);
  brakeServo.write(0);        // Start with the servo at 0 degrees (no brake)

  // Initialize stepper motor
  digitalWrite(stepPin, LOW);
 
  Serial.println("ESP32 System Initialized. Use 'space', 'a', 'd' for control.");
}

void loop() {
  // Check if UART data is available
  if (Serial.available() > 0) {
    char command = Serial.read();  // Read a single character command

    // Debugging - Print the received command
    Serial.print("Received command: ");
    Serial.println(command);

    // Perform actions based on commands
    switch (command) {
      case ' ':  // Brake
        handleBrake();
        break;

      case 'a':  // Steer left
        handleSteering(1);  // Left direction
        break;

      case 'd':  // Steer right
        handleSteering(0);  // Right direction
        break;

      default:
        Serial.println("Invalid command!");
        break;
    }

    delay(100); // Small delay for stability
  }
}

// Function to handle braking using servo
void handleBrake() {
  Serial.println("Braking...");
  brakeServo.write(180);  // Apply brake (servo to 180 degrees)
  delay(500);             // Small delay to simulate braking
  brakeServo.write(0);    // Release brake (servo to 0 degrees)
}

// Function to handle steering using stepper motor
void handleSteering(int direction) {
  if (direction == 1) {
    Serial.println("Steering left...");
    digitalWrite(dirPin, HIGH); // Set direction to left
  } else if (direction == 0) {
    Serial.println("Steering right...");
    digitalWrite(dirPin, LOW);  // Set direction to right
  }

  makeStepsSteering(stepsPerRevolution / 4); // Use steering-specific steps
}

// Function to make steps for Steering
void makeStepsSteering(int steps) {
  Serial.println("Executing Steering steps...");
  for (int x = 0; x < steps; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(stepDelayMicroseconds);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(stepDelayMicroseconds);
  }
}
