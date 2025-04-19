// Pin Definitions
const int motorPin = 23;  // PWM pin connected to motor

void setup() {
  Serial.begin(115200);
  pinMode(motorPin, OUTPUT); // Set motor pin as an output
  Serial.println("Send 1 to start motor, 0 to stop motor.");
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read(); // Read a single character

    if (receivedChar == '1') {
      analogWrite(motorPin, 130); // Start motor at PWM 130
      Serial.println("Motor ON");
    } 
    else if (receivedChar == '0') {
      analogWrite(motorPin, 0); // Stop motor
      Serial.println("Motor OFF");
    }
  }
}
