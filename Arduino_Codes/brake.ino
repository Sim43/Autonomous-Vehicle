// brake.ino
#include <ESP32Servo.h>

const int servoPin = 13;
Servo brakeServo;

void setup() {
  Serial.begin(115200);
  brakeServo.attach(servoPin);
  brakeServo.write(0);  // Start with brake released
  Serial.println("Brake control ready. Send 1 (brake) or 0 (release)");
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();
    if (command == '1') {
      Serial.println("Braking...");
      brakeServo.write(180);
    } else if (command == '0') {
      Serial.println("Releasing brake...");
      brakeServo.write(0);
    } else {
      Serial.println("Invalid brake command. Use 1 or 0.");
    }
  }
}
