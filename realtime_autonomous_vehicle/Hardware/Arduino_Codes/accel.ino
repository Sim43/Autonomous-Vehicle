const int motorPin = 23;      // PWM pin connected to motor
const int reversePin = 5;     // Direction control pin

int motorSpeed = 65;
bool motorRunning = false;
bool reverse = false;

void setup() {
  Serial.begin(115200);
  pinMode(motorPin, OUTPUT); 
  pinMode(reversePin, OUTPUT); 
  digitalWrite(reversePin, LOW);
  analogWrite(motorPin, 0);
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();

    switch (receivedChar) {
      case '1': // Forward
        if (!motorRunning || reverse) {
          analogWrite(motorPin, 0); // Stop first if changing direction
          digitalWrite(reversePin, LOW);
          delay(50); // Brief pause for direction switch
          analogWrite(motorPin, motorSpeed);
          reverse = false;
          motorRunning = true;
          Serial.println("Motor ON Forward");
        }
        break;

      case '2': // Reverse
        if (!motorRunning || !reverse) {
          analogWrite(motorPin, 0); // Stop first if changing direction
          digitalWrite(reversePin, HIGH);
          delay(50); // Brief pause for direction switch
          analogWrite(motorPin, motorSpeed);
          reverse = true;
          motorRunning = true;
          Serial.println("Motor ON Reverse");
        }
        break;

      case '0': // Stop
        if (motorRunning) {
          analogWrite(motorPin, 0);
          motorRunning = false;
          Serial.println("Motor OFF");
        }
        break;

      case 'f':
        Serial.println("ACCELERATION_MODULE");
        break;

      default:
        Serial.println("Invalid input");
    }
  }
}
