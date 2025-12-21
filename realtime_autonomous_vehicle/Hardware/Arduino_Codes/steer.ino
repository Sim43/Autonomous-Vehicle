const int stepPin = 5;
const int dirPin  = 18;

float stepPerAngle = 1.8; // degrees per step
int currentAngle = 0;

void setup() {
  Serial.begin(115200);  // Match baud rate with your Python script
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
}

void moveToAngle(int targetAngle) {
  int angleDiff = targetAngle - currentAngle;
  int numSteps = abs(angleDiff) / stepPerAngle;

  digitalWrite(dirPin, angleDiff > 0 ? HIGH : LOW);

  for (int i = 0; i < numSteps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(1000);
  }

  currentAngle = targetAngle;
}

void loop() {
  static String inputString = "";
  
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();

    if (inChar == 'f') {
      Serial.println("STEERING_MODULE");
    }
    
    if (inChar == '\n') {
      int targetAngle = inputString.toInt();
      Serial.print("Moving to angle: ");
      Serial.println(targetAngle);
      moveToAngle(targetAngle);
      inputString = "";
    } else {
      inputString += inChar;
    }
  }
}
