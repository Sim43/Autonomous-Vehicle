#define ANALOG_PIN 33   // A0 connected to GPIO33
#define DIGITAL_PIN 13  // D0 connected to GPIO13

const float WHEEL_CIRCUMFERENCE = 1.822;  // meters
const float MAX_VALID_SPEED = 40.0;       // km/h, 5 km/h margin above max
const unsigned long MIN_VALID_INTERVAL = 218700; // µs = 200 ms (max ~35–40 km/h)
const unsigned long MAX_VALID_INTERVAL = 2000000; // 2 sec = ~3.2 km/h

float voltage = 0.0;
int sensorValue = 0;
int lowerThreshold = 992;   // 0.8 V
int upperThreshold = 1934;  // 1.56 V
int magnetCount = 0;
bool magnetDetected = false;

unsigned long currentTime = 0;
unsigned long lastTime = 0;
unsigned long interval = 0;
unsigned long lastInterval = 0;

float speedKph = 0.0;

void setup() {
  Serial.begin(115200);
  pinMode(DIGITAL_PIN, INPUT);
}

void loop() {
  sensorValue = analogRead(ANALOG_PIN);
  voltage = (sensorValue / 4095.0) * 3.3;

  if (sensorValue >= lowerThreshold && sensorValue <= upperThreshold &&
      !magnetDetected && (micros() - currentTime > MIN_VALID_INTERVAL)) {
    
    magnetDetected = true;
    magnetCount++;

    lastTime = currentTime;
    currentTime = micros();
    interval = currentTime - lastTime;

    if (magnetCount > 1) {
      float intervalSec = interval / 1e6;
      speedKph = (WHEEL_CIRCUMFERENCE / intervalSec) * 3.6;

      if (speedKph <= MAX_VALID_SPEED) {
        Serial.println(speedKph, 2);
        lastInterval = interval;
      } else {
        // Ignore false high speed
        Serial.println("Invalid speed ignored.");
      }
    } else {
      Serial.print("Count: ");
      Serial.print(magnetCount);
      Serial.println("\tWaiting for next interval...");
    }
  } else if (sensorValue < lowerThreshold || sensorValue > upperThreshold) {
    magnetDetected = false;
  }

  delay(10); // Sampling control
}