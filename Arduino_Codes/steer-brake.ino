#include <ESP32Servo.h>

// Pin Definitions
const int stepPin = 5;
const int dirPin = 18;
const int servoPin = 13;

// Constants
const int stepsPerRevolution = 800;
const int stepDelayMicroseconds = 400;

// Servo
Servo brakeServo;

// Variables
volatile int targetAngle = 0;
volatile bool brakeCommand = false;
volatile bool brakeEngaged = false;

// Mutex for shared access
SemaphoreHandle_t mutex;

void setup() {
  Serial.begin(115200);
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  brakeServo.attach(servoPin);
  brakeServo.write(0);

  mutex = xSemaphoreCreateMutex();

  xTaskCreatePinnedToCore(steeringTask, "Steering Task", 2000, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(brakeTask, "Brake Task", 2000, NULL, 1, NULL, 1);

  Serial.println("System Ready. Awaiting commands like 'S:<angle>' and 'B:1'");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.startsWith("S:")) {
      int angle = input.substring(2).toInt();

      if (xSemaphoreTake(mutex, portMAX_DELAY)) {
        targetAngle = constrain(angle, -360, 360); // Limit to safe bounds
        xSemaphoreGive(mutex);
      }
    } else if (input.startsWith("B:")) {
      int b = input.substring(2).toInt();
      if (xSemaphoreTake(mutex, portMAX_DELAY)) {
        brakeCommand = (b == 1);
        xSemaphoreGive(mutex);
      }
    }
  }

  delay(10);
}

void steeringTask(void *pvParameters) {
  int currentAngle = 0;

  while (true) {
    int angleToMove = 0;

    if (xSemaphoreTake(mutex, portMAX_DELAY)) {
      angleToMove = targetAngle - currentAngle;
      xSemaphoreGive(mutex);
    }

    if (angleToMove != 0) {
      int direction = (angleToMove > 0) ? LOW : HIGH;
      digitalWrite(dirPin, direction);

      int steps = abs(angleToMove) * stepsPerRevolution / 360;

      for (int i = 0; i < steps; i++) {
        digitalWrite(stepPin, HIGH);
        delayMicroseconds(stepDelayMicroseconds);
        digitalWrite(stepPin, LOW);
        delayMicroseconds(stepDelayMicroseconds);
      }

      if (xSemaphoreTake(mutex, portMAX_DELAY)) {
        currentAngle += angleToMove;
        xSemaphoreGive(mutex);
      }
    }

    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

void brakeTask(void *pvParameters) {
  while (true) {
    bool applyBrake = false;

    if (xSemaphoreTake(mutex, portMAX_DELAY)) {
      applyBrake = brakeCommand;
      xSemaphoreGive(mutex);
    }

    if (applyBrake && !brakeEngaged) {
      Serial.println("Brake Engaged");
      brakeServo.write(180);
      brakeEngaged = true;
    } else if (!applyBrake && brakeEngaged) {
      Serial.println("Brake Released");
      brakeServo.write(0);
      brakeEngaged = false;
    }

    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}
