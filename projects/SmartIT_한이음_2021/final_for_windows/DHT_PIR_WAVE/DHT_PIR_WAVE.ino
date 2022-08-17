#include <DHT.h>

#define DHTPIN 7
#define PIR 8
#define TRIG 9
#define ECHO 10


#define DHTTYPE DHT11

int value = 0;
bool flag = false;
DHT dht(DHTPIN, DHTTYPE);



void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(PIR, INPUT);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  
  dht.begin();
}

void loop() {
  delay(1000);
  value = digitalRead(PIR);
  int h = dht.readHumidity();
  int t = dht.readTemperature();
  long duration, distance;

  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  duration = pulseIn(ECHO, HIGH);
  distance = duration *17 / 1000;

  if(value == HIGH && flag == true){
    flag=false;
  }
  else if (value == HIGH && flag == false){
    flag = true;
  }
  Serial.print(flag); //pir
  Serial.print(h);    //humi
  Serial.print(t);    //temp
  Serial.print(distance);  //wave
  Serial.print("\n");
}
