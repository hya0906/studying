#include <DHT.h>  //DHT11
#include <MQUnifiedsensor.h>  //MQ2

#define DHTPIN 7
#define PIR 8
#define TRIG 9    /**wave_trig**/
#define ECHO 10    /**wave_echo**/
/** For MQ2 Hardware Related Macro **/
#define Board                   ("Arduino UNO")
#define Pin                     (A2)  //MQ2 pin

/** For MQ2 Software Related Macro **/
#define Type                    ("MQ-2") //MQ2
#define Voltage_Resolution      (5)
#define ADC_Bit_Resolution      (10) // For arduino UNO/MEGA/NANO
#define RatioMQ2CleanAir        (9.83) //RS / R0 = 9.83 ppm 
#define DHTTYPE DHT11

MQUnifiedsensor MQ2(Board, Voltage_Resolution, ADC_Bit_Resolution, Pin, Type);
DHT dht(DHTPIN, DHTTYPE);


int value = 0;
bool flag = false;
char sensorPrintout[20];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  dht.begin();


  /** pinMode for pir,wave  **/
  pinMode(PIR, INPUT);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  
  
  MQ2.setRegressionMethod(1); //_PPM =  a*ratio^b
  MQ2.setA(574.25); MQ2.setB(-2.222); // Configurate the ecuation values to get LPG concentration

  MQ2.init(); 
  /* 
    //If the RL value is different from 10K please assign your RL value with the following method:
    MQ2.setRL(10);
  */
  /*****************************  MQ CAlibration ********************************************/ 
  // Explanation: 
  // In this routine the sensor will measure the resistance of the sensor supposing before was pre-heated
  // and now is on clean air (Calibration conditions), and it will setup R0 value.
  // We recomend execute this routine only on setup or on the laboratory and save on the eeprom of your arduino
  // This routine not need to execute to every restart, you can load your R0 if you know the value
  // Acknowledgements: https://jayconsystems.com/blog/understanding-a-gas-sensor
  //Serial.print("Calibrating please wait.");
  float calcR0 = 0;
  for(int i = 1; i<=10; i ++)
  {
    MQ2.update(); // Update data, the arduino will be read the voltage on the analog pin
    calcR0 += MQ2.calibrate(RatioMQ2CleanAir);
    //Serial.print(".");
  }
  MQ2.setR0(calcR0/10);
  //Serial.println("  done!.");
  
  if(isinf(calcR0)) {Serial.println("Warning: Conection issue founded, R0 is infite (Open circuit detected) please check your wiring and supply"); while(1);}
  if(calcR0 == 0){Serial.println("Warning: Conection issue founded, R0 is zero (Analog pin with short circuit to ground) please check your wiring and supply"); while(1);}
  /*****************************  MQ CAlibration ********************************************/ 

  MQ2.serialDebug(false); //true는 앞에 정보나옴

  
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
  Serial.print(flag);
  Serial.print("/");
  Serial.print(h);
  Serial.print("/");
  Serial.print(t);
  Serial.print("/");


  MQ2.update(); // Update data, the arduino will be read the voltage on the analog pin
  float ppm = MQ2.readSensor();// Sensor will read PPM concentration using the model and a and b values setted before or in the setup
  
  Serial.print(ppm);
  Serial.print("/");
  Serial.print(distance);
  Serial.print("/");
  Serial.print("\n");
  

}
