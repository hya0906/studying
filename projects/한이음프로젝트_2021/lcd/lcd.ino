#include <TFT.h> //아두이노 LCD 라이브러리 포함 
#include <SPI.h> //아두이노 SPI 라이브러리 포함


// 핀 정의하기 
#define cs   10  //CS or Chip Select
#define dc   9  //DC or A0, Data/Command
#define rst  8   //Reset



// TFT 클래스 생성 
//CS와 DC, Reset만 추가하면 된다.
//하드웨어 SPI는 클래스 내부에서 자동 생성 됨.
TFT TFTscreen = TFT(cs, dc, rst);

// 센서 값을 저장할 변수
char sensorPrintout1[2];  //pir
char sensorPrintout2[3];  //temp
char sensorPrintout3[3];  //humi
char sensorPrintout4[6]; //ppm
char sensorPrintout5[6]; //ultra


String s_result;

String pir;
String temp;
String humi;
String ppm;
String ultra;


void setup() {

  //TFT 클래스 시작
  Serial.begin(9600);
  TFTscreen.begin();

  // 검정색으로 초기화 
  TFTscreen.background(0, 0, 0);

  // 폰트 칼라 설정 (흰색)
  TFTscreen.stroke(255, 255, 255);
  // 폰트 크기를 5로 설정
  TFTscreen.setTextSize(2);

  TFTscreen.setTextSize(2);
  TFTscreen.text("PIR: ", 0, 0);
  TFTscreen.text("HUMI: ", 0, 20);
  TFTscreen.text("TEMP: ", 0, 40);
  TFTscreen.text("PPM: ", 0, 60);
  TFTscreen.text("ULTRA: ", 0, 80);
  
}

void loop() {
  
  String result = Serial.readString();

  int first = result.indexOf("/");
  int sec = result.indexOf("/",first +1);
  int third = result.indexOf("/",sec +1);
  int fourth = result.indexOf("/",third +1);
  int last = result.indexOf("/", fourth +1);

  pir = result.substring(0,first);
  temp = result.substring(first+1,sec);
  humi = result.substring(sec+1,third);
  ppm = result.substring(third+1,fourth);
  ultra = result.substring(fourth+1,last);

  pir.toCharArray(sensorPrintout1, 2);
  temp.toCharArray(sensorPrintout2, 3);
  humi.toCharArray(sensorPrintout3, 3);
  ppm.toCharArray(sensorPrintout4, 6);
  ultra.toCharArray(sensorPrintout5, 6);
    
  // 흰색으로 변경 
  TFTscreen.stroke(255, 255, 255);
  // LCD에 X : 0, Y : 20 위치에 글씨 표시 
  TFTscreen.text(sensorPrintout1, 50, 0);
  TFTscreen.text(sensorPrintout2, 50, 20);
  TFTscreen.text(sensorPrintout3, 50, 40);
  TFTscreen.text(sensorPrintout4, 50, 60);
  TFTscreen.text(sensorPrintout5, 70, 80);

  delay(5000);
  
  // 지금 쓴 글씨를 제거하기 위해 색을 검정색으로 변경
  TFTscreen.stroke(0, 0, 0);
  // 동일한 글씨를 검정색으로 표시하기 때문에 지워짐.
  TFTscreen.text(sensorPrintout1, 50, 0);
  TFTscreen.text(sensorPrintout2, 50, 20);
  TFTscreen.text(sensorPrintout3, 50, 40);
  TFTscreen.text(sensorPrintout4, 50, 60);
  TFTscreen.text(sensorPrintout5, 70, 80);

  //다시 loop함수 처음으로 이동
 
  
}
