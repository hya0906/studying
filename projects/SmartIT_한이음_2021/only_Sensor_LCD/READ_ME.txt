필요 라이브러리
DHT_PIR_WAVE_MQ2 ->DHT sensor library by Adafruit / MQUnifiedsensor by Miguel Califa / Adafruit Unified Sensro by Adafruit 설치
LCD -> TFT by Built-In by Arduino"  설치(대부분 기본적으로 설치되어 있음)


port연결
DHT_PIR_WAVE_MQ2 -> COM3
lcd -> COM4

핀 연결하기
DHT_PIR_WAVE_MQ2 -> COM3에만 연결
-> DHT7				7
    PIR				8
    TRIG				9
    ECHO				10
    MQ2				A2
    GND는 묵어서 			GND
    VCC는 묶어서			5V    

lcd -> COM4에만 연결
-> LED 				3.3v
    SDK				13
    SDA				11
    A0				9
    RESET				8
    CS				10
    GND				GND
    VCC     			5V








