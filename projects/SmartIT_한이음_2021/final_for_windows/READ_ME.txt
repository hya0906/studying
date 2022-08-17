window 환경변수에 able-store-324904-53433566eb54.json 파일 추가 (tts API용)

able-store-324904-53433566eb54.json --> tts API key json파일
abcd-27823-firebase-adminsdk-6hp4e-38c6900af9.json --> firestore key json 파일

DHT_PIR_WAVE 폴더의 DHT_PIR_WAVE.ino파일을 아두이노에 업로드.
DHT -> 7번핀
PIR -> 8번핀
WAVE-TRIG -> 9번핀
WAVE-ECHO -> 10번핀

마리아DB 쿼리 종류
date(char)
temp(int)
humid(int)
wave(char)
decibel(int)
PIR(char)

receive_data모듈에서 아두이노 환경 값 설정.
record_data 모듈에서 마리아db 환경 값 설정.
main 모듈 작동 시 프로그램 작동 가능.

t 입력 시 tts파일 생성.(output.mp3 파일로 생성됨)