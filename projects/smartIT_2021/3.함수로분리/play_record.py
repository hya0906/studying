'''
import winsound

winsound.PlaySound('sound.wav', winsound.SND_FILENAME)
'''
#음성녹음한 것 출력-실행됨
#https://bskyvision.com/940
import playsound

playsound.playsound('./file.wav') #절대,상대경로로 넣어야 함. 그냥 파일이름 쓰면 실행안됨