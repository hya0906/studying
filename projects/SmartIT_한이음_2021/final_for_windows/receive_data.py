import pyaudio
import time
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
import serial
import math
import os
import psutil
import record_data


port = 'COM3'
b_rate = 9600

port2 = 'COM4' #
b_rate2 = 9600 #

CHUNK = 2**11
RATE = 44100
class Sensor:
    global cap,font,face_cascade,body_cascade,seri,gas_seri
    cap= cv.VideoCapture(0, cv.CAP_DSHOW)  # 카메라는0번, 동영상은 이름넣으면 됨
    font = cv.FONT_HERSHEY_SIMPLEX  # human detect font??
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')  # 이게 있어야 얼굴인식가능
    body_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')  # 이게 있어야 몸인식가능
    seri = serial.Serial(port, baudrate=b_rate, timeout=None)
    gas_seri = serial.Serial(port2, baudrate=b_rate2, timeout=None) #

    def __init__(self,lbl1):
        self.temp = 0     #온도
        self.humi = 0     #습도
        self.pir = 0      #PIR
        self.ultra = 0    #초음파
        self.peak = 0     #데시벨
        self.gas = 0      #가스 #
        self.p = pyaudio.PyAudio() #오디오객체
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        self.start, self.now, self.t = time.time(), 0, 0 #시간계산
        self.decibel, self.d = np.zeros([1]),np.array([1,21]) #np.zeros([1,21])
        self.lbl1 = lbl1

    def sensor_data(self, ser, ser2): #
        self.pir = ser[:1]
        self.humi = ser[1:3]
        self.temp = ser[3:5]
        self.ultra = ser[5:]
        self.gas = ser2[-5:-1] #

    def sound(self):
        while True:
            self.now = time.time()
            if self.t != int(self.now - self.start):#매초 카운트
                self.decibel = np.vstack([self.decibel, self.d.sum() / 21]) #1초동안 수집된 데시벨의 평균값수집
                self.d = np.array([]) #데시벨 값 수집
                if self.t == 0:#처음 21개로 0으로 찬 리스트 삭제
                    self.decibel = np.delete(self.decibel, 0, 0)
            self.t = int(self.now - self.start) #초계산
            if len(self.d) == 21: #데시벨측정 1초에 21,22개->일정하게
                continue
            data = np.fromstring(self.stream.read(CHUNK), dtype=np.int16)
            self.peak = 20 * math.log10(np.average(np.abs(data)) * 2) #데시벨 구하기
            #print("peak : ", type(self.peak))
            self.d = np.hstack([self.d, self.peak])
            if len(self.decibel)>10: #180->10 실험
                self.decibel = np.delete(self.decibel, 0, 0) #3분만 평균냄

    def video_play(self):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return
        frame = cv.flip(frame, 1)  # 뒤집기
        frame = cv.resize(frame, dsize=(600, 480), interpolation=cv.INTER_AREA)
        face = face_cascade.detectMultiScale(frame, 1.8, 1, 0, (30, 30))
        body = body_cascade.detectMultiScale(frame, 1.8, 1, 0, (30, 30))
        # print("Number of body, face detected: " + str(len(body)) + ',' + str(len(face)))
        a = str(len(face))
        cv.putText(frame, 'people: '+ str(a), (450, 30), font, 0.9, (255, 255, 0), 2)
        cv.putText(frame, 'humid: ' + str(self.humi), (450, 70), font, 0.9, (255, 255, 0), 2)
        cv.putText(frame, 'temp : ' + str(self.temp), (450, 120), font, 0.9, (255, 255, 0), 2)
        cv.putText(frame, 'Ultra : ' + str(self.ultra), (450, 170), font, 0.9, (255, 255, 0), 2)
        cv.putText(frame, 'PIR : ' + str(self.pir), (450, 220), font, 0.9, (255, 255, 0), 2)
        for (x, y, w, h) in body:
            cv.rectangle(frame, (x, y), (x + w + 10, y + h + 10), (255, 0, 0), 3, 4, 0)  # 물체표시 사각형
            cv.putText(frame, 'Detected human', (x - 5, y - 5), font, 0.9, (255, 255, 0), 2)  # 물체표시 글

        if len(body) == 0:
            for (x, y, w, h) in face:
                cv.rectangle(frame, (x, y), (x + w + 10, y + h + 10), (255, 0, 0), 3, 4, 0)  # 물체표시 사각형
        
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lbl1.imgtk = imgtk
        self.lbl1.configure(image=imgtk)
        self.lbl1.after(10, self.video_play)  # 재귀호출이라서 오류나는 것 같음.



    def all(self):
        a = record_data.record_data()           
        while True:
            ser = seri.readline().rstrip().decode()
            ser2 = gas_seri.readline().rstrip().decode()#
            self.sensor_data(ser,ser2)#
            print(self.gas) #
            a.record_data(self)


