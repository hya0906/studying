import pyaudio
import time
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
import serial
import math
from record_data import Record_data
from process import Machine
import os
import psutil

CHUNK = 2**11
RATE = 44100

#for pi
#port = '/dev/ttyACM0'
#port = '/dev/ttyACM1'


#for DESKTOP
port = 'COM3'
b_rate = 9600

port2 = 'COM4'
b_rate2 = 9600

delay = 3

class Sensor:
    global cap,font,face_cascade,body_cascade,seri,lcd_seri
    cap= cv.VideoCapture(0)  # 카메라는0번, 동영상은 이름넣으면 됨
    font = cv.FONT_HERSHEY_SIMPLEX  # human detect font??
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')  # 이게 있어야 얼굴인식가능
    body_cascade = cv.CascadeClassifier('haarcascade_fullbody.xml')  # 이게 있어야 몸인식가능
    seri = serial.Serial(port, baudrate=b_rate, timeout=None)
    lcd_seri = serial.Serial(port2, baudrate=b_rate2, timeout=None)  ##

    def __init__(self, lbl1, machine):
        self.temp = 0     #온도
        self.humi = 0     #습도
        self.pir = 0      #PIR
        self.ultra = 0    #초음파
        self.peak = 0     #데시벨
        self.ppm = ''     # 가스 ##
        self.people=''    #사람 유무
        self.p = pyaudio.PyAudio() #오디오객체
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        self.start, self.now, self.t = time.time(), 0, 0 #시간계산
        self.decibel, self.d = np.zeros([1]),np.array([1,21]) #np.zeros([1,21])
        self.lbl1 = lbl1
        self.machine = machine

    def sensor_data(self, ser): ##
        result = ser.split("/")
        try:
            self.pir = result[0]
            self.humi = result[1]
            self.temp = result[2]
            self.ppm = result[3]
            if int(result[4]) < 0:
                self.ultra = 0
            else:
                self.ultra = result[4]

        except IndexError as e:
            print(e)
        #print("ppm:", self.ppm)

    def lcd_on(self, ser2):
         lcd_seri.write(ser2)

    def sound(self):
        self.timer_for_noise_alert_start = time.time()
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
            self.peak = int(self.peak)
            self.d = np.hstack([self.d, self.peak])
            if len(self.decibel)>10: #180->10 실험
                self.decibel = np.delete(self.decibel, 0, 0) #3분만 평균냄

            #pid = os.getpid()
            #py = psutil.Process(pid)
            #memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
            #print('memory use:', memoryUse)

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
        self.people = int(a)
        
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
                # cv.putText(frame, 'Detected human', (x - 5, y - 5), font, 0.9, (255, 255, 0), 2)  # 물체표시 글

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lbl1.imgtk = imgtk
        self.lbl1.configure(image=imgtk)
        self.lbl1.after(10, self.video_play)  # 재귀호출이라서 오류나는 것 같음.

    def all(self, rec):
        r = rec
        while True:
            try:
                ser = seri.readline().rstrip().decode()
                #print(ser)
            except UnicodeDecodeError as e:
                continue

            ser2 = ser.encode('utf-8')
            self.sensor_data(ser)

            self.lcd_on(ser2)

            if self.machine.flag: #1이면 1초마다
                try:
                    if type(float(self.ppm)) == float:
                        print("저장")
                        if self.pir == 0:
                            self.timer_for_pir_start = time.time()
                        else:
                            self.timer_for_pir_end = time.time()
                        r.record_data()  # ◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆
                        self.machine.ppm_alert() #ppm 경보작동
                        time.sleep(0.99)  # every 1min, records
                    else:
                        print(type(float(self.ppm)))
                        print("넘어감")
                        pass
                except ValueError as e:
                    #print("all - ",e)
                    pass

            elif self.machine.flag == 0:            #0이면 5초마다
                try:
                    if type(float(self.ppm)) == float:
                        print("저장")
                        if self.pir == 0:
                            self.timer_for_pir_start = time.time()
                        else:
                            self.timer_for_pir_end = time.time()
                        r.record_data()  # ◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆
                        self.machine.ppm_alert()
                        time.sleep(4.99)  # every 5min, records
                    else:
                        print(type(float(self.ppm)))
                        print("넘어감")
                        pass
                except ValueError as e:
                    #print("all - ", e)
                    pass

