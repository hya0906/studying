#1.실시간으로 음성을 입력받아서 저장하기 완료
#2.UI만들기 대략 완성
#UI에다가 버튼으로 녹음기능 만들기 또는 실시간으로?
#딥러닝 결과값 나누기
#https://tykimos.github.io/2017/06/10/Model_Save_Load/
from tkinter import *
import pyaudio
import wave
from threading import Thread
import cv2 as cv # OpenCV
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image, ImageTk
import imutils
import os
import psutil

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
#RECORD_SECONDS = 0
WAVE_OUTPUT_FILENAME = "file.wav"

# Face detection XML load and trained model loading
face_detection = cv.CascadeClassifier('C:/Users/USER/Desktop/files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('C:/Users/USER/Desktop/files/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

class Audio:
    def __init__(self):
        self.audio, self.stream = 0, 0
        self.frames = []
        global count
        count = 0

    def newstream(self):
        self.audio = pyaudio.PyAudio()
        # start Recording
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      input_device_index=2,
                                      frames_per_buffer=CHUNK)
        self.frames = []

    def recording(self):
        self.newstream()
        print("recording...")
        self.frames = []
        while yesno == 1:
            data = self.stream.read(CHUNK)
            self.frames.append(data)
            if yesno == 0:
                break
        print("finished recording")
        # stop Recording
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def after_rec(self):
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()
        self.frames = []

def changeText():
    global yesno, count
    if yesno == 0:
        yesno = 1
        count+=1
        label1.configure(text= str(count)+" recording...")
        a = Thread(target=rec.recording)
        a.setDaemon(True)
        a.start()
    elif yesno == 1:
        rec.after_rec()
        label1.configure(text= str(count)+" finished recording")
        yesno = 0

def video_play():
    ret, frame = cap.read()  # 프레임이 올바르게 읽히면 ret은 True
    frame = cv.flip(frame, 1)
    frame = imutils.resize(frame, width=480, height=360)
    if not ret:
        cap.release()  # 작업 완료 후 해제
        return
    # Convert color to gray scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Face detection in frame
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Perform emotion recognition only when face is detected
    if len(faces) > 0:
        # For the largest image
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # Resize the image to 48x48 for neural network
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Emotion predict
        preds = emotion_classifier.predict(roi)[0]
        #print(preds)
        emotion_probability = np.max(preds)
        # print(type(emotion_probability[0].astype('float64')),emotion_probability[0].astype('float64'),dtype='int32')
        label = EMOTIONS[preds.argmax()]

        # Assign labeling
        cv.putText(frame, label, (fX, fY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)


    img = Image.fromarray(frame)  # Image 객체로 변환
    imgtk = ImageTk.PhotoImage(image=img)  # ImageTk 객체로 변환
    # OpenCV 동영상
    lbl1.imgtk = imgtk
    lbl1.configure(image=imgtk)
    lbl1.after(10, video_play)

    #pid = os.getpid()
    #py = psutil.Process(pid)
    #memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    #print('memory use:', memoryUse)

cap = cv.VideoCapture(0)  # VideoCapture 객체 정의
yesno = 0
rec = Audio()
window = Tk()
window.title("감정인식")
window.geometry("710x400")

#b = Thread(target=c.cam)
#b.setDaemon(True)
#b.start()

# 라벨 추가
lbl = Label(window, text="Tkinter와 OpenCV를 이용한 GUI 프로그래밍")
lbl.grid(row=0, column=0) # 라벨 행, 열 배치

# 프레임 추가
frm = Frame(window, bg="white", width=480, height=360) # 프레임 너비, 높이 설정
frm.grid(row=1, column=0) # 격자 행, 열 배치

# 라벨1 추가
lbl1 = Label(frm)
lbl1.grid()

video_play()

label1 = Label(window, text="Process")
label2 = Label(window, text="ver VideoFile")
label3 = Label(window, text="ver AudioFile")
label4 = Label(window, text="ver Real time")
label5 = Label(window, text="ver Real time")
#button1 = Button(window, text="Recording", command=changeText)
button2 = Button(window, text="Face/Voice")
button3 = Button(window, text="Only Voice")
button4 = Button(window, text="Only Face")
button5 = Button(window, text="Only VoiceRec", command=changeText)

label2.place(x=500, y=50)
label3.place(x=500, y=100)
label4.place(x=500, y=150)
label5.place(x=500, y=200)
label1.place(x=580, y=230)

button2.place(x=580, y=50)
button3.place(x=580, y=100)
button4.place(x=580, y=150)
button5.place(x=580, y=200)
#button1.place(x=500, y=300)

window.mainloop()

