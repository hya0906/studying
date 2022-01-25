import pyaudio
import cv2 as cv # OpenCV
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from tkinter import messagebox as msg
from PIL import Image as Img
from PIL import ImageTk
import imutils
import wave
import os, glob
import math
from tkinter.simpledialog import *

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
#RECORD_SECONDS = 0
WAVE_OUTPUT_FILENAME = "file"

# Face detection XML load and trained model loading
front_path = "./haarcascade_frontalface_default.xml"
profile_path = './haarcascade_profileface.xml'
eye_path = "./haarcascade_eye.xml"
face_detection1 = cv.CascadeClassifier(front_path)
face_detection2 = cv.CascadeClassifier(profile_path)
eyeCascade = cv.CascadeClassifier(eye_path)
emotion_classifier = load_model('./emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

class Audio_Video:
    def __init__(self):
        self.audio, self.stream, self.func, self.window, self.lbl1 = 0, 0, 0, 0, 0
        self.faces, self.face_color, self.frame_raw = self, self, self
        self.stop, self.cap, self.count = 0, 0, 1
        self.frames = []
        self.eye_1, self.eye_2 = (0,), (0,)

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

    def recording(self): #파일 녹음
        self.newstream()
        print("recording...")
        self.frames = []
        while self.func.yesno == 1:
            data = self.stream.read(CHUNK)
            self.frames.append(data)
            if self.func.yesno == 0:
                break
        print("finished recording")
        # stop Recording
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def after_rec(self): #녹음후처리
        try:
            if not os.path.exists("./audio"):
                os.makedirs("./audio")
        except OSError:
            print('Error: Creating directory. ' + "./audio")
        while True:
            WAVE_OUTPUT_FILENAME = askstring("확인", '저장하고자 하는 파일의 이름을 쓰세요\n("Angry":1, "Disgusting":2, "Fearful":3, "Happy":4, "Sad":5, "Surpring":6, "Neutral":7)')
            if not WAVE_OUTPUT_FILENAME:
                msg.showinfo('error', "이름을 쓰세요")
            else:
                break
        os.chdir("./audio")
        condition = f"{WAVE_OUTPUT_FILENAME}_*.wav"
        wavfiles = glob.glob(condition)
        if not wavfiles:
            self.count = 1
        else:
            wavfile = wavfiles.pop(-1)
            c = wavfile.split(".")[0].split('_')[-1]
            self.count = int(c) + 1
        waveFile = wave.open(f"{WAVE_OUTPUT_FILENAME}_{self.count}"+".wav", 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()
        self.frames = []
        self.count+=1
        os.chdir("../")

    def get_elements(self,func, window, lbl1, cap=0):
        self.func = func
        self.window = window
        self.lbl1 = lbl1
        self.cap = cap

    def get_cap(self, cap):
        self.cap = cap

    def video_play(self): # cap에 따라 실시간,동영상 가능
        ret, frame = self.cap.read()  # 프레임이 올바르게 읽히면 ret은 True

        frame = cv.flip(frame, 1)

        if frame is None:  # 영상끝나면 영상만 종료
            self.window.after_cancel(self.stop)

        if not ret:
            self.cap.release()  # 작업 완료 후 해제
            return

        self.frame_raw = frame.copy()
        frame = imutils.resize(frame, width=480, height=400)
        #cv.imshow("f",frame)
        # Convert color to gray scale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rotated = self.detect(gray, frame)
        #cv.imshow('r',rotated)
        # Face detection in frame
        faces = face_detection1.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)) #정면
        if faces is None:
            faces = face_detection1.detectMultiScale(rotated, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))  # 정면
        if faces is None:
            faces = face_detection2.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)) #측면

        canvas = np.zeros((350, 300, 3), dtype="uint8")
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
            label = EMOTIONS[preds.argmax()]

            # Assign labeling
            cv.putText(frame, label, (fX, fY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        img = Img.fromarray(frame)  # Image 객체로 변환
        imgtk = ImageTk.PhotoImage(image=img)  # ImageTk 객체로 변환
        # OpenCV 동영상
        self.lbl1.imgtk = imgtk
        self.lbl1.configure(image=imgtk)
        self.stop = self.lbl1.after(10, self.video_play)
        #메모리사용량
        #pid = os.getpid()
        #py = psutil.Process(pid)
        #memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        #print('memory use:', memoryUse)

    def euclidean_distance(self,a, b):
        x1 = a[0]
        y1 = a[1]
        x2 = b[0]
        y2 = b[1]
        return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

    def detect(self, gray, frame):
        global new_img
        # 등록한 Cascade classifier 를 이용 얼굴을 찾음
        faces = face_detection1.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100),
                                                 flags=cv.CASCADE_SCALE_IMAGE)
        # 얼굴에 사각형을 그리고 눈을 찾자
        for (x, y, w, h) in faces:
            # 얼굴: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 255 0 0 , 굵기 2)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 이미지를 얼굴 크기 만큼 잘라서 그레이스케일 이미지와 컬러이미지를 만듬
            face_gray = gray[y:y + h, x:x + w]
            self.face_color = frame[y:y + h, x:x + w]

            # 등록한 Cascade classifier 를 이용 눈을 찾음(얼굴 영역에서만)
            eyes = eyeCascade.detectMultiScale(face_gray, 1.1, 3)
            # lefteye = lefteyeCascade.detectMultiScale(face_gray, 1.1, 3)

            # 눈: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 0 255 0 , 굵기 2)
            for i, (eye_x, eye_y, eye_w, eye_h) in enumerate(eyes):
                #cv.rectangle(self.face_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 2)
                #cv.circle(self.face_color, (eye_x, eye_y), 2, (0, 0, 255), 2)
                #cv.circle(self.face_color, (eye_x + eye_w, eye_y + eye_h), 2, (0, 0, 255), 2)
                #cv.circle(self.face_color, (eye_x + int(eye_w / 2), eye_y + int(eye_h / 2)), 2, (0, 0, 255), 2)
                if i == 0:
                    self.eye_1 = (eye_x, eye_y, eye_w, eye_h)
                elif i == 1:
                    self.eye_2 = (eye_x, eye_y, eye_w, eye_h)

        if self.eye_1[0] < self.eye_2[0]:
            left_eye = self.eye_1
            right_eye = self.eye_2
        else:
            left_eye = self.eye_2
            right_eye = self.eye_1

        #print(left_eye, right_eye)
        try:
            left_eye_center = (left_eye[0] + int(left_eye[2] / 2), left_eye[1] + int(left_eye[3] / 2))
            left_eye_x = left_eye_center[0]
            left_eye_y = left_eye_center[1]
            right_eye_center = (right_eye[0] + int(right_eye[2] / 2), right_eye[1] + int(right_eye[3] / 2))
            right_eye_x = right_eye_center[0]
            right_eye_y = right_eye_center[1]
        except IndexError:
            #print("IndexError")
            return

        #print("center", left_eye_center, right_eye_center)
        ##cv.circle(self.face_color, left_eye_center, 2, (0, 0, 255), 2)
        ##cv.circle(self.face_color, right_eye_center, 2, (0, 0, 255), 2)
        ##cv.line(self.face_color, right_eye_center, left_eye_center, (67, 67, 67), 2)
        try:
            if left_eye_y < right_eye_y:
                point_3rd = (right_eye_x, left_eye_y)
                direction = -1  # rotate same direction to clock
                #print("rotate to clock direction")
            else:
                point_3rd = (left_eye_x, right_eye_y)
                direction = 1  # rotate inverse direction of clock
                #print("rotate to inverse clock direction")
            ##cv.circle(self.face_color, point_3rd, 2, (255, 0, 0), 2)
            ##cv.line(self.face_color, right_eye_center, left_eye_center, (67, 67, 67), 2)
            ##cv.line(self.face_color, left_eye_center, point_3rd, (67, 67, 67), 2)
            ##cv.line(self.face_color, right_eye_center, point_3rd, (67, 67, 67), 2)

            a = self.euclidean_distance(left_eye_center, point_3rd)
            b = self.euclidean_distance(right_eye_center, left_eye_center)
            c = self.euclidean_distance(right_eye_center, point_3rd)
            #print("length:", a ** 2 + c ** 2)
        except:
            pass
        try:
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            #print("cos(a) = ", cos_a)

            angle = np.arccos(cos_a)
            #print("angle: ", angle, " in radian")

            angle = (angle * 180) / math.pi
            #print("angle: ", angle, " in degree")

            if direction == -1:
                angle = 90 - angle
            if angle<2:
                angle = 0.1

            new_img = Img.fromarray(self.frame_raw)
            new_img = np.array(new_img.rotate(direction * -angle))

        except ZeroDivisionError as e:
            pass

        return new_img  # frame