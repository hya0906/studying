from tkinter.filedialog import *
from tkinter import messagebox as msg
from threading import Thread
import cv2 as cv
from multiprocessing import Process, Queue
import time
import winsound
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class UI_func:
    def __init__(self, window, rec, learning):
        self.flag = 3
        self.window = window
        self.rec = rec
        self.yesno = 0
        self.count = 0
        self.fname = ''
        self.test = learning

    def record_audio(self, label1):
        if self.flag != 0 and self.flag != 4:
            msg.showinfo('error', str(self.flag) + '번이 실행 중입니다.')
        elif self.flag == 4:
            msg.showinfo('error', '이미 실행 중입니다.')
        else:
            if self.yesno == 0:
                self.yesno = 1
                self.count += 1
                label1.configure(text=str(self.count) + " recording...")
                a = Thread(target=self.rec.recording)
                a.setDaemon(True)
                a.start()
            elif self.yesno == 1:
                self.rec.after_rec()
                label1.configure(text=str(self.count) + " finished recording")
                self.yesno = 0

    def getVideoFile(self, label6, label8):
        if self.flag != 0 and self.flag != 1:
            msg.showinfo('error', str(self.flag) + '번이 실행 중입니다.')
            label6.configure(text="FilePath")
        elif self.flag == 1:
            msg.showinfo('error', '이미 실행 중입니다.')
        else:
            self.fName = askopenfilename(parent=self.window, filetypes=(("Mp4 파일", "*.mp4"), ("모든 파일", "*.*")))
            label6.configure(text=str(self.fName))
            if len(label6.cget("text")) == 0:
                msg.showinfo('확인', '파일이 선택되지 않았습니다.')
                label6.configure(text="FilePath")
            else:
                self.flag = 1
                results =[]; q = Queue()
                p = Process(target=self.test.get_feature, args=(self.fName, self.flag, q))
                p.start(); p.join()
                while not q.empty():
                    results.append(q.get())
                result = results[0].tolist()
                tr = results[2]
                cap = cv.VideoCapture(str(self.fName))
                b2 = Thread(target=self.update_label, args = (result,label8,self.flag,cap, tr))
                b2.start()
                self.rec.get_cap(cap)
                self.rec.video_play()

    #tkinter과 합치면서 속도 느려져서 속도를 맞추기 위해 프레임수에 맞춰서 레이블 출력
    def update_label(self, result,label,flag, cap, tr = 1, length = 0):
        emo = {1:'Angry', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Sad', 6:'Surprise', 7:'Neutral'}
        if flag == 1: #camera or video
            divided = cap.get(cv.CAP_PROP_FRAME_COUNT) / tr
            for i in result:
                label.configure(text=emo[i])
                if cap.get(cv.CAP_PROP_POS_FRAMES) % divided == 0:
                    time.sleep(0.1)
                while cap.get(cv.CAP_PROP_POS_FRAMES) % divided != 0:
                    if cap.get(cv.CAP_PROP_POS_FRAMES) == (divided * tr):
                        break
        elif flag == 2: #audio
            for a,i in enumerate(result):
                label.configure(text=emo[i])
                time.sleep(length//tr)

    def play_audio(self, path):
        winsound.PlaySound(path, winsound.SND_FILENAME)

    def getAudioFile(self, label7, label8):
        if self.flag != 0 and self.flag != 2:
            msg.showinfo('error', str(self.flag) + '번이 실행 중입니다.')
        elif self.flag == 2:
            msg.showinfo('error', '이미 실행 중입니다.')
        else:
            self.fName = askopenfilename(parent=self.window, filetypes=(("Wav 파일", "*.wav"), ("모든 파일", "*.*")))
            label7.configure(text=str(self.fName))
            if len(label7.cget("text")) == 0:
                msg.showinfo('확인', '파일이 선택되지 않았습니다.')
                label7.configure(text="FilePath")
            else:
                self.flag = 2
                results = []; q = Queue()
                p2 = Process(target=self.test.get_feature, args=(self.fName, self.flag, q))
                p2.start(); p2.join()
                c = Thread(target=self.play_audio, args=(self.fName,))
                c.start()
                while not q.empty():
                    results.append(q.get())
                result = results[0].tolist()
                length = results[1]
                tr = results[2]
                b = Thread(target=self.update_label, args=(result, label8, self.flag, self, tr, length))
                b.start()

    def realtimeVideo(self):
        if self.flag != 0 and self.flag != 3:
            msg.showinfo('error', str(self.flag) + '번이 실행 중입니다.')
        elif self.flag == 3:
            msg.showinfo('error', '이미 실행 중입니다.')
        elif self.flag == 0:
            self.flag = 3
            cap = cv.VideoCapture(0)
            self.rec.get_cap(cap)
            self.rec.video_play()

    def flag_reset(self, label6, label7, stop, cap):
        if self.flag == 0:
            msg.showinfo('error', '이미 flag=0입니다.')
        else:
            self.flag = 0
            self.window.after_cancel(stop)
            cap.release()
            label6.configure(text="FilePath")
            label7.configure(text="FilePath")

    def quit_UI(self, cap):
        cap.release()
        quit()