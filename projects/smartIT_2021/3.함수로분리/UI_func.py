from tkinter.filedialog import *
from tkinter import messagebox as msg
from threading import Thread
import cv2 as cv
from sound_learning import Test

class UI_func:
    def __init__(self, window, rec):
        self.flag = 3
        self.window = window
        self.rec = rec
        self.yesno = 0
        self.count = 0
        self.fname = ''
        self.test = Test()


    def record_audio(self,label1):
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

    def getVideoFile(self, label6):
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
                self.test.get_videofeather(self.fName)
                cap = cv.VideoCapture(str(self.fName))
                self.rec.get_cap(cap)
                self.rec.video_play()

    def getAudioFile(self, label7):
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