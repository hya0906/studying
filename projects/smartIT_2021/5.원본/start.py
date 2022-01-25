import cv2 as cv
from tkinter.simpledialog import *
from ui_test import Application
from Audio_Video import Audio_Video
from UI_func import UI_func
from sound_learning import Test
from multiprocessing import Process
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#음성 (1=Angry, 2=Disgust, 3=Fear, 4=Happy, 5=Sad, 6=Surprise, 7=Neutral)
def main():
    cap = cv.VideoCapture(0)  # VideoCapture 객체 정의

    window = Tk()
    window.title("감정인식")
    window.geometry("710x400")
    learning = Test()
    rec = Audio_Video()
    func = UI_func(window, rec, learning)
    app = Application(window, func, rec, cap, master=window)

    value = askstring("확인","다시 학습하시겠습니까?(yes/no)")
    if value == "yes":
        messagebox.showinfo("확인", "학습을 다시 합니다.")
        p1 = Process(target = learning.main)
        p1.start(); p1.join()
    elif value =="no":
        messagebox.showinfo("확인", "학습하지 않습니다.")

    rec.get_elements(func, window, app.lbl1, cap)
    rec.video_play()
    app.mainloop()

if  __name__ =="__main__":
    main()

