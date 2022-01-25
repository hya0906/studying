import cv2 as cv
from tkinter.filedialog import *
from tkinter.simpledialog import *
from ui_test import Application
from Audio_Video import Audio_Video
from UI_func import UI_func
from sound_learning import Test

def main():
    cap = cv.VideoCapture(0)  # VideoCapture 객체 정의
    # cap = cv.VideoCapture('short.mp4')
    # eye_1, eye_2, face_color = 0, 0, 0

    window = Tk()
    window.title("감정인식")
    window.geometry("710x400")

    rec = Audio_Video()
    func = UI_func(window, rec)
    app = Application(window, func, rec, cap, master=window)
    learning = Test()

    value = askstring("확인","다시 학습하시겠습니까?(yes/no)")
    if value == "yes":
        messagebox.showinfo("확인", "학습을 다시 합니다.")
        learning.main()
    elif value =="no":
        messagebox.showinfo("확인", "학습하지 않습니다.")

    rec.get_elements(func, window, app.lbl1, cap)
    rec.video_play()
    # window.after(10, rec.video_play)
    app.mainloop()

if  __name__ =="__main__":
    main()

