#https://blog.daum.net/geoscience/1502
# 모듈 호출
import tkinter as tk # Tkinter
from PIL import ImageTk, Image # Pillow
import cv2 as cv # OpenCV
import os


cap = cv.VideoCapture(0)  # VideoCapture 객체 정의


def video_play():
    ret, frame = cap.read()  # 프레임이 올바르게 읽히면 ret은 True
    if not ret:
        cap.release()  # 작업 완료 후 해제
        return
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = Image.fromarray(frame)  # Image 객체로 변환
    imgtk = ImageTk.PhotoImage(image=img)  # ImageTk 객체로 변환
    # OpenCV 동영상
    lbl1.imgtk = imgtk
    lbl1.configure(image=imgtk)
    lbl1.after(10, video_play)

win = tk.Tk() # 인스턴스 생성

win.title("AniWatch") # 제목 표시줄 추가
win.geometry("920x640+50+50") # 지오메트리: 너비x높이+x좌표+y좌표
win.resizable(False, False) # x축, y축 크기 조정 비활성화

# 라벨 추가
lbl = tk.Label(win, text="Tkinter와 OpenCV를 이용한 GUI 프로그래밍")
lbl.grid(row=0, column=0) # 라벨 행, 열 배치

# 프레임 추가
frm = tk.Frame(win, bg="white", width=720, height=480) # 프레임 너비, 높이 설정
frm.grid(row=1, column=0) # 격자 행, 열 배치

# 라벨1 추가
lbl1 = tk.Label(frm)
lbl1.grid()

video_play()
win.mainloop()  # GUI 시작