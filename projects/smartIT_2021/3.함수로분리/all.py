#1.실시간으로 음성을 입력받아서 저장하기 완료
#UI만들기
#UI에다가 버튼으로 녹음기능 만들기 또는 실시간으로?
#딥러닝 결과값 나누기
#https://tykimos.github.io/2017/06/10/Model_Save_Load/
from tkinter import *
from tkinter.filedialog import *
import cv2 as cv # OpenCV
from UI_func import UI_func
from Audio_Video import Audio_Video


#cap = cv.VideoCapture(0)  # VideoCapture 객체 정의
cap = cv.VideoCapture('short.mp4')
rec = Audio_Video()

window = Tk()
window.title("감정인식")
window.geometry("710x400")

func = UI_func(window, rec)

# 라벨 추가
lbl = Label(window, text="OpenCV/Deep learning emotion detection")
lbl.grid(row=0, column=0) # 라벨 행, 열 배치

# 프레임 추가
frm = Frame(window, bg="white", width=480, height=360) # 프레임 너비, 높이 설정
frm.grid(row=1, column=0) # 격자 행, 열 배치

# 라벨1 추가
lbl1 = Label(frm)
lbl1.grid()

rec.get_elements(func, window, lbl1, cap)
rec.video_play()

label1 = Label(window, text="Process")
label2 = Label(window, text="ver VideoFile")
label3 = Label(window, text="ver AudioFile")
label4 = Label(window, text="ver Real time")
label5 = Label(window, text="ver Real time")
label6 = Label(window, text="FilePath")
label7 = Label(window, text="FilePath")
label8 = Label(window, text="Result",font=16)
#button1 = Button(window, text="Recording", command=changeText)
button2 = Button(window, text="Face/Voice", command = lambda: func.getVideoFile(label6)) #flag = 1
button3 = Button(window, text="Only Voice", command = lambda: func.getAudioFile(label7)) #flag = 2
button4 = Button(window, text="Only Face") #flag = 3
button5 = Button(window, text="Only VoiceRec", command=lambda: func.changeText(label1)) #flag =4
button6 = Button(window, text="flag reset and video quit", command=lambda: func.flag_reset(label6, label7, rec.stop)) #flag =4
button7 = Button(window, text="quit", command=func.quit_UI) #flag =4

label2.place(x=500, y=50)
label3.place(x=500, y=100)
label4.place(x=500, y=150)
label5.place(x=500, y=200)
label1.place(x=580, y=230)
label6.place(x=500, y=75)
label7.place(x=500, y=125)
label8.place(x=550, y=300)

button2.place(x=580, y=50)
button3.place(x=580, y=100)
button4.place(x=580, y=150)
button5.place(x=580, y=200)
#button1.place(x=500, y=300)
button6.place(x=580, y=330)
button7.place(x=580, y=360)

window.mainloop()
