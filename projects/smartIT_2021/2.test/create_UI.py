from matplotlib import pyplot as plt
import numpy as np
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as Tk
import collections

from receive_data import Sensor
import keyboard
#from read_to_gcloud import read_data as tts


class UI:
    def __init__(self,obj):
        self.machine = obj

    def thread_process(self, rec):
        thread_sensor = threading.Thread(target=sensor.all, args = (rec,))
        thread_sensor.setDaemon(True)
        thread_sensor.start()

    def tk(self):
        global lbl1, root, text
        root = Tk.Tk()  # 추가
        root.geometry("980x620")
        label = Tk.Label(root,text="라벨").grid(column=0, row=0)#추가
        streamL = Tk.Label(root, text="STREAMING")
        streamL.grid(row=0, column=0)

        text = Tk.StringVar()
        text.set("안녕하세요 반갑습니다!")
        e = Tk.Label(root, textvariable=text)
        e.place(x=150, y=535)

        frm = Tk.Frame(root, bg='white', width=320, height=180)  # 스트리밍 기본화면
        frm.grid(row=1, column=0)  # 위치를 1,0로

        lbl1 = Tk.Label(frm)
        lbl1.grid()

    def graph(self):
        canvas = FigureCanvasTkAgg(fig, master=root)  #tkinter에 canvas합침
        canvas.get_tk_widget().grid(column=1, row=1)
        root.after(10, sensor.video_play)
        Tk.mainloop()

    def ui_all(self):
        global sensor
        self.tk()              #tkinter 생성
        sensor = Sensor(lbl1)  # 센서 클래스 객체생성
        self.graph()           #tkinter과 matplotlib합체


if __name__ == '__main__':
    lbl1 = 0
    a = UI(lbl1)
    a.ui_all()
