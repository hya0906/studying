from matplotlib import pyplot as plt
import numpy as np
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as Tk
import collections
from matplotlib.animation import FuncAnimation
from receive_data import Sensor

#from read_to_gcloud import read_data as tts
import keyboard

class UI:
    def thread_process(self):
        thread_sound = threading.Thread(target=sensor.sound)  # 쓰레드 타겟지정 객체
        thread_sensor = threading.Thread(target=sensor.all)
        thread_tts = threading.Thread(target = self.tts_on)

        thread_sound.setDaemon(True)  # 프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread_sensor.setDaemon(True)
        thread_tts.setDaemon(True)

        thread_sound.start()  # 쓰레드 시작
        thread_sensor.start()
        thread_tts.start()

    def tk(self):
        global lbl1, root
        root = Tk.Tk()  # 추가
        root.geometry("980x620")
        # label = Tk.Label(root,text="라벨").grid(column=0, row=0)#추가
        streamL = Tk.Label(root, text="STREAMING")
        streamL.grid(row=0, column=0)

        e = Tk.Label(root, text="대응문장넣을자리")
        e.place(x=150, y=535)

        frm = Tk.Frame(root, bg='white', width=320, height=180)  # 스트리밍 기본화면
        frm.grid(row=1, column=0)  # 위치를 1,0로

        lbl1 = Tk.Label(frm)
        lbl1.grid()

    def graph(self):
        global fig, cpu, ax
        # start collections with zeros
        cpu = collections.deque(np.zeros(20)) #그래프 값
        # define and adjust figure
        fig = plt.figure()
        ax = plt.subplot(121)
        canvas = FigureCanvasTkAgg(fig, master=root)  #tkinter에 canvas합침
        canvas.get_tk_widget().grid(column=1, row=1)
        # animate
        ani = FuncAnimation(fig, self.my_function, interval=200) #실시간 그래프 그림
        root.after(10, sensor.video_play)
        Tk.mainloop()

    def my_function(self,i):
        # get data
        cpu.popleft()           #그래프 마지막값 삭제
        cpu.append(sensor.peak) #그래프 리스트추가
        
        # clear axis
        ax.cla()                #그래프 삭제
        
        # plot cpu
        ax.plot(cpu)            #그래프 그림
        ax.set_ylim(0, 100)     #범위 0~100

    def tts_on(self):
        while True:
            if keyboard.read_key() == 't':
                pass
                #x = tts()
                #x.tts_data()

    def all(self):
        global sensor
        self.tk()              #tkinter 생성
        sensor = Sensor(lbl1)  # 센서 클래스 객체생성
        self.thread_process()  #모든 센서 쓰레드 돌리기
        self.graph()           #tkinter과 matplotlib합체


if __name__ == '__main__':
    a = UI()
    a.all()
