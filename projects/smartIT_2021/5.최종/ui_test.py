#https://runebook.dev/ko/docs/python/library/tkinter
from tkinter.filedialog import *

class Application(Frame):
    global window, cap
    def __init__(self, window, func, rec, cap, master=None):
        super().__init__(master)
        self.master = master
        self.window = window
        self.func = func
        self.rec = rec
        self.cap = cap
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        global func, rec,window
        # 라벨 추가
        self.lbl = Label(self.window, text = "OpenCV/Deep learning emotion detection")
        self.lbl.place(x=10, y=3)
        # 프레임 추가
        self.frm= Frame(self.window, bg="white", width=480, height=400)
        self.frm.place(x=10, y=25)
        # 라벨1 추가
        self.lbl1 = Label(self.frm)
        self.lbl1.grid()

        #label
        self.label1 = Label(self.window, text="ver VideoFile")
        self.label2 = Label(self.window, text="ver AudioFile")
        self.label3 = Label(self.window, text="ver Real time")
        self.label4 = Label(self.window, text="ver Real time")
        self.label5 = Label(self.window, text="Process")
        self.label6 = Label(self.window, text="FilePath")
        self.label7 = Label(self.window, text="FilePath")
        self.label8 = Label(self.window, text="Result", font=16)
        #button
        self.button1 = Button(self.window, text="Face/Voice", command=lambda: self.func.getVideoFile(self.label6, self.label8))  # flag = 1
        self.button2 = Button(self.window, text="Only Voice", command=lambda: self.func.getAudioFile(self.label7, self.label8))  # flag = 2
        self.button3 = Button(self.window, text="Only Face", command = self.func.realtimeVideo)             # flag = 3
        self.button4 = Button(self.window, text="Only VoiceRec", command=lambda: self.func.record_audio(self.label5)) # flag = 4
        self.button5 = Button(self.window, text="flag reset and video quit",
                         command=lambda: self.func.flag_reset(self.label6, self.label7, self.rec.stop, self.cap))   # flag = 0(default)
        self.button6 = Button(self.window, text="quit", command=lambda: self.func.quit_UI(self.cap))

        #place(=position) labels and buttons
        self.label1.place(x=500, y=50)
        self.label2.place(x=500, y=100)
        self.label3.place(x=500, y=150)
        self.label4.place(x=500, y=200)
        self.label5.place(x=580, y=230)
        self.label6.place(x=500, y=75)
        self.label7.place(x=500, y=125)
        self.label8.place(x=550, y=300)


        self.button1.place(x=580, y=50)
        self.button2.place(x=580, y=100)
        self.button3.place(x=580, y=150)
        self.button4.place(x=580, y=200)
        self.button5.place(x=530, y=330)
        self.button6.place(x=580, y=360)



