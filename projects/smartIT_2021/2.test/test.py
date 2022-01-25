#1.실시간으로 음성을 입력받아서 저장하기 완료
#UI만들기
#UI에다가 버튼으로 녹음기능 만들기 또는 실시간으로?
#딥러닝 결과값 나누기
#https://tykimos.github.io/2017/06/10/Model_Save_Load/
from tkinter import *
import pyaudio
import wave
from threading import Thread

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 0
WAVE_OUTPUT_FILENAME = "file.wav"

class Audio:
    def __init__(self):
        self.audio, self.stream = 0, 0
        self.frames = []
        global count
        count = 0

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

    def recording(self):
        self.newstream()
        print("recording...")
        self.frames = []
        while yesno == 1:
            data = self.stream.read(CHUNK)
            self.frames.append(data)
            if yesno == 0:
                break
        print("finished recording")
        # stop Recording
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def after_rec(self):
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()
        self.frames = []

def changeText():
    global yesno, count
    if yesno == 0:
        yesno = 1
        count+=1
        label.configure(text= str(count)+" recording...")
        a = Thread(target=rec.recording)
        a.setDaemon(True)
        a.start()
    elif yesno == 1:
        rec.after_rec()
        label.configure(text= str(count)+" finished recording")
        yesno = 0

yesno = 0
rec = Audio()
window = Tk()
window.title("감정인식")
window.geometry("640x480")
label = Label(window, text="Text")
button = Button(window, text="Click to change text below", command=changeText)
button.pack()
label.pack()

window.mainloop()

