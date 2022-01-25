from tkinter import *
from tkinter.filedialog import *

w = Tk()
w.geometry("400x100")

lbl = Label(w, text = "선택된 파일의 이름")
lbl.pack()

fName = askopenfilename(parent=w, filetypes=(("Mp4 파일", "*.mp4"),
    ("모든 파일", "*.*")))

lbl.configure(text = str(fName))

w.mainloop()