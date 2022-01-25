from tkinter import *

# 새로운 창 설정
win1=Tk()
win1.title('정광규 tkinter')
win1.geometry('400x400')
win1.resizable(False, False)

# 문제 지시문 라벨 설정

lab1 = Label(win1)
lab1['text']='다음 영어 단어의 뜻을 입력하세요'
lab1.pack()

# 문제 라벨 설정
lab2 = Label(win1)
lab2['text'] = ' 영어 단어 next의 의미는?'
lab2.pack()

# 입력 결과 받을 라벨 설정
lab3 = Label(win1)  # 입력 받은 결과 출력
lab4 = Label(win1)  # 결과에 대한 채점 내용 출력


def 결과(n):
    lab3.config(text="입력한 내용입니다. > " + ent1.get())
    if ent1.get() == '다음':
        lab4.config(text='정답입니다.')
    else:
        lab4.config(text='오답입니다.')


# 입력창 설정
ent1 = Entry(win1)
ent1.bind("<Return>", 결과)
ent1.pack()
lab3.pack()
lab4.pack()
win1.mainloop()