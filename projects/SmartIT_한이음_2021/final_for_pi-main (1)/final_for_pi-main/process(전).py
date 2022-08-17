import numpy as np
import joblib
import time

class Machine:
    def __init__(self):
        self.obj = self
        self.lbl1 = self
        self.c, self.pir = 0, 0
        self.result = np.array([])
        self.flag = 0

    def get_obj(self, object, lbl1, record):
        self.obj = object  # Sensor의 객체
        self.lbl1 = lbl1  # lbl1
        self.record = record

    def decision(self):
        # PIR(0,1),마이크(0,1),온도(0,1,2),습도(0,1,2) 순
        loaded_model = joblib.load('./demo2_model.pkl')
        time.sleep(2)
        a, b, ultra = np.array([]), np.array([]), 0
        while True:
            if abs(ultra - int(self.obj.ultra)) > 2: #초음파값 변경
                ultra = int(self.obj.ultra)
                if self.obj.people > 0:              #사람 들어옴
                    self.flag = 1
                    print("들어왔습니다.")
                    arr = np.append(a, np.array([self.obj.temp, self.obj.humi, self.obj.peak, self.obj.pir]))
                    b = np.append(b, loaded_model.predict([arr]))
                    self.result = np.append(arr, b[-1])
                    print(self.result)
                    self.labeling()
                    self.c = b[-1]  ##list없앰
                    self.pir = self.obj.pir
                    time.sleep(0.98)
                else:                                #사람 나감
                    self.flag = 0
                    print("나갔습니다.")
                    time.sleep(0.98)
                    continue

            else: #초음파값 변경X
                ultra = int(self.obj.ultra)
                if self.obj.people > 0:              # 사람 그냥 있음
                    self.flag = 1
                    print("계속 있습니다.")
                    arr = np.append(a, np.array([self.obj.temp, self.obj.humi, self.obj.peak, self.obj.pir]))
                    b = np.append(b, loaded_model.predict([arr]))
                    self.result = np.append(arr, b[-1])
                    print(self.result)
                    self.labeling()
                    self.c = b[-1]  ##list없앰
                    self.pir = self.obj.pir
                    time.sleep(0.98)
                else:                                # 사람 원래 없었음
                    self.flag = 0
                    print("사람이 없습니다")
                    time.sleep(0.98)
                    continue

            '''
            #print('temp', self.obj.temp)
            #print('humi', self.obj.humi)
            #print('decibel', self.obj.peak)
            #print('PIR', self.obj.pir)
            #print('\n')
            arr = np.append(a, np.array([self.obj.temp, self.obj.humi, self.obj.peak, self.obj.pir]))
            b = np.append(b, loaded_model.predict([arr]))
            ###self.result = np.append(arr,b[-1])
            ####self.labeling()
            #print(arr, '\n', b)
            #print('result:', self.result)
            self.c = b[-1] ##list없앰
            self.pir = self.obj.pir
            #print(self.c)
            #print("\n---------------------------\n")
            time.sleep(0.98)
            '''

    def labeling(self):
        p = int(self.obj.pir)
        t = int(self.obj.temp)
        h = int(self.obj.humi)
        m = self.obj.peak
        if p == 1:  # 1움직임있음
            r = '1'
            if m > 65:  # 2시끄러움
                r += '1'
                if t < 18:  # 온도낮음
                    r += '0'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'
                elif 18 <= t and t <= 30:  # 온도적정
                    r += '1'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'
                elif t > 30:  # 온도높음
                    r += '2'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'

            elif m <= 65:  # 2조용함
                r += '0'
                if t < 18:  # 온도낮음
                    r += '0'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'
                elif 18 <= t and t <= 30:  # 온도적정
                    r += '1'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'
                elif t > 30:  # 온도높음
                    r += '2'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'

        else:  # 1움직임없음
            r = '0'
            if m > 65:  # 2시끄러움
                r += '1'
                if t < 18:  # 온도낮음
                    r += '0'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'
                elif 18 <= t and t <= 30:  # 온도적정
                    r += '1'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'
                elif t > 30:  # 온도높음
                    r += '2'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'

            elif m <= 65:  # 2조용함
                r += '0'
                if t < 18:  # 온도낮음
                    r += '0'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'
                elif 18 <= t and t <= 30:  # 온도적정
                    r += '1'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'
                elif t > 30:  # 온도높음
                    r += '2'
                    if h > 75:  # 습도높음
                        r += '2'
                    elif h <= 75 and h >= 35:  # 습도적정
                        r += '1'
                    elif h < 35:  # 습도낮음
                        r += '0'
        self.r = r
        #print('self.r:', self.r)
        r = ''

    '''
    def sentence(self):
        while True:
            r = ''
            if self.c[1] == '1':
                r += '소음 주의하세요'
                if self.c[2] == '0':
                    r += '온도가 낮습니다'
                    if self.c[3] == '0':
                        r += '습도가 낮습니다'
                    elif self.c[3] == '2':
                        r += '습도가 높습니다'
                elif self.c[2] == '2':
                    r += '온도가 높습니다'
                    if self.c[3] == '0':
                        r += '습도가 낮습니다'
                    elif self.c[3] == '2':
                        r += '습도가 높습니다'
            else:
                if self.c[2] == '0':
                    r += '온도가 낮습니다'
                    if self.c[3] == '0':
                        r += '습도가 낮습니다'
                    elif self.c[3] == '2':
                        r += '습도가 높습니다'
                elif self.c[2] == '2':
                    r += '온도가 높습니다'
                    if self.c[3] == '0':
                        r += '습도가 낮습니다'
                    elif self.c[3] == '2':
                        r += '습도가 높습니다'
            print(r)
            time.sleep(0.8)
    '''

'''
if __name__ == "__main__":
    main1()
'''