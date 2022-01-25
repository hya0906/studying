import numpy as np
import joblib
import time
from read_to_gcloud_new import read_data as tts


class Machine:
    def __init__(self):
        self.obj = self
        self.c, self.pir = 0, 0
        self.result = np.array([])
        self.flag = 0
        self.start, self.end = 0, 0
        self.timer_for_noise_alert_start, self.timer_for_noise_alert_end = 0, 0
        self.timer_for_alert_time_start, self_timer_for_alert_time_end = 0, 0
        self.timer_flag = 0
        self.count = 0

    def get_obj(self, object):
        self.obj = object  # Sensor의 객체


    def decision(self):
        # PIR(0,1),마이크(0,1),온도(0,1,2),습도(0,1,2) 순
        loaded_model = joblib.load('demo2_model.pkl') ##??
        time.sleep(2) #???????????
        a, b, ultra = np.array([]), np.array([]), 0
        while True:
            if abs(ultra - int(self.obj.ultra)) > 2: #초음파값 변경
                ultra = int(self.obj.ultra)
                float(self.obj.people)
                if int(self.obj.people) > 0:              #사람 들어옴
                    self.flag = 1
                    ##print("들어왔습니다.")
                    arr = np.append(a, np.array([self.obj.temp, self.obj.humi, self.obj.peak, self.obj.pir]))
                    b = np.append(b, loaded_model.predict([arr]))
                    self.result = np.append(arr, b[-1])
                    ##print(self.result)
                    self.labeling()
                    self.c = b[-1]  ##list없앰
                    self.pir = self.obj.pir
                    self.pir_alert()
                    time.sleep(0.98)
                else:                                #사람 나감
                    self.flag = 0
                    ##print("나갔습니다.")
                    time.sleep(0.98)
                    continue

            else: #초음파값 변경X
                ultra = int(self.obj.ultra)
                if self.obj.people > 0:              # 사람 그냥 있음
                    self.flag = 1
                    ##print("계속 있습니다.")
                    arr = np.append(a, np.array([self.obj.temp, self.obj.humi, self.obj.peak, self.obj.pir]))
                    b = np.append(b, loaded_model.predict([arr]))
                    self.result = np.append(arr, b[-1])
                    ##print(self.result)
                    self.labeling()
                    self.c = b[-1]  ##list없앰
                    self.pir = self.obj.pir
                    self.pir_alert()
                    time.sleep(0.98)
                else:                                # 사람 원래 없었음
                    self.flag = 0
                    ##print("사람이 없습니다")
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

    #tts때문에 막음
    def tts_for_alert(self,alert_flag):
         x = tts
         if alert_flag == 0:
             x.tts_data_for_noise(self)
         elif alert_flag == 1:
             x.tts_data_for_pir(self)


    def alert(self):
        if self.obj.peak > 60:
            self.timer_for_noise_alert_end = time.time()
        else:
            self.timer_for_noise_alert_start = time.time()

        if int(self.timer_for_noise_alert_end - self.timer_for_noise_alert_start) >= 180:
            self.tts_for_alert(0)
            self.timer_for_noise_alert_start = time.time()

    def pir_alert(self):
        if int(self.obj.pir) == 0:
            self.start = time.time()
        #print(float(time.time() - self.start))
        if time.time() - self.start > 28800: #일단 1초
            self.tts_for_alert(1)
            #print("tts작동")
            pass

    def ppm_alert(self):
        #print(self.obj.ppm)
        #print("count",self.count)
        if not type(float(self.obj.ppm)) == float:
            pass
        else:
            if float(self.obj.ppm) > 20: #ppm20이상일때
                self.count += 1
            else:
                self.count = 0
            if self.flag == 0 and self.count%5 == 1:#사람 있으면 1초마다 움직이므로
                # self.tts_for_alert(2)
                #print("tts작동")
                pass
            elif self.flag == 1 and self.count == 0: #사람 없으면 5초대기하므로
                # self.tts_for_alert(2)
                #print("tts작동")
                pass
            if self.count % 5 != 0:
                self.count = 0

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
if __name__ == "__main__":
    main1()
'''
