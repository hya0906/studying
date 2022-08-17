from datetime import datetime
import pymysql
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import csv

from pynput import keyboard

my_host='127.0.0.1'
my_user = 'root'
my_passwd = '1234'
my_db = 'raspi_db'
my_port = 3306

cred = credentials.Certificate("abcd-27823-firebase-adminsdk-6hp4e-38c6900af9.json")
firebase_admin.initialize_app(cred)
db =  firestore.client()

Maria = pymysql.connect(host = my_host, user = my_user, passwd = my_passwd, db = my_db, port = my_port)
Cursor = Maria.cursor()

class Record_data:
    def __init__(self):
        self.temp = 0
        self.humi = 0
        self.wave = 0
        self.decibel = 0
        self.pir = 0
        self.date = ''
        self.r = ''

    def set_data(self, sensor):
        self.temp = int(sensor.temp)
        self.humi = int(sensor.humi)
        self.wave = int(sensor.ultra)
        self.decibel = int(sensor.peak)
        self.pir = sensor.pir
        self.date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        
        
    def record_to_maria(self):
        print("{}\nTEMP = {}HUMI={}WAVE={}DECIBEL={}".format(self.date, self.temp, self.humi, self.wave, self.decibel))

        #data input at SQL 
        Query = 'INSERT INTO raw_data VALUES(%s, %s, %s, %s, %s, %s)'   #table == raw_data
        values = [(self.date, self.temp, self.humi, self.wave, self.decibel, self.pir, )]
        
        Cursor.executemany(Query, values)    
        Maria.commit()          #Input and save data at Server

    def record_to_fire(self):
        doc_ref = db.collection(u'hanium_db')
        data = {u'TEMP' : self.temp, u'HUMI' : self.humi, u'WAVE' : self.wave, u'DECIBEL' : self.decibel, u'PIR' : self.pir, u'result' : self.r}
        doc_ref.document(u'{}'.format(self.date)).set(data)
        
    def labeling(self):        
        p = self.pir
        t = self.temp
        h = self.humi
        m = self.decibel
        with open("data_2.csv", "w", newline='') as csv_file:
            fieldnames = ['date', 'temp', 'humid', 'decibel', 'PIR', 'result']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            current_time = datetime.now()
            if p==1: #1움직임있음
              r = '1'
              if m>65: #2시끄러움
                 r += '1'
                 if t<18: #온도낮음
                  r += '0'
                  if h>75: #습도높음
                    r += '2'
                  elif h<=75 and h>=35: #습도적정
                    r += '1'
                  elif h<35: #습도낮음
                    r += '0'
                 elif 18<=t and t<=30: #온도적정
                  r += '1'
                  if h>75: #습도높음
                    r += '2'
                  elif h<=75 and h>=35: #습도적정
                    r += '1'
                  elif h<35: #습도낮음
                    r += '0'
                 elif t>30: #온도높음
                  r += '2'
                  if h>75: #습도높음
                    r += '2'
                  elif h<=75 and h>=35: #습도적정
                    r += '1'
                  elif h<35: #습도낮음
                    r += '0'

              elif m<=65: #2조용함
                r += '0'
                if t<18: #온도낮음
                  r += '0'
                  if h>75: #습도높음
                    r += '2'
                  elif h<=75 and h>=35: #습도적정
                    r += '1'
                  elif h<35: #습도낮음
                    r += '0'
                elif 18<=t and t<=30: #온도적정
                  r += '1'
                  if h>75: #습도높음
                    r += '2'
                  elif h<=75 and h>=35: #습도적정
                    r += '1'
                  elif h<35: #습도낮음
                    r += '0'
                elif t>30: #온도높음
                  r += '2'
                  if h>75: #습도높음
                    r += '2'
                  elif h<=75 and h>=35: #습도적정
                    r += '1'
                  elif h<35: #습도낮음
                    r += '0'
                   
            else: #1움직임없음
              r = '0'
              if m>65: #2시끄러움
                r += '1'
                if t<18: #온도낮음
                  r += '0'
                  if h>75: #습도높음
                    r += '2'
                  elif h<=75 and h>=35: #습도적정
                    r += '1'
                  elif h<35: #습도낮음
                    r += '0'
                elif 18<=t and t<=30: #온도적정
                  r += '1'
                  if h>75: #습도높음
                    r += '2'
                  elif h<=75 and h>=35: #습도적정
                    r += '1'
                  elif h<35: #습도낮음
                    r += '0'
                elif t>30: #온도높음
                  r += '2'
                  if h>75: #습도높음
                    r += '2'
                  elif h<=75 and h>=35: #습도적정
                    r += '1'
                  elif h<35: #습도낮음
                    r += '0'

            self.r = r
            writer.writerow({'date': current_time, 'temp': self.temp, 'humid': self.humi, 'decibel': self.decibel, 'PIR': self.pir, 'result': self.r})
            #print(self.r)

    
        
    def record_data(self, sensor):
        self.set_data(sensor)
        self.labeling()
        self.record_to_maria()
        self.record_to_fire()
