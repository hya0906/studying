from datetime import datetime
import pymysql
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime, timezone

my_host='127.0.0.1'
my_user = 'root'
my_passwd = '1234'
my_db = 'raspi_db'
my_port = 3306

# cred = credentials.Certificate("./hanium-321506-firebase-adminsdk-c68ph-5fc036f7c4.json")
# firebase_admin.initialize_app(cred)
# db = firestore.client()

#import read_to_gcloud as tts
#from read_to_gcloud import read_data as tts

Maria = pymysql.connect(host = my_host, user = my_user, passwd = my_passwd, db = my_db, port = my_port)
Cursor = Maria.cursor()

class Record_data:
    def __init__(self):
        self.temp, self.humi = 0, 0
        self.wave, self.decibel, self.pir = 0, 0, 0
        self.people, self.date, self.r = '', '', ''
        self.sensor, self.machine = self, self

    def set_obj(self, sensor, machine):
        self.sensor = sensor
        self.machine = machine


    def set_data(self):
        self.temp = int(self.sensor.temp)
        self.humi = int(self.sensor.humi)
        self.wave = int(self.sensor.ultra)
        self.decibel = int(self.sensor.peak)
        self.date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        self.people = self.sensor.people
        try:
            self.MQ2 = self.sensor.ppm
        except ValueError:
            pass


    def record_to_maria(self):
        # print("{}\nTEMP = {} HUMI={} DECIBEL={} PIR={} WAVE={} GAS={} result={} PEOPLE (from record_data)\n".format(self.date,
        #                                                                                               self.temp,
        #                                                                                               self.humi,
        #                                                                                               self.decibel,
        #                                                                                               self.machine.pir,
        #                                                                                               self.wave,
        #                                                                                               self.MQ2,
        #                                                                                               self.machine.c,
        #                                                                                               self.people))

        # data input at SQL
        Query = 'INSERT INTO data VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)'  # table == raw_data
        values = [(self.date, self.temp, self.humi, self.decibel, self.machine.pir, self.wave, self.MQ2, self.machine.c, self.people)]

        Cursor.executemany(Query, values)
        Maria.commit()  # Input and save data at Server


    # def record_to_fire(self):
    #     doc_ref = db.collection(u'hanium_db')
    #     data = {u'TEMP' : self.temp, u'HUMI' : self.humi, u'WAVE' : self.wave, u'DECIBEL' : self.decibel, u'PIR' : self.pir, u'result' : self.machine.c, u'PEOPLE': self.people}
    #     #doc_ref.document(u'{}'.format(self.date)).set(data)

        
    def record_data(self):
        self.set_data()
        self.record_to_maria()
        #self.record_to_fire()
