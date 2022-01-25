#--gcloud tts
import google.cloud

#--mediaplayer
import os

#--firebase import
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud import texttospeech

# db = firestore.client()

#--mariadb
import pymysql

my_host='127.0.0.1'
my_user = 'root'
my_passwd = '1234'
my_db = 'raspi_db'
my_port = 3306

Maria = pymysql.connect(host = my_host, user = my_user, passwd = my_passwd, db = my_db, port = my_port)
Cursor = Maria.cursor()



#gcloud tts
client = texttospeech.TextToSpeechClient()

address = "/C:/Users/johnK/pythonProject/test)"

class read_data:
    def __init__(self):
        self.temp = 0
        self.humi = 0
        self.wave = 0
        self.decibel = 0
        self.pir = 0
        self.ppm = 0
        self.people= ''
        self.date = ''
        self.result = ''
        self.txt = ''               #for tts
        self.txt_pir_add = ''       #pir txt
        self.txt_temp_add = ''      #temp text
        self.txt_humi_add = ''      #humi text
        self.txt_people_add = ''    #people text
        self.txt_decibel_add = ''   #decibel text
        self.txt_ppm_add = ''       #ppm text

    def get_data(self):
        #print(self.people)
        
        # #-- get latest document ID
        # user_ref = db.collection(u'hanium_db')
        # query = user_ref.order_by(u"DATE", direction =firestore.Query.DESCENDING).limit(1)
        # docs= query.get()
        # doc = docs[0].to_dict()
        #
        # self.date = doc["DATE"]
        #
        # print(f'DATE: {self.date}')
        # print(f'from databas: {doc}')
        #
        # # dict to string
        # self.result = doc["result"]
        # self.people = doc['PEOPLE']

        #--get data from mariadb
        query = "select DATE, RESULT, PEOPLE from data order by DATE DESC LIMIT 1"
        Cursor.execute(query)
        data_result = (Cursor.fetchone())
        self.date = data_result[0]
        self.result = data_result[1]
        self.people = data_result[2]
        print(self.date, self.result, self.people)


    def txt_people(self):
        result_people = self.people
        hour = self.date[11:13]
        minute = self.date[14:16]
        print(result_people)
        if result_people == "0":
            self.txt_people_add = "지금은 사람이 없습니다. 소등 및 각종 디바이스들을 종료합니다. "
        else:
            self.txt_people_add = "현재 시간 {}시 {}분,".format(hour,minute)
            
    def txt_decibel(self):
        # print(type(self.result))
        result_decibel = self.result[1:2]
        if result_decibel == "0":
            self.txt_decibel_add = "지금은 소음이 없습니다."
        else:
            self.txt_decibel_add = "소음이 발생하고 있습니다. 필요에 따라 창문을 닫아주세요."
            
    def txt_temp(self):
        result_temp = self.result[2:3]
    
        if result_temp == "2":
            self.txt_temp_add = "온도가 높습니다. 에어컨을 작동 시킬게요. 야외활동을 되도록이면 자제해주시길 바랍니다."
        elif result_temp == "1":
            self.txt_temp_add = "적정 온도입니다."
        else:
            self.txt_temp_add = "온도가 낮습니다. 난방기를 작동합니다. 나가실 때 적절한 겉옷 챙겨주시길 바랍니다."
    
    def txt_humi(self):
        result_humi = self.result[3:]
    
        if result_humi== "2":
            self.txt_humi_add = "습도가 높습니다. 제습기를 실행시킬게요."
        elif result_humi == "1":
            self.txt_humi_add = "적정 습도입니다."
        else:
            self.txt_humi_add = "습도가 낮습니다. 가습기를 실행시킬게요."

    def txt_ppm(self):
        result_ppm = float(self.ppm)
        if result_ppm > 3.14:   # 바꿔야됨
            self.tts_data_for_ppm()
    
    def add_txt(self):
        if self.people == "0":
            self.txt = self.txt_people_add
        else:
            self.txt = self.txt_people_add + self.txt_decibel_add + self.txt_temp_add + self.txt_humi_add
        #print(self.txt)


    
    def make_tts(self):
        print("make_tts start")
        synthesis_input = texttospeech.SynthesisInput(text=self.txt)
        print(self.txt)
        voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        
        #audio
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3)

        #all together
        response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config)


        with open('output.mp3', 'wb') as out:
            out.write(response.audio_content)
            print('Audio content written to file "output.mp3"')
    
        #speaker
        os.system('omxplayer -o local output.mp3 "{}" /&'.format(address))
        print("gcloud end")

        
    def tts_data(self):
        self.get_data()
        self.txt_people()
        self.txt_decibel()
        self.txt_temp()
        self.txt_humi()
        self.add_txt()
        #self.make_tts()

    def tts_data_for_ppm(self):
        txt = "연기가 감지되었습니다."
        print(txt)
        #self.make_tts()

    def tts_data_for_noise(self):
        txt = "소음 경보, 소음 경보, 3분 이상 소음이 발생하고 있습니다."
        print(txt)
        #self.make_tts()

    def tts_data_for_pir(self):
        txt = "사용자의 움직임이 7시간 이상 감지되지 않습니다 위급상황대처를 준비해주세요."
        print(txt)
        #self.make_tts()


if __name__ == '__main__':
    x =  read_data()
    x.tts_data()
