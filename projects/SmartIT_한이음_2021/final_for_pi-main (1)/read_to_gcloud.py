#--gcloud tts
from google.cloud import texttospeech

#--mediaplayer
import os

#--firebase import
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import record_data

db =  firestore.client()

#gcloud tts
client = texttospeech.TextToSpeechClient()



class read_data:
    def __init__(self):
        self.temp = 0
        self.humi = 0
        self.wave = 0
        self.decibel = 0
        self.pir = 0
        self.people= ''
        self.date = ''
        self.result = ''
        self.txt = ""               #for tts
        self.txt_pir_add = ""       #pir txt
        self.txt_temp_add = ""      #temp text
        self.txt_humi_add = ""      #humi text
        self.txt_people_add = ""    #people text
        self.txt_decibel_add =""    #decibel text
        

    def get_data(self):
        print(self.people)
        #-- get latest document ID
        user_ref = db.collection(u'hanium_db')
        docs = user_ref.stream()
        *_, last = docs
        self.date = last.id

        #-- get values from document
        doc_ref = db.collection('hanium_db').document(f'{self.date}')
        #get data from firestore
        doc = doc_ref.get().to_dict()
    
        print(f'DATE: {self.date}')
        print(f'from databas: {doc}')
    
        # dict to string 
        self.result = doc.get('result')
        self.people = doc.get('PEOPLE')

    def txt_people(self):
        result_people = self.people
        hour = self.date[11:13]
        minute = self.date[14:16]
        print(result_people)
        if result_people == "0":
            self.txt_people_add = "지금은 사람이 없습니다. 소등 및 각종 디바이스들을 종료합니다. "
        else:
            self.txt_people_add = "현재 시간 {}시 {}분,".format(hour,minute)
    

#    def txt_pir(self):
#        result_pir = self.result[:1]
        
    def txt_decibel(self):
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
    
    def add_txt(self):
        
        if self.people == "0":
            self.txt = self.txt_people_add
        else:
            self.txt = self.txt_people_add + self.txt_decibel_add + self.txt_temp_add + self.txt_humi_add
        print(self.txt)
    
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
        #os.system('omxplayer -o local output.mp3 "{}" /&'.format(address))
        print("gcloud end")

        
    def tts_data(self):
        print('read_data start')

        self.get_data()
        self.txt_people()
        self.txt_decibel()
        self.txt_temp()
        self.txt_humi()
        self.add_txt()
        #self.make_tts()


if __name__ == '__main__':
    x=  read_data()
    x.tts_data()
