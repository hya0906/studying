import glob
import librosa #음원 데이터를 분석
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical
import moviepy.editor as mp
from keras.models import load_model #save the model
import contextlib
import wave
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #remove warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#(1=Angry, 2=Disgust, 3=Fear, 4=Happy, 5=Sad, 6=Surprise, 7=Neutral)

def extract_feature(file_name): #파일의 음성 특징 추출
    X, sample_rate = librosa.load(file_name, sr=96000)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"): #처음 기본 학습파일의 특징을 추출
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('/')[2].split('-')[2])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels): #레이블 원핫인코딩
    n_labels = len(labels)+1
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels+1))
    one_hot_encode[np.arange(n_labels), labels] = 1
    one_hot_encode=np.delete(one_hot_encode, 0, axis=1)
    return one_hot_encode

# model layers
def create_model(activation_function='relu', init_type='normal', dropout_rate=0.25):  # 0.25
    global n_dim, n_classes, n_hidden_units_1, n_hidden_units_2
    # 층마다 차원
    model = Sequential()
    model.add(Dense(n_hidden_units_1, kernel_regularizer=regularizers.l2(0.004), input_dim=n_dim, init=init_type,
                        activation=activation_function, bias_initializer='zeros'))  # 1
    model.add(Dense(n_hidden_units_2, kernel_regularizer=regularizers.l2(0.004), init=init_type,
                        activation=activation_function,
                        bias_initializer='zeros'))  # 2
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes, init=init_type, activation='softmax'))  # output
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0004), metrics=['accuracy'])
    return model

class Test:
    global saved_model,result
    saved_model = load_model('emotion_result.h5')
    def __init__(self):
        self.i=0
        self.name = []
        self.tr = 1

    def trim_audio(self, audio_file, save_file,length): #오디오 자름
        sr = 96000
        if length <= 30:
            self.tr = 1
        elif length > 30:
            self.tr = int(length//20)
        sec = int(length)//self.tr
        a = 0
        y, sr = librosa.load(audio_file, sr=sr)
        while(sr*(sec*(a+1))< len(y)):
            ny = y[sr*(0+sec*a):sr*(sec*(a+1))]
            librosa.output.write_wav(save_file + str(self.i+1) +'.wav', ny, sr)
            name = save_file + str(self.i + 1) + ".wav"
            self.name.append(name.split("/")[-1])
            print(name + "완료")
            self.i += 1
            a += 1

    def make_test(self,main_dir, sub_dir): #특징추출하고 features,labels만들기
        i = 0
        features, labels = np.empty((0, 193)), np.empty(0)
        while (i < len(sub_dir)):
            fn = main_dir + sub_dir[i]
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split('/')[-1].split('-')[2])
            #print(labels)
            i += 1
        labels = np.array(labels, dtype=np.int)
        labels = to_categorical(labels - 1, 8)
        return features, labels

    def audio_feature(self):
        # 데이터 추출. 이미 데이터 추출되었으면 시간 오래 걸려서 생략
        # change the main_dir acordingly....
        main_dir = 'D:/MyFile/Audio_speech'
        sub_dir = os.listdir(main_dir)
        print("\ncollecting features and labels...")
        print("\nthis will take some time...")
        features, labels = parse_audio_files(main_dir, sub_dir)
        print("done")
        np.save('X', features)  # 저장
        # one hot encoding labels
        labels = one_hot_encode(labels)
        np.save('y', labels)
        labels = one_hot_encode('08')
        print(labels)

    def video_feature(self): #맨 처음 데이터 특징추출하기 위한 부분-npy로 저장
        # 영상
        main_dir = os.getcwd()
        main_dir = main_dir + "/video/"
        sub_dir = os.listdir(main_dir)
        print('subdir', sub_dir)

        for i in range(0, len(os.listdir(main_dir))):
            head, tail = os.path.split(sub_dir[i])
            clip = mp.VideoFileClip("./video/" + tail)
            length = clip.duration
            print("length:", length)
            print("1-", "./video/" + tail)
            tail = tail[0:2]
            print("2-", tail)
            audio_file = "00-00-" + tail + "-00-00-00-00.wav"
            print("3-", audio_file)
            clip.audio.write_audiofile(audio_file)
            print("4trim-", audio_file)
            save_file = "./sound/" + audio_file.replace("0.wav", "")
            print(save_file)
            self.trim_audio(audio_file, save_file, length)

        main_dir = os.getcwd()
        main_dir = main_dir + "/sound/"
        sub_dir = os.listdir(main_dir)
        print('subdir', sub_dir)
        features, labels = self.make_test(main_dir, sub_dir)
        np.save('f', features)
        np.save('l', labels)

    def musical_feature(self): #맨 처음 뮤지컬 음악 학습할때 필요
        # 뮤지컬
        main_dir = 'D:/MyFile/Musical/'
        sub_dir = os.listdir(main_dir)
        print("\ncollecting features and labels...")
        print("\nthis will take some time...")
        features, labels = self.make_test(main_dir, sub_dir)  ##
        print("done")
        np.save('Xmusic', features)  # 저장
        np.save('ymusic', labels)

    def get_feature_save(self):
        self.audio_feature()
        self.video_feature()
        self.musical_feature()

    def get_feature(self, fName, flag, q):
        global saved_model, result
        main_dir = fName
        head, tail = os.path.split(main_dir)
        if flag == 1:
            clip = mp.VideoFileClip("./video/" + tail)
            length = clip.duration
            tail = tail[0:2]
        elif flag == 2:
            with contextlib.closing(wave.open(main_dir, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                length = frames / float(rate)
            tail = tail[0:2]
        audio_file = "00-00-" + tail + "-00-00-00-00.wav"
        save_file = "./sound/" + audio_file.replace("0.wav", "")
        self.trim_audio(main_dir, save_file, length)
        #print("name", self.name)
        features, labels = self.make_test("./sound/", self.name)
        predict = saved_model.predict(features, batch_size=4)
        #print(predict[0])
        #print('\n영상 예측값:', np.argmax(predict, 1) + 1, '\ny:\t\t  ', np.argmax(labels, 1) + 1)
        q.put(np.argmax(predict, 1) + 1)
        q.put(length)
        q.put(self.tr)

    def main(self):
        features = np.load('f.npy')  # 영상
        labels = np.load('l.npy')
        Xmusic = np.load('Xmusic.npy')  # 뮤지컬
        ymusic = np.load('ymusic.npy')
        X = np.load('X.npy')  # 기본데이터
        y = np.load('y.npy')
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=60)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=60)
        Xmusic, a, ymusic, b = train_test_split(Xmusic, ymusic, test_size=0.05, random_state=60)  # 5개
        features, c, labels, d = train_test_split(features, labels, test_size=0.07, random_state=60)  # 2개
        train_x = np.append(train_x, a, axis=0)
        train_x = np.append(train_x, c, axis=0)
        train_y = np.append(train_y, b, axis=0)
        train_y = np.append(train_y, d, axis=0)

        global n_dim, n_classes, n_hidden_units_1, n_hidden_units_2
        n_dim = train_x.shape[1]  # 193
        n_classes = train_y.shape[1]  # 8
        n_hidden_units_1 = n_dim
        n_hidden_units_2 = 400  # 400

        model = create_model()  # 모델생성
        epoch = 120
        train_history = model.fit(train_x, train_y, epochs=epoch, batch_size=15, validation_data=(val_x, val_y), verbose = 2)
        predict = model.predict(test_x, batch_size=4)
        (test_loss, test_acc) = model.evaluate(test_x, test_y, verbose=0)

        print('\n테스트데이터 정확도:', test_acc)
        #######
        predict = model.predict(Xmusic)
        print('\n뮤지컬음악 예측값:', np.argmax(predict, 1) + 1, '\ny:\t  ', np.argmax(ymusic, 1) + 1)
        # print(predict)

        (test_loss, test_acc) = model.evaluate(Xmusic, ymusic, verbose=0)
        print('\n뮤지컬음악 테스트 정확도:', test_acc)
        ###########
        predict = model.predict(features)
        print('\n영상 예측값:', np.argmax(predict, 1) + 1, '\ny:\t\t  ', np.argmax(labels, 1) + 1)
        # print(predict)

        (test_loss, test_acc) = model.evaluate(features, labels, verbose=0)
        print('\n영상 테스트 정확도:', test_acc)

        model.save('emotion_result.h5')
        ##############
        # 그래프
        epochs = range(1, epoch + 1)
        accuracy = train_history.history['accuracy']
        val_accuracy = train_history.history['val_accuracy']
        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']

        plt.subplot(1, 2, 1)
        plt.plot(epochs, accuracy, 'b', label='accuracy')
        plt.plot(epochs, val_accuracy, 'g', label='val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'r', label='loss')
        plt.plot(epochs, val_loss, 'k', label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.grid()
        plt.tight_layout()  # 떨어져있게 간격조정
        plt.show()

if __name__ == "__main__":
    a = Test()
    #a.get_feature_save()
    a.main()