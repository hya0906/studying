import cv2
import face_recognition
import pickle
from PIL import Image, ImageDraw
import numpy as np
import collections
import time
#MakeData-데이터 만드는 부분 개수 통일화 필요!!
#속도개선을 위해 numpy로 만듦. 시험은 더 해봐야할듯

image_file = './TestImages/yuna_hong.jpg'
encoding_file = '.\CreateModel/face_model_15.pickle'
pic_count = {'barack':15, "IU":15, "michelle":15, "yuna_hong":15 } #파일개수지정으로 수정필요

aa = pic_count.get('barack')
bb = pic_count.get('IU')
cc = pic_count.get('michelle')
dd = pic_count.get('yuna_hong')

def display_features(image):
    # model=“small” only returns 5 points but is faster.
    face_landmarks_list = face_recognition.face_landmarks(image, model='small')
    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')  # Creates an object that can be used to draw in the given image.
        for feature, points in face_landmarks.items():
            d.point(points, fill=(0, 0, 255)) #red
    image = np.array(pil_image)
    return image

#Lister=[]
def detectAndDisplay(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = display_features(image) # which displays face features
    face_locations = face_recognition.face_locations(rgb, model="HOG")
    encodings = face_recognition.face_encodings(rgb, face_locations, model='small')

    for encoding in encodings:  # 인풋 사진의 얼굴 하나에 대해
        matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=0.4)

        Lister = np.zeros(shape=(4,15), dtype=np.int8)
        print(Lister)

        all = np.concatenate((matches[0:aa],matches[aa:aa + bb],matches[aa + bb: aa + bb + cc],matches[aa + bb + cc: aa + bb + cc + dd]),axis=0)
        all = np.reshape(all, (4,15))
        print("all",all)

        c=[]
        for i in all:
            #n=i.count(True)
            n = collections.Counter(i)
            c.append(n)
        print(c)
        print(c[0].most_common())
        a = max(list(n.values()))
        print(a)
        for aaa,i in enumerate(c):
            for j in i:
                print("i",j)
                if j==True:
                    print(aaa,"yes")
                    aaaa=aaa



        name = list(pic_count.keys())[aaaa]

    for (top, right, bottom, left) in face_locations:
        y = top - 15
        color = (255, 255, 0)
        line = 1
        if (name == 'unknown'):
            color = (255, 255, 255)

        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)
    end_time = time.time()
    # 소요시간 체크
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    return image

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame = detectAndDisplay(frame)
            cv2.imshow('Recognition', frame)
            key = cv2.waitKey(1) & 0xFF # ESC를 누르면 종료
            if (key == 27):
                break
        #time.sleep(0.005)

if __name__ == "__main__":
    data = pickle.loads(open(encoding_file, 'rb').read())
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
