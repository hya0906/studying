import cv2
import face_recognition
import pickle
from PIL import Image, ImageDraw
import numpy as np
import time


image_file = './TestImages/yuna_hong.jpg'
encoding_file = '.\CreateModel/resmall_face_model.pickle'

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

def detectAndDisplay(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = display_features(image) #which displays face features
    face_locations = face_recognition.face_locations(rgb, model="HOG")
    encodings = face_recognition.face_encodings(rgb, face_locations, model='small')
    names = []  # for recognized faces

    for encoding in encodings:  # 인풋 사진의 얼굴 하나에 대해
        matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=0.4)
        #print(matches)
        #print(matches.index(True))
        name = 'unknown'
        #print("matches",matches)
        if True in matches:  # matches 에 true 가 있다면, true 의 인덱스를 뽑아
            #print(True)
            matchedIndxs = []
            for (i, b) in enumerate(matches):
                if (b == True):
                    matchedIndxs.append(i) # [8,9,10,11,12,13,14,15]
                    #전체의 몇퍼센트정도 되면 그냥 다음단계로 넘어가기?
            counts = {}
            for i in matchedIndxs:
                name = data['names'][i]  # dataset 의 8번째 이름을 name으로 지정.
                counts[name] = 0
                counts[name] = counts.get(name) + 1  # count={'rory':1}
            name = max(counts, key=counts.get)
        names.append(name)

    for ((top, right, bottom, left), name) in zip(face_locations, names):
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
        time.sleep(0.005)

if __name__ == "__main__":
    data = pickle.loads(open(encoding_file, 'rb').read())
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
