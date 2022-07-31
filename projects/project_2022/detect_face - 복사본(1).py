import cv2
import face_recognition
import pickle
from PIL import Image, ImageDraw
import numpy as np

image_file = './TestImages/yuna_hong.jpg'
encoding_file = '.\CreateModel/face_model.pickle'


def detectAndDisplay(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)
    #print("pil", pil_image)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')  # Creates an object that can be used to draw in the given image.
        #print("d", d)
        for feature, points in face_landmarks.items():
            d.point(points, fill=(0, 0, 255))

    image = np.array(pil_image)
    face_locations = face_recognition.face_locations(rgb, model="HOG")
    for face_location in face_locations:
        print(face_location)
        top, right, bottom, left = face_location
        # face_image = image[top - 50:bottom + 50, left - 50:right + 50] #full faces
        face_image = image[top:bottom, left:right]

        #pil_image = Image.fromarray(face_image)
        #pil_image.show()
    encodings = face_recognition.face_encodings(rgb, face_locations)
    names = []  # for recognized faces

    for encoding in encodings:  # 인풋 사진의 얼굴 하나에 대해
        matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=0.4)
        name = 'unknown'
        #print("matches",matches)
        if True in matches:  # matches 에 true 가 있다면, true 의 인덱스를 뽑아
            matchedIndxs = []
            for (i, b) in enumerate(matches):
                if (b == True):
                    matchedIndxs.append(i)
            # print(matchedIndxs) # [8,9,10,11,12,13,14,15]

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
            line = 1

        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, color, line)
    return image

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame = detectAndDisplay(frame)
            cv2.imshow('Recognition', frame)
            # ESC를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if (key == 27):
                break

data = pickle.loads(open(encoding_file, 'rb').read())

main()
cv2.waitKey(0)
cv2.destroyAllWindows()
