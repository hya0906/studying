# 모듈 호출
import numpy as np
import cv2 as cv
import os
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import time

prev_time = 0
FPS = 10

# Face detection XML load and trained model loading
face_detection = cv.CascadeClassifier('C:/Users/USER/Desktop/files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('C:/Users/USER/Desktop/files/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

# VideoCapture 객체 정의
cap = cv.VideoCapture('새 비디오 만들기.mp4')

# 프레임 너비/높이, 초당 프레임 수 확인
width = cap.get(cv.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = cap.get(cv.CAP_PROP_FPS) # 또는 cap.get(5)
print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(width, height, fps))

#while cap.isOpened(): # cap 정상동작 확인
while True:
    ret, frame = cap.read(cv.COLOR_BGR2GRAY)
    # 프레임이 올바르게 읽히면 ret은 True
    if not ret:
        print("프레임을 수신할 수 없습니다(스트림 끝?). 종료 중 ...")
        break
    current_time = time.time() - prev_time
    if (ret is True) and (current_time > 1. / FPS):
        prev_time = time.time()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Face detection in frame
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Create empty image (height,width,channel)
        canvas = np.zeros((350, 300, 3), dtype="uint8")

        # Perform emotion recognition only when face is detected

        if len(faces) > 0:
            # For the largest image
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            # Resize the image to 48x48 for neural network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Emotion predict
            preds = emotion_classifier.predict(roi)[0]
            print(preds)
            emotion_probability = np.max(preds)
            # print(type(emotion_probability[0].astype('float64')),emotion_probability[0].astype('float64'),dtype='int32')
            label = EMOTIONS[preds.argmax()]

            # Assign labeling
            cv2.putText(frame, label, (fX, fY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

            cv.imshow('Otter', frame)
            if cv.waitKey(42) == ord('q'):
                break
# 작업 완료 후 해제
cap.release()
cv.destroyAllWindows()