#동영상을 빠르게 재생함 video__의 완성형
import cv2
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

prev_time = 0
FPS = 17

# Face detection XML load and trained model loading
face_detection = cv2.CascadeClassifier('C:/Users/USER/Desktop/files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('C:/Users/USER/Desktop/files/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]

#video = cv2.VideoCapture(0)
cap = cv2.VideoCapture('short.mp4')

while True:
    ret, frame = cap.read(cv2.COLOR_BGR2GRAY)
    current_time = time.time() - prev_time

    if current_time > 1. / FPS:
        prev_time = time.time()

        # Convert color to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            roi = cv2.resize(roi, (48, 48))
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
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)


        cv2.imshow('VideoCapture', frame)

        if cv2.waitKey(1) > 0:
            break

# 작업 완료 후 해제
cap.release()
cv.destroyAllWindows()