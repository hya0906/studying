import cv2
import numpy as np
#import dlib
import shutil
import math
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
#회전했을때도 인식할 수 있도록 함

# Face detection XML load and trained model loading
face_detection1 = cv2.CascadeClassifier('C:/Users/USER/Desktop/files/haarcascade_frontalface_default.xml')
path = 'C:/Users/USER/anaconda3/pkgs/libopencv-3.4.2-h20b85fd_0/Library/etc/haarcascades/haarcascade_profileface.xml'
face_detection2 = cv2.CascadeClassifier(path)
eyeCascade = cv2.CascadeClassifier('C:/Users/USER/anaconda3/pkgs/libopencv-3.4.2-h20b85fd_0/Library/etc/haarcascades/haarcascade_eye.xml')

lefteyeCascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
#noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
emotion_classifier = load_model('C:/Users/USER/Desktop/files/emotion_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surpring", "Neutral"]


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detect(gray, frame):
    # 등록한 Cascade classifier 를 이용 얼굴을 찾음
    faces = face_detection1.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    global eye_1, eye_2, face_color, new_img, left_eye, right_eye,a,b,c
    # 얼굴에 사각형을 그리고 눈을 찾자
    for (x, y, w, h) in faces:
        # 얼굴: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 255 0 0 , 굵기 2)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 이미지를 얼굴 크기 만큼 잘라서 그레이스케일 이미지와 컬러이미지를 만듬
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # 등록한 Cascade classifier 를 이용 눈을 찾음(얼굴 영역에서만)
        eyes = eyeCascade.detectMultiScale(face_gray, 1.1, 3)
        #lefteye = lefteyeCascade.detectMultiScale(face_gray, 1.1, 3)

        # 눈: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 0 255 0 , 굵기 2)
        for i,(eye_x, eye_y, eye_w, eye_h) in enumerate(eyes):
            cv2.rectangle(face_color, (eye_x, eye_y),(eye_x+eye_w, eye_y+eye_h), (0, 255, 0), 2)
            cv2.circle(face_color, (eye_x, eye_y), 2, (0, 0, 255), 2)
            cv2.circle(face_color, (eye_x+eye_w, eye_y+eye_h), 2, (0, 0, 255), 2)
            cv2.circle(face_color, (eye_x + int(eye_w/2), eye_y + int(eye_h/2)), 2, (0, 0, 255), 2)
            if i == 0:
                eye_1 = (eye_x, eye_y, eye_w, eye_h)
            elif i == 1:
                eye_2 = (eye_x, eye_y, eye_w, eye_h)
    try:
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1
    except:
        pass

    #print(left_eye,right_eye)
    left_eye_center = (left_eye[0] + int(left_eye[2]/2), left_eye[1] + int(left_eye[3]/2))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]
    right_eye_center = (right_eye[0] + int(right_eye[2]/2), right_eye[1] + int(right_eye[3]/2))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]
    print("center",left_eye_center,right_eye_center)
    cv2.circle(face_color, left_eye_center, 2, (0, 0, 255), 2)
    cv2.circle(face_color, right_eye_center, 2, (0, 0, 255), 2)
    cv2.line(face_color, right_eye_center, left_eye_center, (67, 67, 67), 2)
    try:
        if left_eye_y < right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
            print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock
            print("rotate to inverse clock direction")
        cv2.circle(face_color, point_3rd, 2, (255, 0, 0), 2)
        cv2.line(face_color, right_eye_center, left_eye_center, (67, 67, 67), 2)
        cv2.line(face_color, left_eye_center, point_3rd, (67, 67, 67), 2)
        cv2.line(face_color, right_eye_center, point_3rd, (67, 67, 67), 2)

        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, left_eye_center)
        c = euclidean_distance(right_eye_center, point_3rd)
        print("length:", a**2+c**2)
    except:
        pass
    try:
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        print("cos(a) = ", cos_a)

        angle = np.arccos(cos_a)
        print("angle: ", angle, " in radian")

        angle = (angle * 180) / math.pi
        print("angle: ", angle, " in degree")

        if direction == -1:
            angle = 90 - angle

        new_img = Image.fromarray(frame_raw)
        new_img = np.array(new_img.rotate(direction * -angle))

    except ZeroDivisionError as e:
        pass

    return new_img #frame


# Video capture using webcam
camera = cv2.VideoCapture(0)
eye_1, eye_2,face_color, new_img = 0, 0, 0, 0
while True:
    # Capture image from camera
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1) #flip images
    frame_raw = frame.copy()
    # Convert color to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rotated = detect(gray, frame)
    cv2.imshow("R",rotated)

    faces = face_detection1.detectMultiScale(rotated, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
    #global eye_1, eye_2, face_color, new_img
    # 얼굴에 사각형을 그리고 눈을 찾자
    for (x, y, w, h) in faces:
        # 얼굴: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 255 0 0 , 굵기 2)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 이미지를 얼굴 크기 만큼 잘라서 그레이스케일 이미지와 컬러이미지를 만듬
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # 등록한 Cascade classifier 를 이용 눈을 찾음(얼굴 영역에서만)
        # 눈: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 0 255 0 , 굵기 2)
        nose = eyeCascade.detectMultiScale(face_gray, 1.1, 3)

        for i, (nose_x, nose_y, nose_w, nose_h) in enumerate(nose):
            cv2.rectangle(face_color, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (0, 255, 0), 2)
            nose = (nose_x, nose_y, nose_w, nose_h)
        print("nose", nose)

        # eyes = eyeCascade.detectMultiScale(face_gray, 1.1, 3)
        lefteye = lefteyeCascade.detectMultiScale(face_gray, 1.1, 3)

        # 눈: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 0 255 0 , 굵기 2)
        for i, (eye_x, eye_y, eye_w, eye_h) in enumerate(lefteye):
            cv2.rectangle(face_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 2)
            if eye_x > nose[0]:
                lefteye = (eye_x, eye_y, eye_w, eye_h)
            if i==len(lefteye)-1 and lefteye is not tuple:
                lefteye = (eye_x, eye_y, eye_w, eye_h)

        print("lefteye",lefteye)

        left_eye_center = (lefteye[0] + int(lefteye[2] / 2), lefteye[1] + int(lefteye[3] / 2))
        noseCenter = (nose[0] + int(nose[2] / 2), nose[1] + int(nose[3] / 2))
        cv2.circle(face_color, left_eye_center, 2, (0, 0, 255), 2)
        cv2.circle(face_color, noseCenter, 2, (0, 0, 255), 2)
        cv2.line(face_color, left_eye_center, noseCenter, (67, 67, 67), 2)
    cv2.imshow('Emotion Recognition', frame)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clear program and close windows
camera.release()
cv2.destroyAllWindows()