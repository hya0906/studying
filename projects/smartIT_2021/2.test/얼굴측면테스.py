#1.얼굴 측면 인식 성공
#2.측면을 affine변환으로 얼굴표정감정
import numpy as np
import cv2

rightxml = 'haarcascade_righteye_2splits.xml'
#leftxml = 'haarcascade_lefteye_2splits.xml'
leftxml = 'haarcascade_mcs_nose.xml'
right_cascade = cv2.CascadeClassifier(rightxml)
left_cascade = cv2.CascadeClassifier(leftxml)

cap = cv2.VideoCapture(0)  # 노트북 웹캠을 카메라로 사용
cap.set(3, 640)  # 너비
cap.set(4, 480)  # 높이

while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 좌우 대칭
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    right = right_cascade.detectMultiScale(gray, 1.05, 5)
    left = left_cascade.detectMultiScale(gray, 1.05, 5)
    #print(faces)
    #print("Number of faces detected: " + str(len(faces)))

    if len(left):
        for (x, y, w, h) in left:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if len(right):
        for (x, y, w, h) in right:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('result', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Esc 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()