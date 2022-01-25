import numpy as np
import cv2
import math
from PIL import Image
# 찾고자하는 것의 cascade classifier 를 등록
# 경로는 상대경로로 바뀔 수 있음
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:/Users/USER/anaconda3/pkgs/libopencv-3.4.2-h20b85fd_0/Library/etc/haarcascades/haarcascade_eye.xml')

""" 
    def = haar를 이용 얼굴과 눈을 찾는 함수
    input = 그레이 스케일 이미지
    output = 얼굴과 눈에 사각형이 그려진 이미지 프레임
"""

def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def detect(gray, frame):
    # 등록한 Cascade classifier 를 이용 얼굴을 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    global eye_1, eye_2, face_color
    # 얼굴에 사각형을 그리고 눈을 찾자
    for (x, y, w, h) in faces:
        # 얼굴: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 255 0 0 , 굵기 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 이미지를 얼굴 크기 만큼 잘라서 그레이스케일 이미지와 컬러이미지를 만듬
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # 등록한 Cascade classifier 를 이용 눈을 찾음(얼굴 영역에서만)
        eyes = eyeCascade.detectMultiScale(face_gray, 1.1, 3)

        # 눈: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 0 255 0 , 굵기 2)
        for i,(eye_x, eye_y, eye_w, eye_h) in enumerate(eyes):
            cv2.rectangle(face_color, (eye_x, eye_y),(eye_x+eye_w, eye_y+eye_h), (0, 255, 0), 2)
            #cv2.circle(face_color, (eye_x, eye_y), 2, (0, 0, 255), 2)
            #cv2.circle(face_color, (eye_x+eye_w, eye_y+eye_h), 2, (0, 0, 255), 2)
            #cv2.circle(face_color, (eye_x + int(eye_w/2), eye_y + int(eye_h/2)), 2, (0, 0, 255), 2)
            if i == 0:
                eye_1 = (eye_x, eye_y, eye_w, eye_h)
            elif i == 1:
                eye_2 = (eye_x, eye_y, eye_w, eye_h)

    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1

    print(left_eye,right_eye)
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

    return  new_img #frame


# 웹캠에서 이미지 가져오기
video_capture = cv2.VideoCapture(0)

eye_1, eye_2,face_color = 0, 0, 0
while True:
    # 웹캠 이미지를 프레임으로 자름
    _, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    frame_raw = frame.copy()
    # 그리고 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 만들어준 얼굴 눈 찾기
    canvas = detect(gray, frame)

    faces = faceCascade.detectMultiScale(canvas, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # 얼굴: 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+길이)까지의 사각형을 그림(색 255 0 0 , 굵기 2)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # 찾은 이미지 보여주기
    cv2.imshow("haha", canvas)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 끝
video_capture.release()
cv2.destroyAllWindows()
