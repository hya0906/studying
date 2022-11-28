import insightface
import cv2
from insightface.app import FaceAnalysis
import onnxruntime
import numpy as np
from multiprocessing import Process
import matplotlib.pyplot as plt
import os
from numba import jit
from numba import int32, float32    # import the types
from numba.experimental import jitclass

#0jessi 1IU 2yuna_hong 3gongyu 4dongyuk 5barack
#'ariana', 'barack', 'brendan', 'Chanhyuk', 'christopher', 'dongyuk', 'gongyu', 'IU', 'jessi', 'justin', 'suhyun', 'yuna_hong'
names = {0: 'ariana', 1: 'barack', 2: 'brendan', 3: 'Chanhyuk', 4: 'christopher', 5: 'dongyuk', 6: 'gongyu', 7: 'IU', 8: 'jessi', 9: 'justin', 10: 'suhyun', 11: 'yuna_hong'}
#buffalo_s는 인식안됨


model_name = 'buffalo_m'
app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition'])  # enable detection model only app.prepare(ctx_id=0, det_size=(640, 640))  # 옆,기울어진얼굴 다 인식가능 detector = insightface.model_zoo.get_model('D:\insightface_folder\lab_test\data\det_10g.onnx')  # retinaface
app.prepare(ctx_id=0, det_size=(640, 640))  # 옆,기울어진얼굴 다 인식가능

detector = insightface.model_zoo.get_model('D:\insightface_folder\lab_test\data\det_10g.onnx')  # retinaface
detector.prepare(ctx_id=0, det_size=(640, 640))

ort_model = onnxruntime.InferenceSession('D:\insightface_folder\lab_test\data\model_12.onnx')
input = ort_model.get_inputs()[0].name
output = ort_model.get_outputs()[0].name
flag = False
name = ''
bboxes, landmarks = None, None


def who_are_you(pic):  # 일단 한명예측
    name = "Unknown"
    res = ort_model.run([output], {input: pic})
    if res[0][0][np.argmax(res)] > 0.95:
        name = names[np.argmax(res)]
        flag = True
    else:
        flag = False
    return name, flag

#여러 명 중에 인식된 얼굴 크기가 큰사람을 기준으로 잡을예정
def recognize_faces(frame):
    global bboxes, name
    bboxes, landmarks = detector.detect(frame, (512, 512), 5, 2)  # img, input_size, max_num, metric #얼굴크면 img크기키워야함
    for i, box in enumerate(bboxes):  # 예측확률이 일정이상 넘어가지 않으면 unknown으로★
        # 좌표 추출
        x, y, w, h, _ = map(int, box)
        try:
            face = app.get(frame)[0]  # 얼굴이 사진의 일정비율을 차지하면 원본사진을 그대로 쓰는것추가★
            pic = np.array([face.embedding], dtype=np.float32)
            name, flag = who_are_you(pic)
        except:
            name = "Unknown"
            flag = False

        # 오류구문추가/몇번이상 사람이 인식이 안될경우 크기조정기능필요★
        # rectangle그림따로 얼굴처리용이미지따로★
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), thickness=2)
        cv2.putText(frame, name, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)  # 이름으로 바꾸기★
        #print(name)
        return flag

def draw_faces(frame):
    x, y, w, h, _ = map(int, bboxes[0])
    cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), thickness=2)
    cv2.putText(frame, name, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)