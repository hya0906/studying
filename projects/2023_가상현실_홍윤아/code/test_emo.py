'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from collections import Counter
import os
import argparse
import numpy as np

import torch
from tqdm import tqdm
import cv2
from PIL import Image

def load_model_and_data(device):
    net = torchvision.models.resnet50(pretrained=False, num_classes=7)
    net = net.to(device)
    weight = torch.load(f'..\\test_epoch21_84.5176_191.pth', map_location=device)
    net.load_state_dict(weight['net'], strict=False)  # state_dict를 불러 온 후, 모델에 저장
    net.eval()

    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    face_cascade = cv2.CascadeClassifier('..\\haarcascade_frontalface_alt.xml')
    return net, transform, face_cascade

def detect_face(frame, gray):
    global roi_color
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_color = cv2.resize(roi_color, (112, 112))
    return frame, roi_color


def test_img(roi_color):
    # convert from openCV2 to PIL
    color_coverted = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)

    input = transform(pil_image)
    input = input.unsqueeze(0)  # 차원추가
    input = input.to(device)  # test
    output = net.forward(input)

    probs = torch.nn.functional.softmax(output, dim=1)
    conf, class_ = torch.max(probs, 1)  # 정확도, 클래스 인덱스
    return conf, class_


if __name__ == "__main__":
    classes = {0:'surprised', 1:'fearful', 2:'disgusted', 3:'happy', 4:'sad', 5:'angry', 6:'neutral'}
    emotion = []
    device = torch.device('cpu')
    net, transform, face_cascade = load_model_and_data(device)

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()

        while ret:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #face detect
            frame, roi_color = detect_face(frame, gray)
            #emotion predict
            conf, class_ = test_img(roi_color) # 정확도, 클래스 인덱스

            conf = conf.item()
            emotion.append(class_[0].item())
            #print(class_)
            if len(emotion) == 6: #1/5프레임당 가장 많이 나온 값 선정-정확도를 위해(변경가능)
                count = Counter(emotion)
                c = count.most_common(n=1)
                cls = classes[c[0][0]]#클래스 이름만
                print(cls)
                emotion.clear()
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
