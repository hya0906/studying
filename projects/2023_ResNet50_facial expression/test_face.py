'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from emo_resnet import *
import seaborn as sn
import pandas as pd
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import glob
import shutil
from matplotlib import pyplot as plt
import time
import datetime
import gc
import torch
from tqdm import tqdm
import cv2
from PIL import Image
#https://justkode.kr/deep-learning/pytorch-save/
#http://machinelearningkorea.com/2020/01/24/predict-single-image-with-pytorch-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%8B%B1%EA%B8%80-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%98%88%EC%B8%A1/

if __name__ == "__main__":
    epoch = 60
    learning_rate= 0.001
    classes = ('surprised', 'fearful', 'disgusted', 'happy', 'sad', 'angry', 'neutral')
    
    #모델 불러오기
    device = torch.device('cpu')
    net = ResNet50().to(device)
    #net = nn.DataParallel(net).to(device)
    #net = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    weight=torch.load(f'./test_81.4863_191.pth', map_location=device)
    net.load_state_dict(weight["net"])  # state_dict를 불러 온 후, 모델에 저장
    net.eval()

    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # load your image(s)
    #img = Image.open('happy1.jpg')
    #gray = Image.open("happy1.jpg").convert("L")
    #img.show()
    #gray.show()

    img = cv2.imread('./sad.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image",img)
    cv2.imshow("gray",gray)
    
    #face detect
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("image",roi_color)
    
    #PIL로 변환
    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert from openCV2 to PIL
    pil_image = Image.fromarray(color_coverted)
    #pil_image.show()
    
    #data transform
    tensor = transform(pil_image)
    tensor = tensor.unsqueeze(0) # input 데이터 차원 추가
    tensor = tensor.to(device)
    output = net.forward(tensor) # predict

    probs = torch.nn.functional.softmax(output, dim=1)
    print(probs)
    conf, classes = torch.max(probs, 1)
    print(conf, classes)
    conf = conf.item()
    cls = classes[classes.item()]
    print(cls, ' at confidence score:{0:.2f}'.format(conf))

