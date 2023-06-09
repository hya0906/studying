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
from pytorch_cifar_master.utils import progress_bar
from sklearn.metrics import confusion_matrix
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
#https://justkode.kr/deep-learning/pytorch-save/

if __name__ == "__main__":
    epoch = 10
    learning_rate= 0.001
    classes = ('surprised', 'fearful', 'disgusted', 'happy', 'sad', 'angry', 'neutral')

    net = ResNet50().cuda()
    #net = nn.DataParallel(net).cuda()
    #net = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    net.load_state_dict(torch.load(f'C:\\Users\\711_2\\Desktop\\Yuna_Hong\\23년1학기_수업\\가상현실시스템_금\\Facial_expression\\test_81.4863_191.pth')['net'])
    #net.load_state_dict(torch.load(f'./weight/epoch-{epoch}_model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장
    net.eval()

    transform_test = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    testset = torchvision.datasets.ImageFolder(root="C:/Users/711_2/Desktop/Yuna_Hong/facial_expression/aligned/test",
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

    y_pred = []
    y_true = []
    plt.title(f"emotion-confusion matrix_112_Adam_{epoch}_{learning_rate}")
    # iterate over test data
    for inputs, labels in testloader:
        inputs = inputs.cuda()
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth
    print(y_pred, y_true)
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print("cf_matrix", cf_matrix)
    print("result", cf_matrix / np.sum(cf_matrix, axis=1)[:, None])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f"./new1.png")
    #plt.savefig(f"./emotion-confusion matrix_112_Adam_{epoch + 1}_{learning_rate}.png")
