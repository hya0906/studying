import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
import cv2 as cv
from torchsummary import summary

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        #self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)

        self.localization = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3), #8
            nn.MaxPool2d(2, stride=4),
            nn.ReLU(True),
            # block 2
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=3),#3
            nn.MaxPool2d(2, stride=4),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(90, 32), #640
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        #print(x.size())
        xs = self.localization(x)
        #print("xs",xs.size())
        xs = xs.view(-1, 90) #640
        #print("xs",xs.size())
        theta = self.fc_loc(xs)
        #print(theta)
        theta = theta.view(-1, 2, 3)
        #print(theta.size())
        #print(x.size())
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):

        #print("1",input.size())
        out = self.stn(input)

        # block 1
        #print("2", out.size())
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))
        #print("3", out.size())

        #block 2
        out = F.relu(self.conv3(out))
        #out = self.norm(self.conv4(out))
        out = self.conv4(out)
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


def test():
     img = Image.open("C:\\Users\\711_2\\Desktop\\Yuna_Hong\\Deep-Emotion_code\\data\\train\\train0.jpg")
     #print(img.size) #gray scale
     model = Deep_Emotion()
     summary(model, input_size = (1,48,48), device='cpu')

#test()