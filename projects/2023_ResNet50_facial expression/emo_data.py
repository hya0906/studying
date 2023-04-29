import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import shutil
import PIL
import torchvision.transforms as transforms

img = PIL.Image.open('./aligned/train/1/train_00006_aligned.jpg')
#img.show()

tf = transforms.ToTensor()
img_t = tf(img)

# torch.Size([3, 1280, 1920])
print(img_t.size())

'''
path = './aligned/test/'

os.makedirs(path,exist_ok = True)
for i in range(1,8):
    os.makedirs(path+ str(i), exist_ok=True)

src = 'C:/Users/711_2/Desktop/Yuna_Hong/facial_expression/aligned/'
dest = 'C:/Users/711_2/Desktop/Yuna_Hong/facial_expression/aligned/test/'
with open("list_patition_label.txt", "r") as f:
    while True:
        line = f.readline()
        if not line: # 파일 읽기가 종료된 경우
            break
        data = line.split(" ")[0].split(".")[0]+"_aligned.jpg"
        label = line.split(" ")[1].strip("\n")
        print(data, label)
        print(data)
        if "test" in data:
            print(src+data,"\n", dest+str(label))
            shutil.move(src+data, dest+str(label))

'''
'''
class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)

        # 들어온 데이터의 형식 바꾸기

        self.x_data = self.x_data.permute(0, 3, 1, 2)  # 이미지 개수, 채널 수, 이미지 너비, 높이

        self.y_data = torch.LongTensor(y_data)

        self.len = self.y_data.shape[0]

    # x,y를 튜플형태로 바깥으로 내보내기

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

train_data = TensorData(train_images, train_labels)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
'''