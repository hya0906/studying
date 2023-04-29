'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from PIL import Image

c=0
class BasicBlock(nn.Module): #conv2개, shortcut1개 구현
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        #print(in_planes, self.expansion,planes)
        #???????????????????
        if stride != 1 or in_planes != self.expansion*planes: #stride가 1이 아니거나 들어오는 차원과 planes가 다를때
            global c
            #print("???????",c)
            c+=1
            #print(in_planes, self.expansion*planes)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module): #50layer이상일때
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7): #감정데이터 7클래스
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #print("$$$$$",block.expansion)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes) #block.expansion=4

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        #print("!",strides,[stride],[1]*(num_blocks-1),num_blocks)
        layers = []
        for stride in strides:
            #print("@@@@@",stride)
            layers.append(block(self.in_planes, planes, stride)) #self.in_planes-size of filters
            self.in_planes = planes * block.expansion
            #print("!!",self.in_planes, layers)
        # print("!!!",layers,*layers)
        return nn.Sequential(*layers) #리스트에 있는 것을 빼서

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print("1",out.size())
        out = F.avg_pool2d(out, 14) #kernel_size = 4
        #print("2",out.size())
        #print(out.size(0))
        out = out.view(out.size(0), -1)
        #print("3",out.size())
        out = self.linear(out)
        #print("4",out.size())
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3]) #num_blocks-64세트3개, 128세트4개,256세트6개,512세트3개


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3]) #50layer이상은 Bottleneck사용


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    img = Image.open('C:/Users/711_2/Desktop/Yuna_Hong/facial_expression/aligned/test/1/test_0002_aligned.jpg')
    print(img.size)
    net = ResNet50()
    summary(net, input_size = (3,112,112), device='cpu')
    #print(net)
    #y = net(torch.randn(1, 3, 32, 32))
    #print(y.size())

#test()
