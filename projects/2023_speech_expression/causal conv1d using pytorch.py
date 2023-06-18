#causal conv1d 구현
#https://stopspoon.tistory.com/48
import torch.nn as nn
import torch

# 입력 데이터 생성
#(1440, 215, 60)
input_size = (64, 215, 60)
input_data = torch.randn(input_size)
print("data before: ",input_data.shape)
input_data = input_data.permute(0, 2, 1) # batch/mfcc_coefficient/length
print("data after: ",input_data.shape)

def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    pad=(kernel_size - 1)* dilation
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CausalConv1d(60,60,kernel_size=3, dilation=2)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = x[:,:,:-self.conv1.padding[0]]
        print(x.shape)
        return x

model = Network()
print(model(input_data).shape)