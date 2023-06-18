from torchsummary import summary
import torch.nn.functional as F
import torch
import torch.nn as nn
#this code is not working(original_x.shape[-1] != output_2.shape[-1] they are not the same)
#this is just for backup
#####################
features = 60
filter_size = 39
dropout_rate = 0.1
#####################

# 입력 데이터 생성
# (1440, 215, 60)
input_size = (64, 215, 60)
input_data = torch.randn(input_size)
print("data before: ", input_data.shape)
input_data = input_data.permute(0, 2, 1)  # batch/mfcc_coefficient/length
print("data after: ", input_data.shape)


def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    pad = (kernel_size - 1) * dilation
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=filter_size, out_channels=filter_size, kernel_size=2, padding=(2 - 1) * 1,
                                  dilation=1)
        self.batchnorm_1 = nn.BatchNorm1d(filter_size)
        self.activation_1 = nn.ReLU()
        self.dropout2d_1 = nn.Dropout2d(dropout_rate)

        self.conv1d_2 = nn.Conv1d(in_channels=filter_size,out_channels=filter_size,kernel_size=2,padding=(2 - 1) * 1, dilation=1)
        self.batchnorm_2 = nn.BatchNorm1d(filter_size)
        self.activation_2 = nn.ReLU()
        self.dropout2d_2 = nn.Dropout2d(dropout_rate)

        self.original_x = nn.Conv1d(in_channels=filter_size, out_channels=filter_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        original_x = x
        # # 1
        # print("2",x.shape)
        x = self.conv1d_1(x)
        # print("3",x.shape)
        x = self.batchnorm_1(x)
        # print("4",x.shape)
        x = self.activation_1(x)
        # print("5",x.shape)
        output_2 = self.dropout2d_1(x)
        # print("6",output_2.shape)

        # # 2
        # print("2",x.shape)
        x = self.conv1d_2(x)
        # print("3",x.shape)
        x = self.batchnorm_2(x)
        # print("4",x.shape)
        x = self.activation_2(x)
        # print("5",x.shape)
        output_2 = self.dropout2d_2(x)
        # print("6",output_2.shape)

        print(original_x.shape[-1], output_2.shape[-1])
        if original_x.shape[-1] != output_2.shape[-1]:
            x = self.original_x(output_2)
        print("7", x.shape)
        output_2_1 = self.sigmoid(output_2)
        F_x = torch.mul(original_x, output_2_1)
        return F_x


model = Model()
print(model(input_data))