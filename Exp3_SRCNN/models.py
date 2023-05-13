import torch
from math import ceil
from torch import nn


def check_conv(conv_model, X):
    ch_in = conv_model.in_channels
    ch_out = conv_model.out_channels
    ker = conv_model.kernel_size[0]
    stride = conv_model.stride[0]
    batch, _ , height, width = X.shape

    # nn.conv2d
    # conv_model = nn.Conv2d(ch_in, ch_out, ker, stride=stride, padding=(ker - 1) // 2, bias=False)
    # y_conv = conv_model(X)

    # unfold conv2d
    unfold = nn.Unfold(ker, padding=(ker - 1) // 2, stride=stride)
    X_unfold = unfold(X).transpose(1, 2)                       # size = batch, (height/ stride) * (width/ stride), C_in * k^2
    W = conv_model.weight.reshape(ch_out, -1).transpose(0, 1)  # C_in * k^2, C_out
    y_unfold = torch.matmul(X_unfold, W).transpose(1, 2).reshape(batch, ch_out, ceil(height / stride), ceil(width / stride))

    return X_unfold, y_unfold

class SRCNN_pre(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()

        self.ch0, self.ch1, self.ch2 = 1, 64, 32
        self.k1, self.k2, self.k3 = 9, 5, 5

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.ch0, self.ch1, kernel_size=self.k1, padding=self.k1 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2, kernel_size=self.k2, padding=self.k2 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(self.ch2, self.ch0, kernel_size=self.k3, padding=self.k3 // 2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x_latent = self.conv2(x)
        x_pred = self.conv3(x_latent)

        # manually check convolution operation
        # x_latent_unfold, y_unfold = check_conv(self.conv3, x_latent)

        return x_latent, x_pred
    
class SRCNN_01(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN_01, self).__init__()

        self.ch0, self.ch1, self.ch2, self.ch3, self.ch4 = 1, 32, 32, 32, 32
        self.k1, self.k2, self.k3, self.k4, self.k5 = 9, 5, 5, 5, 5

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.ch0, self.ch1, kernel_size=self.k1, padding=self.k1 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2, kernel_size=self.k2, padding=self.k2 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch2)
        )
        self.conv3 = nn.Conv2d(self.ch4, self.ch0, kernel_size=self.k3, padding=self.k3 // 2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x_latent = self.conv2(x)
        x_pred = self.conv3(x_latent)

        return x_latent, x_pred

    
class SRCNN_02(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN_02, self).__init__()

        self.ch0, self.ch1, self.ch2, self.ch3, self.ch4 = 1, 32, 32, 32, 32
        self.k1, self.k2, self.k3, self.k4, self.k5 = 9, 5, 5, 5, 5

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.ch0, self.ch1, kernel_size=self.k1, padding=self.k1 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2, kernel_size=self.k2, padding=self.k2 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.ch2, self.ch3, kernel_size=self.k4, padding=self.k4 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.ch3, self.ch4, kernel_size=self.k5, padding=self.k5 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch4)
        )
        self.conv3 = nn.Conv2d(self.ch4, self.ch0, kernel_size=self.k3, padding=self.k3 // 2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv4(x)
        x_latent = self.conv5(x)
        x_pred = self.conv3(x_latent)

        # manually check convolution operation
        # x_latent_unfold, y_unfold = check_conv(self.conv3, x_latent)

        return x_latent, x_pred


class SRCNN_03(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN_03, self).__init__()

        self.ch0, self.ch1, self.ch2, self.ch3, self.ch4 = 1, 32, 32, 32, 32
        self.k1, self.k2, self.k3, self.k4, self.k5 = 9, 5, 5, 5, 5

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.ch0, self.ch1, kernel_size=self.k1, padding=self.k1 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ch1, self.ch2, kernel_size=self.k2, padding=self.k2 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.ch2, self.ch3, kernel_size=self.k4, padding=self.k4 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.ch3, self.ch4, kernel_size=self.k5, padding=self.k5 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch4)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(self.ch2, self.ch3, kernel_size=self.k4, padding=self.k4 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch3)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(self.ch3, self.ch4, kernel_size=self.k5, padding=self.k5 // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.ch4)
        )
        self.conv3 = nn.Conv2d(self.ch4, self.ch0, kernel_size=self.k3, padding=self.k3 // 2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x_latent = self.conv7(x)
        x_pred = self.conv3(x_latent)

        # manually check convolution operation
        # x_latent_unfold, y_unfold = check_conv(self.conv3, x_latent)

        return x_latent, x_pred
