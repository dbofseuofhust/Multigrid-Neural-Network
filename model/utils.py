import torch
import torch.nn as nn


def extract_feature(args, batchnorm=False):
    """
    function: build module extract feature for your neural network model( aditional)
    input parameter:
        -args: list alias of layer
            + number: output of Conv2d layer
            + "M": Maxpooling
            example [64, 64, 'M', 128] -> Conv2d(in_channels, 64, 3)-Conv2d(64, 64, 3)-Maxpooling(2)-Conv2d(64, 128, 3)
        -batchnorm: batch normalization after each Conv2d if true
    """

    layers = []
    in_channels = 3
    for para in args:
        if para == 'M':
            layers += [nn.MaxPool2d(2, 2)]
        else:
            if batchnorm:
                layers += [nn.Conv2d(in_channels, para, 3, padding=1), nn.BatchNorm2d(para), nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, para, 3, padding=1), nn.ReLU(inplace=True)]
        in_channels = para

    return nn.Sequential(*layers)


class mg_input(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(mg_input, self).__init__()
        self.scale = 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels//self.scale, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels//(self.scale**2), kernel_size, stride, padding)
        self.downsample2 = nn.MaxPool2d(kernel_size=self.scale, stride=self.scale)
        self.downsample3 = nn.MaxPool2d(kernel_size=self.scale**2, stride=self.scale**2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x2 = self.downsample2(x2)
        x3 = self.downsample3(x3)
        return x1, x2, x3


class Downsampling3_3(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Downsampling3_3, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x1, x2, x3):
        x1 = self.downsample(x1)
        x2 = self.downsample(x2)
        x3 = self.downsample(x3)
        return x1, x2, x3


class Downsampling3_2(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Downsampling3_2, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x1, x2, x3):
        x1 = self.downsample(x1)
        x2 = self.downsample(x2)
        x2 = torch.cat((x2, x3), dim=1)
        return x1, x2


class Downsampling2_1(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Downsampling2_1, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x1, x2):
        x1 = self.downsample(x1)
        x1 = torch.cat((x1, x2), dim=1)
        return x1


class mg_conv3_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(mg_conv3_3, self).__init__()
        self.scale = 2
        self.downsample = nn.MaxPool2d(kernel_size=self.scale, stride=self.scale)
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels//2*3, out_channels, kernel_size, stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels//4*7, out_channels//2, kernel_size, stride, padding=1)
        self.conv3 = nn.Conv2d(in_channels//4*3, out_channels//4, kernel_size, stride, padding=1)

    def forward(self, x1, x2, x3):
        x1_down = self.downsample(x1)
        x2_down = self.downsample(x2)
        x2_up = self.upsample(x2)
        x3_up = self.upsample(x3)

        x1 = torch.cat((x1, x2_up), dim=1)
        x2 = torch.cat((x1_down, x2, x3_up), dim=1)
        x3 = torch.cat((x2_down, x3), dim=1)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        return x1, x2, x3


class mg_conv2_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(mg_conv2_2, self).__init__()
        self.scale=2
        self.conv1 = nn.Conv2d(in_channels//4*7, out_channels, kernel_size, stride, padding=1)
        self.conv2 = nn.Conv2d(in_channels//4*7, out_channels//4*3, kernel_size, stride, padding=1)

        self.downsample = nn.MaxPool2d(kernel_size=self.scale, stride=self.scale)
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='nearest')

    def forward(self, x1, x2):
        x1_down = self.downsample(x1)
        x2_up = self.upsample(x2)

        x1 = torch.cat((x1, x2_up), dim=1)
        x2 = torch.cat((x1_down, x2), dim=1)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        return x1, x2


class res_mg_unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(res_mg_unit, self).__init__():
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.mg_conv1 = mg_conv3_3(in_channels, out_channels)
        self.mg_conv2 = mg_conv3_3(out_channels, out_channels)

    def forward(self, x1, x2, x3):
        x1_temp, x2_temp, x3_temp = x1, x2, x3

        x1, x2, x3 = self.mg_conv1((x1, x2, x3))

        x1 = self.batchnorm(x1)
        x2 = self.batchnorm(x2)
        x3 = self.batchnorm(x3)
        x1 = self.activation(x1)
        x2 = self.activation(x2)
        x3 = self.activation(x3)

        x1, x2, x3 = self.mg_conv2((x1, x2, x3))

        x1 = self.batchnorm(x1)
        x2 = self.batchnorm(x2)
        x3 = self.batchnorm(x3)

        x1 += x1_temp
        x2 += x2_temp
        x3 += x3_temp

        x1 = self.activation(x1)
        x2 = self.activation(x2)
        x3 = self.activation(x3)

        return x1, x2, x3
