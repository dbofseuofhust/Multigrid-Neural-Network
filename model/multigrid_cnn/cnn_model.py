import torch
import torch.nn as nn

from ..utils import mg_conv3_3, mg_input, Downsampling2_1, Downsampling3_2, Downsampling3_3


class VGG(nn.Module):
    """
    Build VGG model with input parameters:
        args: list of values which represent layer name and in-out channels
        num_classes: numbers of output class
    """

    def __init__(self, args, num_classes):
        super(VGG, self).__init__()

        layers = []
        in_channels = 3
        for para in args:
            if para == 'M':
                layers += [nn.MaxPool2d(2, 2)]
            else:
                layers += [nn.Conv2d(in_channels, para, 3, padding=1), nn.BatchNorm2d(para), nn.ReLU(inplace=True)]
                in_channels = para

        self.extract_feature = nn.Sequential(*layers)mg_conv3_3
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class VGG_mg(nn.Module):
    """
    Build multigrid VGG model with input parameters:
        args: list of values which represent layer name and in-out channels
        num_classes: numbers of output class
    """

    def __init__(self, num_classes):
        super(VGG_mg, self).__init__()

        self.inputs = mg_input(3, 64, 3, padding=1)
        self.conv1_2 = mg_conv3_3(64, 64, 3, padding=1)
        self.downsample1 = Downsampling3_3(kernel_size=(2, 2), stride=(2, 2))

        self.conv2_1 = mg_conv3_3(64, 128, 3, padding=1)
        self.conv2_2 = mg_conv3_3(128, 128, 3, padding=1)
        self.downsample2 = Downsampling3_3(kernel_size=(2, 2), stride=(2, 2))

        self.conv3_1 = mg_conv3_3(128, 256, 3, padding=1)
        self.conv3_2 = mg_conv3_3(256, 256, 3, padding=1)
        self.conv3_3 = mg_conv3_3(256, 256, 3, padding=1)
        self.downsample3 = Downsampling3_3(kernel_size=(2, 2), stride=(2, 2))

        self.conv4_1 = mg_conv3_3(256, 512, 3, padding=1)
        self.conv4_2 = mg_conv3_3(512, 512, 3, padding=1)
        self.conv4_3 = mg_conv3_3(512, 512, 3, padding=1)
        self.downsample4 = Downsampling3_2(kernel_size=(2, 2), stride=(2, 2))

        self.conv5_1 = mg_conv2_2(512, 512, 3, padding=1)
        self.conv5_2 = mg_conv2_2(512, 512, 3, padding=1)
        self.conv5_3 = mg_conv2_2(512, 512, 3, padding=1)
        self.downsample5 = Downsampling2_1(kernel_size=(2, 2), stride=(2, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(896 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1, x2, x3 = self.inputs(x)
        x1, x2, x3 = self.conv1_2(x1, x2, x3)
        x1, x2, x3 = self.downsample1(x1, x2, x3)
        x1, x2, x3 = self.conv2_1(x1, x2, x3)
        x1, x2, x3 = self.conv2_2(x1, x2, x3)
        x1, x2, x3 = self.downsample2(x1, x2, x3)
        x1, x2, x3 = self.conv3_1(x1, x2, x3)
        x1, x2, x3 = self.conv3_2(x1, x2, x3)
        x1, x2, x3 = self.conv3_3(x1, x2, x3)
        x1, x2, x3 = self.downsample3(x1, x2, x3)
        x1, x2, x3 = self.conv4_1(x1, x2, x3)
        x1, x2, x3 = self.conv4_2(x1, x2, x3)
        x1, x2, x3 = self.conv4_3(x1, x2, x3)
        x1, x2 = self.downsample4(x1, x2, x3)
        x1, x2 = self.conv5_1(x1, x2)
        x1, x2 = self.conv5_2(x1, x2)
        x1, x2 = self.conv5_3(x1, x2)
        x = self.downsample5(x1, x2)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
