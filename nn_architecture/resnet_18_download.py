import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):     #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1): #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class StraightBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(StraightBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class StraightResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(StraightResNet18, self).__init__()
        self.inplanes = 64

        # Initial convolution and maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.conv2 = StraightBasicBlock(64, 64)
        self.conv3 = StraightBasicBlock(64, 64)

        # Layer 2
        self.downsample2 = nn.Sequential(
            conv1x1(64, 128, stride=2),
            nn.BatchNorm2d(128)
        )
        self.conv4 = StraightBasicBlock(64, 128, stride=2, downsample=self.downsample2)
        self.conv5 = StraightBasicBlock(128, 128)

        # Layer 3
        self.downsample3 = nn.Sequential(
            conv1x1(128, 256, stride=2),
            nn.BatchNorm2d(256)
        )
        self.conv6 = StraightBasicBlock(128, 256, stride=2, downsample=self.downsample3)
        self.conv7 = StraightBasicBlock(256, 256)

        # Layer 4
        self.downsample4 = nn.Sequential(
            conv1x1(256, 512, stride=2),
            nn.BatchNorm2d(512)
        )
        self.conv8 = StraightBasicBlock(256, 512, stride=2, downsample=self.downsample4)
        self.conv9 = StraightBasicBlock(512, 512)

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * StraightBasicBlock.expansion, num_classes)

    def forward(self, x):
        # Initial block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # Layer 1
        out = self.conv2(out)
        out = self.conv3(out)

        # Layer 2
        out = self.conv4(out)
        out = self.conv5(out)

        # Layer 3
        out = self.conv6(out)
        out = self.conv7(out)

        # Layer 4
        out = self.conv8(out)
        out = self.conv9(out)

        # Pooling and classification
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def straight_resnet18(num_classes=1000):
    return StraightResNet18(num_classes=num_classes)
