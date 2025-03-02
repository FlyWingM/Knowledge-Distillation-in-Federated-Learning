import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class StraightResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(StraightResNet18, self).__init__()
        
        # Initial convolutional block
        self.conv1 = conv_block(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        # Layer 1
        self.conv2 = conv_block(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv_block(64, 64, kernel_size=3, stride=1, padding=1)
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # Layer 2
        self.conv4 = conv_block(64, 128, kernel_size=3, stride=2, padding=1)  # Downsample with stride=2
        self.conv5 = conv_block(128, 128, kernel_size=3, stride=1, padding=1)
        self.res2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Layer 3
        self.conv6 = conv_block(128, 256, kernel_size=3, stride=2, padding=1)  # Downsample with stride=2
        self.conv7 = conv_block(256, 256, kernel_size=3, stride=1, padding=1)
        self.res3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        
        # Layer 4
        self.conv8 = conv_block(256, 512, kernel_size=3, stride=2, padding=1)  # Downsample with stride=2
        self.conv9 = conv_block(512, 512, kernel_size=3, stride=1, padding=1)
        self.res4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        
        # Average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial block
        out = self.conv1(x)
        
        # Layer 1
        residual = self.res1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = F.relu(out)
        
        # Layer 2
        residual = self.res2(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out += residual
        out = F.relu(out)
        
        # Layer 3
        residual = self.res3(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out += residual
        out = F.relu(out)
        
        # Layer 4
        residual = self.res4(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out += residual
        out = F.relu(out)
        
        # Pooling and classification
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def straight_resnet18():
    return StraightResNet18(in_channels=3, num_classes=10)
