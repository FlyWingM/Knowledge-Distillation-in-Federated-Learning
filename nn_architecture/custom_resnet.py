import torch.nn as nn
import torch
import torch.nn.functional as F


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


#CustomResNet18--------------------------------------
class CustomResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Initial block
        self.conv1 = conv_block(in_channels, 64, pool=True)  # Added pooling here
        
        # Layer 1
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        
        # Layer 2
        self.conv4 = conv_block(64, 128, pool=True)  # Adjusted pooling to this layer
        self.conv5 = conv_block(128, 128)
        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        # Layer 3
        self.conv6 = conv_block(128, 256, pool=True)  # Adjusted pooling to this layer
        self.conv7 = conv_block(256, 256)
        self.res3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        # Layer 4
        self.conv8 = conv_block(256, 512, pool=True)  # Adjusted pooling to this layer
        self.conv9 = conv_block(512, 512)
        self.res4 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        # Classifier
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1),  # Changed to AdaptiveMaxPool
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.res1(out) + out
        
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.res2(out) + out
        
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.res3(out) + out
        
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.res4(out) + out
        
        out = self.classifier(out)
        return out


#CustomResNet18_v1--------------------------------------
class CustomResNet18_v1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Initial block
        self.conv1 = conv_block(in_channels, 64)
        
        # Layer 1
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        
        # Layer 2
        self.conv4 = conv_block(64, 128)
        self.conv5 = conv_block(128, 128, pool=True)
        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        # Layer 3
        self.conv6 = conv_block(128, 256)
        self.conv7 = conv_block(256, 256, pool=True)
        self.res3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        # Layer 4
        self.conv8 = conv_block(256, 512)
        self.conv9 = conv_block(512, 512, pool=True)
        self.res4 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        # Classifier
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.res1(out) + out
        
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.res2(out) + out
        
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.res3(out) + out
        
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.res4(out) + out
        
        out = self.classifier(out)
        return out


#CustomResNet18_224--------------------------------------
class CustomResNet18_224(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Initial block
        self.conv1 = conv_block(in_channels, 64, pool=True)  # 112x112 after this
        
        # Layer 1
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        
        # Layer 2
        self.conv4 = conv_block(64, 128, pool=True)  # 56x56 after this
        self.conv5 = conv_block(128, 128)
        self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        # Layer 3
        self.conv6 = conv_block(128, 256, pool=True)  # 28x28 after this
        self.conv7 = conv_block(256, 256)
        self.res3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        # Layer 4
        self.conv8 = conv_block(256, 512, pool=True)  # 14x14 after this
        self.conv9 = conv_block(512, 512)
        self.res4 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        # Classifier
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.res1(out) + out
        
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.res2(out) + out
        
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.res3(out) + out
        
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.res4(out) + out
        
        out = self.classifier(out)
        return out


#ResNet9--------------------------------------
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(7), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


#ResNet9_224--------------------------------------
class ResNet9_224(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.control = {}
        self.delta_control = {}
        self.delta_y = {}
        
        # Initial convolution block without pooling to preserve size.
        # Output size: 224x224
        self.conv1 = conv_block(in_channels, 64) 
        
        # After this block, size becomes 112x112
        self.conv2 = conv_block(64, 128, pool=True) 
        
        # Residual block with no pooling to preserve size
        # Output size: 112x112
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        # After this block, size becomes 56x56
        self.conv3 = conv_block(128, 256, pool=True) 
        
        # After this block, size becomes 28x28
        self.conv4 = conv_block(256, 512, pool=True)
        
        # Residual block with no pooling to preserve size
        # Output size: 28x28
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        # Classifier 
        # Global adaptive pooling to ensure 1x1x512 size before linear layer
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out



#CustomResNet34--------------------------------------
class CustomResNet34(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Initial block without pooling
        self.conv1 = conv_block(in_channels, 64)  # stays 32x32

        # Layer 1 - 3 blocks
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.res1_1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.res1_2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        # Layer 2 - 4 blocks
        self.conv4 = conv_block(64, 128, pool=True)  # 16x16 after this
        self.res2_1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.res2_2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.res2_3 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        # Layer 3 - 6 blocks
        self.conv5 = conv_block(128, 256, pool=True)  # 8x8 after this
        self.res3_1 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res3_2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res3_3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res3_4 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res3_5 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        # Layer 4 - 3 blocks
        self.conv6 = conv_block(256, 512, pool=True)  # 4x4 after this
        self.res4_1 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.res4_2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        # Classifier
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.res1_1(out) + out
        out = self.res1_2(out) + out

        out = self.conv4(out)
        out = self.res2_1(out) + out
        out = self.res2_2(out) + out
        out = self.res2_3(out) + out

        out = self.conv5(out)
        out = self.res3_1(out) + out
        out = self.res3_2(out) + out
        out = self.res3_3(out) + out
        out = self.res3_4(out) + out
        out = self.res3_5(out) + out
        
        out = self.conv6(out)
        out = self.res4_1(out) + out
        out = self.res4_2(out) + out

        out = self.classifier(out)
        return out


#CustomResNet34_224--------------------------------------
class CustomResNet34_224(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Initial block
        self.conv1 = conv_block(in_channels, 64, pool=True)  # 112x112
        
        # Layer 1 - 3 blocks
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.res1_1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.res1_2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        
        # Layer 2 - 4 blocks
        self.conv4 = conv_block(64, 128, pool=True)  # 56x56
        self.res2_1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.res2_2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.res2_3 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        # Layer 3 - 6 blocks
        self.conv5 = conv_block(128, 256, pool=True)  # 28x28
        self.res3_1 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res3_2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res3_3 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res3_4 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.res3_5 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        # Layer 4 - 3 blocks
        self.conv6 = conv_block(256, 512, pool=True)  # 14x14
        self.res4_1 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.res4_2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        # Classifier
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.res1_1(out) + out
        out = self.res1_2(out) + out

        out = self.conv4(out)
        out = self.res2_1(out) + out
        out = self.res2_2(out) + out
        out = self.res2_3(out) + out

        out = self.conv5(out)
        out = self.res3_1(out) + out
        out = self.res3_2(out) + out
        out = self.res3_3(out) + out
        out = self.res3_4(out) + out
        out = self.res3_5(out) + out
        
        out = self.conv6(out)
        out = self.res4_1(out) + out
        out = self.res4_2(out) + out

        out = self.classifier(out)
        return out


