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
