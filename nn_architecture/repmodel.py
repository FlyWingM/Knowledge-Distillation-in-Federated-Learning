import torchvision.models as models
from .resnetcifar import ResNet18_cifar10


import torch.nn as nn
import torch.nn.functional as F
import torch


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ResNet_18Moon(nn.Module):
    def __init__(self, out_dim = 1024, n_classes=2):
        super(ResNet_18Moon, self).__init__()
        
        basemodel = ResNet18_cifar10()
        self.features = nn.Sequential(*list(basemodel.children())[:-1])
        num_ftrs = basemodel.fc.in_features
        
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        self.l3 = nn.Linear(out_dim, n_classes)
        
    def forward(self, x):
        h = self.features(x)

        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return y, x
    
    def validation_step(self, batch):
        images, labels = batch 

        images = images.type(torch.cuda.FloatTensor)

        out, _ = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, n_classes=10, dim=1024):
        super(FedAvgCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        preds = self.fc(out)
        return preds, out
   
    def validation_step(self, batch):
        images, labels = batch 

        images = images.type(torch.cuda.FloatTensor)

        out, _ = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
    
class ResNet_9Moon(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
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
        rep = torch.clone(out)
        out = self.res2(out) + out
        preds = self.classifier(out)
        return preds, rep

    def validation_step(self, batch):
        images, labels = batch 

        images = images.type(torch.cuda.FloatTensor)

        out, _ = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
class DNN(nn.Module):
    def __init__(self, input_dim=3*32*32, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        preds = self.fc(x)
        preds = F.log_softmax(x, dim=1)
        return preds, x
    
    def validation_step(self, batch):
        images, labels = batch 

        images = images.type(torch.cuda.FloatTensor)

        out, _ = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    
batch_size = 16
class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        preds = self.fc(x)
        preds = F.log_softmax(x, dim=1)
        return preds, x

    def validation_step(self, batch):
        images, labels = batch 

        images = images.type(torch.cuda.FloatTensor)

        out, _ = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}