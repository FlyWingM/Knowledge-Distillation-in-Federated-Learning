# coding: utf-8?
# Eunil Seo (seoei2@gmail.com)
# ===================================================================

import pandas as pd
import numpy as np
from PIL import Image
import errno    
import os

from pathlib import Path
import requests
import pickle
import gzip

import torch
import torch.nn.functional as F
from torch import nn

import torchvision
from torchvision.datasets import EMNIST
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100

import torchvision.transforms as tt
from torch.utils.data import DataLoader

from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

import util.utility as util 

sns.set(rc={'figure.figsize':(10,6)})
sns.set(font_scale=1.3)
plt.style.use('fivethirtyeight')

'''
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize( 
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
    )
])
'''

image_size = [32, 32]
#image_size = [224, 224]
      

# This normalization comes from https://github.com/xternalz/WideResNet-pytorch
normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
)


def load_cifar100(g_para):

    g_para.input_size = image_size[0]

    # Define a transform to normalize the data
    #stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))

    train_transform = transforms.Compose([
        tt.RandomHorizontalFlip(),
        #tt.RandomCrop(32,padding=4,padding_mode="reflect"),
        #tt.RandomCrop(32,padding=4),
        tt.Resize(image_size),
        tt.ToTensor(),
        normalize
#        tt.Normalize = normalize
        #transforms.Normalize(*stats)
    ])
    #train_transform.transforms.append(normalize)

    test_transform = transforms.Compose([
        tt.ToTensor(),
        tt.Resize(image_size),
        normalize
    ])
    #test_transform.transforms.append(normalize)	
    
    #dataset = datasets.CIFAR10("../data/Cifar10", download = True, train = True, transform = transform)
    #test_dataset = datasets.CIFAR10("../data/Cifar10", download = True, train = False, transform = test_transform)

    train_dataset = CIFAR100(root="../data/Cifar100",
                                        train=True,                                        
                                        transform=train_transform,
                                        download=True)
                                        
    test_dataset = CIFAR100(root="../data/Cifar100",
                                        train=False,
                                        transform=test_transform,
                                        download=True)

    # Loading and analysing the train train_dataset -------------------------------------------------
    print("Total No of Images in CIFA100 train_dataset:", len(train_dataset) + len(test_dataset))
    print("No of images in Training dataset:    ",len(train_dataset))
    print("No of images in Testing dataset:     ",len(test_dataset))
    
    print("No of classes: ",len(train_dataset.classes))
    print("List of all classes {} ,labeled as {} (not in order)".format(train_dataset.classes, np.unique(train_dataset.targets)))
    
    #img, target = train_dataset.__getitem__(100)
    #imshow(img)
    
    # We can limite the number of label to evaluate the value of one data
    if (g_para.num_of_label == 0):
        g_para.num_of_label = len(train_dataset.classes)
    else:
        g_para.num_of_label = g_para.num_of_label

    train_dataset = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           #batch_size=g_para.batch_size,
                                           batch_size=len(train_dataset),
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)
    train_dataset = iter(train_dataset)   
    #x, y = train_dataset.next()
    x, y = next(train_dataset)
    
    print(type(x))
    print(x.shape)
    print(y.shape)

    print("\nAfter numpy() x type:({}), x type:({})".format(type(x), type(y)))

    #x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=.15, stratify=y)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=.20, stratify=y)
    print("\nAfter train_test_split")
    print("x_train type:({}), shape({}) / y_train type:({}), shape({})"\
            .format(type(x_train), x_train.shape, type(y_train), y_train.shape))
    print("x_valid type:({}), shape({}) / y_valid type:({}), shape({})"\
            .format(type(x_valid), x_valid.shape, type(y_valid), y_valid.shape))

    
    test_dataset = torch.utils.data.DataLoader(dataset=test_dataset,
                    #batch_size=g_para.batch_size,
                    batch_size=len(test_dataset)) 
                    #shuffle=True, pin_memory=True, num_workers=2)
    test_dataset = iter(test_dataset)
    #x_test, y_test = test_dataset.next()
    x_test, y_test = next(test_dataset)

    print("x_test type:({}), shape({}) / y_test type:({}), shape({})\n"\
          .format(type(x_test), x_test.shape, type(y_test), y_test.shape))

    y_train_total=0
    y_valid_total=0
    y_test_total=0
    total=0
    
    for i in range(g_para.num_of_label):
        print(i,">> train:", sum(y_train==i), ", valid:", sum(y_valid==i), ", test:", sum(y_test==i), 
                                ", total:", sum(y_train==i)+sum(y_valid==i)+sum(y_test==i) )
        g_para.train_amount_per_label.append(sum(y_train==i))
        g_para.valid_amount_per_label.append(sum(y_valid==i))
        g_para.test_amount_per_label.append(sum(y_test==i))
        total=total+sum(y_train==i)+sum(y_valid==i)+sum(y_test==i)

    y_train_total = sum(g_para.train_amount_per_label)
    y_valid_total = sum(g_para.valid_amount_per_label)
    y_test_total    = sum(g_para.test_amount_per_label)

    print("\n\nThe final number of label is {}".format( g_para.num_of_label))
    print("y_train_total({}), train_amount_per_label({})\n".format(y_train_total, g_para.train_amount_per_label))
    print("y_valid_total({}), valid_amount_per_label({})\n".format(y_valid_total, g_para.valid_amount_per_label))
    print("y_test_total({}), test_amount_per_label({})\n-".format(y_test_total, g_para.test_amount_per_label))
    print("total=", total)
    
    #return map(torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)) 
    return x_train, y_train, x_valid, y_valid, x_test, y_test
