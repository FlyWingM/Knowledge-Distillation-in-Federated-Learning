# coding: utf-8?
# Eunil Seo (seoei2@gmail.com)
# ===================================================================

from sys import getsizeof
import pandas as pd
import numpy as np
from PIL import Image
import binascii
import errno    
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot
from pathlib import Path
import requests
import pickle
import gzip

import torch
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset


import torchvision.transforms as tt
from torch.utils.data import random_split

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10,6)})
sns.set(font_scale=1.3)
plt.style.use('fivethirtyeight')


def scale_data_f(data):

	minMaxScaler = MinMaxScaler(feature_range=(0, 1))
	minMaxScaler.fit(data)
	
	data_minMaxScaled = minMaxScaler.transform(data)
	print("data_minMaxScaled: \n", data_minMaxScaled)
	
	#data_minMaxScaled_multiple = numpy.uint64(data_minMaxScaled)
	#print("data_minMaxScaled_multiple: \n", data_minMaxScaled_multiple)

	return data_minMaxScaled
	
def load_dataset_mnist():

	# Let's download data set
	DATA_PATH = Path("../data")
	PATH = DATA_PATH / "mnist"
	# PATH = os.join.path(p1, p2)
	PATH.mkdir(parents=True, exist_ok=True)

	URL = "http://deeplearning.net/data/mnist/"
	FILENAME = "mnist.pkl.gz"

	if not (PATH / FILENAME).exists():
			content = requests.get(URL + FILENAME).content
			(PATH / FILENAME).open("wb").write(content)


	with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
			((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

	# Let's see the dataset size
	x_train.shape, y_train.shape , x_valid.shape, y_valid.shape, x_test.shape, y_test.shape
	print(x_train.shape, y_train.shape)
	print(x_test.shape, y_test.shape)
	
	fig, axes = pyplot.subplots(8,8,figsize=(8,8))
	for i in range(8):
		for j in range(8):
			num_index = np.random.randint(len(x_train))
			axes[i,j].imshow(x_train[num_index].reshape((28,28)), cmap="gray")
			axes[i,j].axis("off")
	pyplot.show()

	# Let's check how many of each tag are.
	y_train_total=0
	y_valid_total=0
	y_test_total=0
	total=0
	for i in range(10):
		print(i,">> train:", sum(y_train==i), ", valid:", sum(y_valid==i), 
			  ", test:", sum(y_test==i), ", total:", sum(y_train==i)+sum(y_valid==i)+sum(y_test==i) )
		y_train_total=y_train_total + sum(y_train==i)
		y_valid_total=y_valid_total + sum(y_valid==i)
		y_test_total=y_test_total + sum(y_test==i)
		total=total+sum(y_train==i)+sum(y_valid==i)+sum(y_test==i)
		
	print("y_train_total=", y_train_total) 
	print("y_valid_total=", y_valid_total) 
	print("y_test_total=", y_test_total)
	print("total=", total)

	return x_train, y_train, x_valid, y_valid, x_test, y_test


def to_char(num):
	if num<10:
		return str(num)
	elif num < 36:
		return chr(num+55)
	else:
		return chr(num+61)


def show_example(data):
	img, label = data
	print("Label: ("+to_char(label)+")")
	plt.imshow(img[0], cmap="gray")


def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise


def model_parameter():

#[0: data_size, 1: data_distribution, 2: num_of_local iteration, 3: extra, 4:beta_1, 5: beta_2, 6: max_acc]
	model_global = np.array([
								[  10,  0.1,    1,    0, 0.0903, 0.0248, 0.9297], 
								[  20,  0.1,    1,    0, 0.0979, 0.0260, 0.9499], 
								[  30,  0.1,    1,    0, 0.1156, 0.0423, 0.9585], 
								[  40,  0.1,    1,    0, 0.0957, 0.0440, 0.9678], 
								[  50,  0.1,    1,    0, 0.0867, 0.0444, 0.9710], 
								[  60,  0.1,    1,    0, 0.0891, 0.0519, 0.9734], 
								[  70,  0.1,    1,    0, 0.0783, 0.0468, 0.9759], 
								[  80,  0.1,    1,    0, 0.0735, 0.0467, 0.9771], 
								[  90,  0.1,    1,    0, 0.0794, 0.0610, 0.9771], 
								[ 100,  0.1,    1,    0, 0.0753, 0.0654, 0.9793]
								])

	return model_global


def trim_dataset_t(mat,batch_size):

	no_of_rows_drop = mat.shape[0]%batch_size
	if no_of_rows_drop > 0:
		return mat[:-no_of_rows_drop]
	else:
		return mat


def get_default_device():
	if torch.cuda.is_available():
		return torch.device('cuda')
	else:
		return torch.device('cpu')
