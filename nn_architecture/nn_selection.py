### nn_selection.py

import torch
import torch.nn as nn

from nn_architecture.custom_resnet import (
	ResNet9, ResNet9_224, CustomResNet18, CustomResNet18_224, CustomResNet34, CustomResNet34_224)

from nn_architecture.densenet import (
	densenet121, densenet161, densenet169, densenet201)

from nn_architecture.resnet import (
	resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2)

from nn_architecture.repmodel import (
	ResNet_18Moon, FedAvgCNN, ResNet_9Moon, DNN)

#from nn_architecture.vgg import (vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_custom, vgg16_bn, vgg19, vgg19_bn)

from nn_architecture.mobilenet_v2 import mobilenet_v2

from nn_architecture.res_net import (
	res_net18, res_net34, res_net50, res_net101, res_net152)

from nn_architecture.dense_net import (
	dense_net121, dense_net161, dense_net169, dense_net201)

#from nn_architecture.mobilenet_v2 import (
from nn_architecture.mobilenetv2 import (
	MobileNetV2)

from save_load.save_load_model import construct_global_model_filename, cmodel_load


nn_archi = {
	#"resnet18": resnet18,
	#"resnet34": resnet34,
	#"resnet50": resnet50,
	#"resnet101": resnet101,

	"resnet18": res_net18,
	"resnet34": res_net34,
	"resnet50": res_net50,
	"resnet101": res_net101,
	"resnet152": res_net152,

	"resnext50_32x4d": resnext50_32x4d,
	"resnext101_32x8d": resnext101_32x8d,
	"resnext101_64x4d": resnext101_64x4d,
	"wide_resnet50_2": wide_resnet50_2,
	"wide_resnet101_2": wide_resnet101_2,

	#"densenet121": densenet121,    #    densenet161, densenet169, densenet201
	"densenet121": dense_net121,    #    densenet161, densenet169, densenet201

	#"vgg11": vgg11, "vgg11_bn": vgg11_bn, "vgg13": vgg13, "vgg13_bn": vgg13_bn, #"vgg16": vgg16, "vgg16": vgg16_custom, "vgg16_bn": vgg16_bn, "vgg19": vgg19, "vgg19_bn": vgg19_bn, 

	"ResNet_18Moon": ResNet_18Moon,

	"FedAvgCNN": FedAvgCNN,
	"ResNet_9Moon": ResNet_9Moon,
	"DNN": DNN,
	"ResNet9": ResNet9,
	"ResNet9_224": ResNet9_224,
	"CustomResNet18": CustomResNet18,
	"CustomResNet18_224": CustomResNet18_224,
	"CustomResNet34": CustomResNet34,
	"CustomResNet34_224": CustomResNet34_224,
	 
	"MobileNetV2": MobileNetV2 
}

default_resnet_model_list = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d", "wide_resnet50_2", "wide_resnet101_2"]
default_vgg_model_list = ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "MobileNetV2" ]
default_simplified_model_list = [res_net18, res_net34, res_net50, res_net101, res_net152]


def nn_archiecture_selection(g_para, device):
	if g_para.Debug['model']:
		print(f"nn_archiecture_selection with the local {g_para.nn['name']}")

	model = None
	model_name = g_para.nn["name"]

	# Check if model name is in nn_archi
	if model_name not in nn_archi:
		print(f"{model_name} is not supported, try another")
		return None

	# To train the customized neural networks
	if model_name in ["ResNet9", "CustomResNet18", "CustomResNet34"]:
		# Select appropriate model key based on image size
		if g_para.data_name == "cinic10_big":
			model_key = f"{model_name}_224"
		else:
			model_key = model_name

		print(f"{model_key} is created with the pretrained model: ({g_para.nn['pretrained']})")
		model = nn_archi[model_key](g_para.nn["num_channel"], g_para.num_of_label)

		# Load existing model if continuation of training is required
		if g_para.nn["cont_train"]:
			model_path = cmodel_load(g_para) if g_para.data_distribution_type == "central" else construct_global_model_filename(g_para)
			model.load_state_dict(torch.load(model_path))
			g_para.nn["cont_train"] = False

		return model.to(device)

	# To train MobileNetV2
	elif model_name == "MobileNetV2":
		print(f"{model_name} is created with the pretrained model: ({g_para.nn['pretrained']})")
		model = nn_archi[model_name]()

		# Load existing model if continuation of training is required
		if g_para.nn["cont_train"]:
			model_path = cmodel_load(g_para) if g_para.data_distribution_type == "central" else construct_global_model_filename(g_para)
			model.load_state_dict(torch.load(model_path))
			g_para.nn["cont_train"] = False

		return model.to(device)

	# To train densenet models
	elif model_name in ["densenet121", "dense_net121"]:
		# Initialize model based on pretrained flag
		if g_para.nn["pretrained"]:
			model = nn_archi[model_name](pretrained=True)
			print(f"{model_name} is created with the pretrained model: ({g_para.nn['pretrained']})")

			# Freeze parameters except for denseblock4
			for param in model.parameters():
				param.requires_grad = False
			for param in model.features.denseblock4.parameters():
				param.requires_grad = True
		else:
			model = nn_archi[model_name]()
			print(f"{model_name} is created without pretrained weights")

		# Load existing model if continuation of training is required
		if g_para.nn["cont_train"]:
			model_path = cmodel_load(g_para) if g_para.data_distribution_type == "central" else construct_global_model_filename(g_para)
			model.load_state_dict(torch.load(model_path))
			g_para.nn["cont_train"] = False

		# Replace the classifier part
		#num_ftrs = model.classifier.in_features
		#model.classifier = nn.Linear(num_ftrs, g_para.num_of_label)

		return model.to(device)

	# For default ResNet and VGG models
	elif model_name in default_resnet_model_list or model_name in default_vgg_model_list:
		# Initialize model based on pretrained flag
		if g_para.nn["pretrained"]:
			model = nn_archi[model_name](pretrained=True)
			print(f"{model_name} is created with the pretrained model: ({g_para.nn['pretrained']})")
		else:
			model = nn_archi[model_name](pretrained=False)
			print(f"{model_name} is created without pretrained weights")

		# Load existing model if continuation of training is required
		if g_para.nn["cont_train"]:
			model_path = cmodel_load(g_para) if g_para.data_distribution_type == "central" else construct_global_model_filename(g_para)
			model.load_state_dict(torch.load(model_path))
			g_para.nn["cont_train"] = False

		# Replace the classifier part
		if model_name in default_resnet_model_list:
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs, g_para.num_of_label)
		elif model_name in default_vgg_model_list:
			num_ftrs = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(num_ftrs, g_para.num_of_label)

		return model.to(device)

	else:
		print(f"{model_name} is not supported, try another")
		return None


def nn_architecture_selection_for_server(g_para, device):
	if g_para.Debug['model']:
		print(f"nn_architecture_selection_for_server with global {g_para.nn['name_server']}")

	model = None
	model_name = g_para.nn["name_server"]

	# Check if model name is in nn_archi
	if model_name not in nn_archi:
		print(f"{model_name} is not supported, try another")
		return None

	# To train densenet models
	if model_name in ["densenet121", "dense_net121"]:
		# Initialize model based on pretrained flag
		if g_para.nn["pretrained"]:
			model = nn_archi[model_name](pretrained=True)
			print(f"{model_name} is created and the pretrained model is loaded?: ({g_para.nn['pretrained']})")

			# Freeze parameters except for denseblock4
			for param in model.parameters():
				param.requires_grad = False
			for param in model.features.denseblock4.parameters():
				param.requires_grad = True
		else:
			model = nn_archi[model_name]()
			print(f"{model_name} is created without pretrained weights")

		# Load existing model if continuation of training is required
		if g_para.nn["cont_train"]:
			model_path = cmodel_load(g_para) if g_para.data_distribution_type == "central" else construct_global_model_filename(g_para)
			model.load_state_dict(torch.load(model_path))
			g_para.nn["cont_train"] = False

		# Replace the classifier part
		#num_ftrs = model.classifier.in_features
		#model.classifier = nn.Linear(num_ftrs, g_para.num_of_label)

		return model.to(device)

	else:
		print(f"{model_name} is not supported, try another")
		return None