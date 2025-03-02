
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from save_load.save_results import save_results
from save_load.save_load_model import cmodel_save

def get_dir_name(g_para):
	return g_para.data_name.split('_')[0] if g_para.data_name in ['cinic10_big', 'cifar10_big'] else g_para.data_name


def construct_merged_data_path(g_para, client_n, shared_suffix=None):
	#path = f'/proj/seo-220318/myfl/data/{dir_name}-fl/'
	if shared_suffix in ['client']:
		#path = os.path.join(g_para.path[g_para.data_name], f'{g_para.pre_distr}-{g_para.num_clients}', f'client-{client_n}/')
		path = os.path.join(g_para.path[g_para.data_name], f'{g_para.pre_distr}-50', f'client-{client_n}/')
	elif shared_suffix in ['orig']:
		if g_para.data_name in ['cifar10', 'cifar10_big']:
			path = os.path.join(g_para.path[g_para.data_name], 'cifar10_orig/')
		elif g_para.data_name in ['cinic10', 'cinic10_big']:
			path = os.path.join(g_para.path[g_para.data_name], 'cinic10_orig/')
		else:
			print(f"construct_merged_data_path does not support {g_para.data_name}")
			exit()
	elif shared_suffix in ['shared_data/', 'shared_data_synthe/', 'shared_data_synthe_progressive_GAN/']:
		path = os.path.join(g_para.path[g_para.data_name], f'{g_para.pre_distr}-{g_para.num_clients}', shared_suffix)
	return path


def update_data_dir(g_para, dir_name, client_id):

	if g_para.dataset['iid_data'] > 0:
		client_data_dir = construct_merged_data_path(g_para, client_id, shared_suffix='client')
		orig_data_dir = construct_merged_data_path(g_para, client_id, shared_suffix='orig')
		#shared_data_dir = construct_merged_data_path(g_para, client_id, shared_suffix='shared_data/')

		shared_data_dir = construct_merged_data_path(g_para, client_id, shared_suffix= g_para.shared_dir+'/')
		g_para.data_dir = [client_data_dir, shared_data_dir, orig_data_dir]

	else:
		client_data_dir = construct_merged_data_path(g_para, client_id, shared_suffix='client')
		orig_data_dir = construct_merged_data_path(g_para, client_id, shared_suffix='orig')
		g_para.data_dir = [client_data_dir, orig_data_dir]


def update_central_monitoring(g_para, epoch, server, central_accuracy):
	g_para.c_global_info["central_acc"][epoch] = central_accuracy
	g_para.epoch_index = epoch
	save_results(g_para, server=server, clients=None, mode="central_acc")

	if central_accuracy > g_para.c_global_info['c_highest_acc'] or epoch % 5 == 0 or epoch == g_para.numEpoch-1:
		cmodel_save(g_para, server, epoch)


def get_mean_and_std(dataset):
	'''Compute the mean and std value of dataset.'''
	dataloader = torch.utils.data.DataLoader(
		dataset, batch_size=1, shuffle=True, num_workers=2)
	mean = torch.zeros(3)
	std = torch.zeros(3)
	print('==> Computing mean and std..')
	for inputs, _ in dataloader:
		for i in range(3):
			mean[i] += inputs[:, i, :, :].mean()
			std[i] += inputs[:, i, :, :].std()
	mean.div_(len(dataset))
	std.div_(len(dataset))
	return mean, std


def init_params(net):
	'''Init layer parameters.'''
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			init.kaiming_normal(m.weight, mode='fan_out')
			if m.bias:
				init.constant(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			init.constant(m.weight, 1)
			init.constant(m.bias, 0)
		elif isinstance(m, nn.Linear):
			init.normal(m.weight, std=1e-3)
			if m.bias:
				init.constant(m.bias, 0)


