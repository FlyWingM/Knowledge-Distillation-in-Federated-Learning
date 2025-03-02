### server.py
import numpy as np
import torch
import torch.nn as nn
import torch.cuda
import statistics
import os
import random
import pandas as pd
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from train_fl.fl_util import get_dir_name, update_data_dir
from train_fl.utility import Utils
from nn_architecture.nn_selection import nn_archiecture_selection, nn_architecture_selection_for_server

class Server:

	def __init__(self, g_para, dataset):
		self.g_para = g_para
		self.device = self.g_para.device
		self.length_iterations = 50

		# The global dataset(s)
		self.dataset = dataset
		self.dataloader_global_test = None
		self.dataloader_global_test_small = None
		self.dataloader_proxy_data_client = None

		# Cached batches for the global test set (preloaded onto GPU or pinned memory)
		self.cached_batches_global_test = []
		self.cached_batches_global_test_small = []

		# Global models
		self.g_model = None
		self.g_model_deposit = None
		self.g_model_highest = None

		self.Debug_1, self.Debug_2 = False, True

		# A bigger global model (if you want a high-capacity model)
		self.g_model_big, self.g_optimizer_big, self.g_criterion_big = None, None, None

		# Dictionary for per-client local models, gradients, etc.
		self.server_local_models = {}
		self.server_local_gradients = {}

		# Scaffold-related
		self.global_control_variate_dict = None
		self.global_control_variate_dict_deposit = None
		self.control_variate_dict = None
		self.delta_w_dict = None
		self.delta_c_dict = None

		# FedProto
		self.protos_dic = {}
		self.global_protos = None

		self.display_server_config_center()
		self.display_config()

		# Create the global dataset(s)
		self.create_global_dataset()

		# Create the global model(s)
		self.create_global_moc()

		# Initialize per-client info (carbon, offsets, etc.)
		self.server_local_info = {}
		self.initialize_client_info()

		# Load carbon data for server/clients
		self.load_carbon_server()

		# Utility class instance
		self.utils = Utils()

	def create_global_dataset(self):
		if self.Debug_2: print(f"\n1. Creating the global dataset")
		dir_name = get_dir_name(self.g_para)
		update_data_dir(self.g_para, dir_name, 0)
		self.dataset.load_dataset_global(self)
		if self.Debug_1: print(f"\n{'#'*10} Compliting the global dataset {'#'*10}\n")


	def setup_load_pretrained_model(self):
		self.g_para.nn["cont_train"] = True
		# Example of a checkpoint file for knowledge distillation
		self.g_para.nn["cont_train_filenames"].append(
			"ResNet9_PFF_b_ls_Dir_d06_iid0_KL_DAF_FedAvg_i900_e1_c50_acc_SS_P99_NA_CR0.6_cT_dT_i421_ST0_ER0_ET0_accH_0.6590.pth"
		)

	def create_global_moc(self):
		if self.Debug_2: print(f"\n3. Creating the global MOC")
		self.g_model = nn_archiecture_selection(self.g_para, self.device)
		self.g_optimizer = torch.optim.SGD(
			self.g_model.parameters(),
			lr=self.g_para.h_param['learning_rate'],
			momentum=self.g_para.h_param['momentum'],
			weight_decay=self.g_para.h_param['weight_decay']
		)
		self.g_criterion = nn.CrossEntropyLoss()

		self.g_model_deposit = nn_archiecture_selection(self.g_para, self.device)
		self.g_model_highest = nn_archiecture_selection(self.g_para, self.device)

		self.g_model_big = nn_architecture_selection_for_server(self.g_para, self.device)
		self.g_optimizer_big = torch.optim.SGD(
			self.g_model_big.parameters(),
			lr=self.g_para.h_param['learning_rate'],
			momentum=self.g_para.h_param['momentum'],
			weight_decay=self.g_para.h_param['weight_decay']
		)


	def display_config(self):
		print(f"{self.g_para.data_distribution_type}-based model trained on {self.g_para.data_name} "
			  f"using {self.g_para.nn['name']} ML Algorithm")
		print(f"learning rate: {self.g_para.h_param['learning_rate']}")
		print(f"{self.g_para.data_name}, {self.g_para.nn['name']}, Trainable({self.g_para.nn['pretrained']}), "
			  f"{self.g_para.avg_algo}, {self.g_para.data_distribution_type},")
		print(f"IID portion: {self.g_para.dataset['iid_data']}, Num of clients: {self.g_para.num_clients},")

	def display_server_config_center(self):
		print(f"\n== A centralized model with {self.g_para.numEpoch} iterations "
			  f"({self.g_para.data_name}), ({self.g_para.nn['name']}), (Pretrained:{self.g_para.nn['pretrained']})")

	def initialize_client_info(self):
		client_info_dict = {
			"client_gradient": None,
			"local_acc": [],
			"local_pca": [0] * self.g_para.num_g_iter,
			"local_weight": [0] * self.g_para.num_g_iter,
			"train_num_samples": 0,
			"local_soft_label": [],
			"country_carbon": None,
			"local_carbon": [0.0] * self.g_para.num_g_iter,
			"local_carbon_intensity": [0] * self.g_para.num_g_iter,
			"local_carbon_intensity_level": [0] * self.g_para.num_g_iter,
			"local_state_1": [0] * self.g_para.num_g_iter,
			"local_state_2": [0] * self.g_para.num_g_iter,
			"local_new_state_1": [0] * self.g_para.num_g_iter,
			"local_new_state_2": [0] * self.g_para.num_g_iter,
			"avg_primary": [0] * self.g_para.num_g_iter,
			"avg_carbon": [0] * self.g_para.num_g_iter,
			"primary_reward": [0] * self.g_para.num_g_iter,
			"carbon_reward": [0] * self.g_para.num_g_iter,
			"local_reward": [0] * self.g_para.num_g_iter,
			"local_select": [False] * self.g_para.num_g_iter,
			"local_distribution": [],
			"kl_div": [0],
			"emd": [0],
			"local_train_time": [0.0] * self.g_para.num_g_iter,
			"local_offset_time": [0.0] * (self.g_para.num_g_iter + 1)
		}

		for client_id in range(self.g_para.num_clients):
			self.server_local_info[client_id] = deepcopy(client_info_dict)


	def load_carbon_server(self):
		column_name = 'Carbon Intensity gCO2-eq/kWh (direct)'  # header/column name to read
		carbon_data = {}
		country_name = []

		carbon_path = self.g_para.path["carbon"]
		for filename in os.listdir(carbon_path):
			if filename.endswith('_hourly.csv'):
				file_path = os.path.join(carbon_path, filename)
				country = filename.split('_')[0]
				country_name.append(country)

		for client_id in range(self.g_para.num_clients):
			selected_country = random.choice(country_name)
			file_path = os.path.join(carbon_path, f"{selected_country}_2023_hourly.csv")
			data = pd.read_csv(file_path, usecols=[column_name], nrows=24 * 14)
			carbon_data[client_id] = data[column_name].tolist()
			self.server_local_info[client_id]["country_carbon"] = selected_country

		self.g_para.carbon_ratio_client = carbon_data

		if self.g_para.Debug["carbon"]:
			for client_id in range(self.g_para.num_clients):
				print(f"{client_id}: {self.g_para.carbon_ratio_client[client_id]}")
				pass


	def preload_global_small_test_dataset_onGPU(self, device):
		"""
		Preloads the global test dataset into GPU memory (or pinned memory).
		This should be called before evaluating the global model (e.g., _train_once()).
		"""
		if not self.dataloader_global_test_small:
			print(f"\n{'#'*10} The server has no global test datasets with small, it might be an error {'#'*10}\n")
			self.create_global_dataset()

		if self.Debug_1: print(f"\n{'#'*10} Creating the cached_batches_global_test_small{'#'*10}\n")
		self.cached_batches_global_test_small = []  # store preloaded data here
		# Move each batch to GPU (or pinned memory) in advance
		for batch_idx, (data, labels) in enumerate(self.dataloader_global_test_small):
			# Using non_blocking=True + pinned memory can overlap CPU-GPU transfer
			data_gpu = data.to(device, non_blocking=True)
			labels_gpu = labels.to(device, non_blocking=True)
			self.cached_batches_global_test_small.append((data_gpu, labels_gpu))
		if self.Debug_1: print(f"\n{'#'*10} Completing the cached_batches_global_test_small {'#'*10}\n")


	def preload_global_test_dataset(self, device):
		"""
		Preloads the global test dataset into GPU memory (or pinned memory).
		This should be called before evaluating the global model (e.g., _train_once()).
		"""
		if not self.dataloader_global_test:
			print(f"\n{'#'*10} The server has no global test datasets, it might be an error {'#'*10}\n")
			self.create_global_dataset()

		if self.Debug_1: print(f"\n{'#'*10} Creating the cached_batches_global_test{'#'*10}\n")
		self.cached_batches_global_test = []  # store preloaded data here
		# Move each batch to GPU (or pinned memory) in advance
		for batch_idx, (data, labels) in enumerate(self.dataloader_global_test):
			# Using non_blocking=True + pinned memory can overlap CPU-GPU transfer
			data_gpu = data.to(device, non_blocking=True)
			labels_gpu = labels.to(device, non_blocking=True)
			self.cached_batches_global_test.append((data_gpu, labels_gpu))
		if self.Debug_1: print(f"\n{'#'*10} Completing the cached_batches_global_test {'#'*10}\n")


	def clear_cached_batches_onGPU(self):
		"""			
		Clears the space allocated by cached_batches_global_test and to free GPU memory.
		"""
		# Clear the testing cached batches
		for batch in self.cached_batches_global_test:
			del batch  # Remove tensor references
		self.cached_batches_global_test.clear()  # Clear the list

		# Release GPU memory
		torch.cuda.empty_cache()
		#print("Cached batches cleared and GPU memory released.")

	@torch.no_grad()
	def evaluate_model(self, model, test_loader, mode):
		"""
		If 'test_loader' is a standard DataLoader, you'll likely move data to device here.
		If 'test_loader' is a list of (already-loaded) GPU batches, you'll skip device.transfer.
		"""
		criterion = nn.CrossEntropyLoss()
		test_loss, correct, total = 0, 0, 0
		model.eval()

		# 'test_loader' could be either a DataLoader or a list of cached GPU batches
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			# If we are passing in a standard DataLoader for small test sets, move data to device:
			if mode == 'global_test_small':
				inputs, targets = inputs.to(self.device), targets.to(self.device)

			outputs = model(inputs)  # Generate predictions
			loss = criterion(outputs, targets)  # Calculate loss

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

		# Logging
		self.utils.progress_bar(
			"The global model evaluation",
			batch_idx,
			len(test_loader),
			f'Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total})'
		)

		avg_acc = correct / total if total > 0 else 0.0
		avg_loss = test_loss / (batch_idx + 1) if batch_idx >= 0 else 0.0
		return {'test_acc': round(avg_acc, 4), 'test_loss': round(avg_loss, 4)}


	def gmodel_evaluate(self, mode='global_test'):
		"""
		Evaluate the main global model (self.g_model). 
		If mode='global_test', we assume cached_batches_global_test is used (already on GPU).
		If mode='global_test_small', we evaluate via a standard DataLoader that is small enough 
		to do direct .to(self.device).
		"""
		g_index = self.g_para.g_iter_index
		try:
			if mode == 'global_test':
				# Evaluate using the preloaded GPU data:
				result = self.evaluate_model(self.g_model, self.cached_batches_global_test, mode)
			elif mode == 'global_test_small':
				# Evaluate using a standard (small) DataLoader:
				result = self.evaluate_model(self.g_model, self.dataloader_global_test_small, mode)
			else:
				raise ValueError(f"Unknown mode: {mode} (expected 'global_test' or 'global_test_small').")

			current_acc = result["test_acc"]
			self.g_para.global_info["g_acc"][g_index] = current_acc
			
			# Check for highest accuracy so far
			if current_acc > self.g_para.global_info["highest_acc"]:
				self.g_para.global_info["highest_acc"] = current_acc
				self.g_para.global_info["highest_iter"] = g_index

			# Example logic to unfreeze pretrained layers (just an illustration):
			if self.g_para.nn["trainable"] and self.g_para.nn["pretrained"] and g_index > self.length_iterations:
				import statistics
				recent_mean_acc = statistics.mean(
					self.g_para.global_info["g_acc"][(g_index - self.length_iterations):g_index]
				)
				if current_acc < recent_mean_acc:
					print("Unfreezing parameters for further training.")
					# Example: your code might track a list/dict of client models
					# for client_id, client_model in self.clients.items():
					#     for param in client_model.parameters():
					#         param.requires_grad = True
					#
					# (Implementation depends on the rest of your code)
				else:
					print("No unfreezing required. Model is still improving.")

			print(f"\n==> Global model on Test Dataset: "
				  f"Acc={result['test_acc']:.4f}, Loss={result['test_loss']:.4f}")

		except Exception as e:
			print(f"An error occurred in gmodel_evaluate: {e}")


	def high_capacity_gmodel_evaluation(self):
		"""
		Evaluate the high-capacity global model (self.g_model_big) 
		on the same test dataset (cached_batches_global_test or normal loader).
		"""
		g_index = self.g_para.g_iter_index
		try:
			# If you want to evaluate with preloaded GPU data, do:
			result = self.evaluate_model(self.g_model_big, self.cached_batches_global_test, 'global_test')

			high_big_current_acc = result["test_acc"]
			curren_acc_light = self.g_para.global_info["g_acc"][g_index]

			print(f"\n# >>>>The high-capacity global model ({high_big_current_acc:7.4f}), "
				  f"low-capacity model ({curren_acc_light:7.4f}), Loss:{result['test_loss']:7.4f}")

			# Update highest_acc_big if needed
			if high_big_current_acc > self.g_para.global_info["highest_acc_big"]:
				self.g_para.global_info["highest_acc_big"] = high_big_current_acc
				print(f"\n# >>>>The high-capacity global model ({high_big_current_acc:7.4f}) "
					  f"achieves the highest accuracy; thus, it will be saved.")
				return True
			return False

		except Exception as e:
			print(f"An error occurred in high_capacity_gmodel_evaluation: {e}")

