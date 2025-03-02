### federated_learning.py
import os
import pandas as pd
import torch

from train_fl.server import Server
from train_fl.client import Client

# Aggregation/transmission modules
#from train_fl.model_aggregation_transmission_1 import (
from train_fl.model_aggregation_transmission_0 import (
	set_averaged_weights_in_global_model,
	transmission_gmodel_to_local_clients,
)

# The persistent manager you revised
from train_fl.client_manager import ClientManager

from train_fl.client_selection import ClientSelection
from knowledge_distillation.knowledge_distillation import KnowledgeDistillation
from save_load.save_results import save_results
from save_load.save_load_model import gmodel_save


class FLParameters:
	"""
	Simple container for FL parameters (g_para, dataset, server, clients).
	"""
	def __init__(self, g_para, dataset, server, clients):
		self.g_para = g_para
		self.dataset = dataset
		self.server = server
		self.clients = clients


class FederatedLearning:
	def __init__(self, g_para, dataset):
		# 1) Set device
		g_para.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# 2) Store references
		self.g_para = g_para
		self.dataset = dataset

		# 3) Create the server
		self.server = Server(g_para, self.dataset)

		# 4) Create all clients in a single process (no Pool)
		self.clients = {}
		for c_index in range(self.g_para.num_clients):
			client_obj = Client(g_para, self.server, c_index, dataset)
			self.clients[c_index] = client_obj

		print("All clients have been created (single-process).")

		# 5) If continuing from a certain accuracy, set that
		self.g_para.global_info["highest_acc"] = (
			-1 if g_para.cont_tr_acc == 0.0 else g_para.cont_tr_acc
		)

		# 6) Build FLParameters
		self.fl_params = FLParameters(self.g_para, self.dataset, self.server, self.clients)

		# 7) Create the (persistent) client manager, which internally spawns one worker per client.
		self.client_manager = ClientManager(self.fl_params)

		self.client_selector = ClientSelection(self.fl_params)
		self.knowledge_distillation = KnowledgeDistillation(self.fl_params)

	def display_begin(self):
		client_select_param = self.g_para.client_select_param
		print(
			f"\n\n5. {self.g_para.g_iter_index}-th, {self.g_para.data_name}, "
			f"{self.g_para.nn['name']}, pretrain({self.g_para.nn['pretrained']}-"
			f"{self.g_para.nn['trainable']}), {self.g_para.pre_distr}, "
			f"sel({client_select_param['criteria']}-"
			f"{client_select_param['weighting']}-"
			f"{client_select_param['c_ratio']}-"
			f"{client_select_param['lr_carbon']})"
		)

	def train(self):
		"""
		Main federated loop using persistent ClientManager.
		"""
		for global_ind in range(self.g_para.cont_tr_iter, self.g_para.num_g_iter):
			self.g_para.g_iter_index = global_ind
			self.display_begin()

			# 1) Select the clients for this round
			self.client_selector.determine_selected_clients_by_next_selected_client()

			# 2) (Optional) Send global model to local clients
			#transmission_gmodel_to_local_clients(self.g_para, self.server, self.clients)
	

			# 3) Train selected clients via the persistent worker model
			self.client_manager.train_clients()

			# 4) Possibly choose next round’s clients
			self.client_selector.determine_next_selected_clients()

			# 5) Aggregate local updates into the global model
			self.client_manager.set_averaged_weights_in_global_model()

			# 6) Evaluate the global model
			self.server.gmodel_evaluate()

			# 7) Optional knowledge distillation if accuracy is above threshold
			if self.g_para.global_info['g_acc'][global_ind] > 1: # 0.65:
				self.knowledge_distillation.distillation_run()
				better_high_capacity_model_acc_ = self.server.high_capacity_gmodel_evaluation()
				if better_high_capacity_model_acc_:
					gmodel_save(self.g_para, self.server, global_ind, mode="high_capacity")

			# After evaluating the global model, let it remove to make space in GPU.
			self.server.clear_cached_batches_onGPU()

			# 8) Save per-round results
			save_results(self.g_para, server=self.server, clients=self.clients, mode="global_model_results")
			if (global_ind == self.g_para.global_info["highest_iter"]
					or global_ind % 5 == 4
					or global_ind == self.g_para.num_g_iter - 1):
				gmodel_save(self.g_para, self.server, global_ind)

		# Final info
		print(
			f"\n{global_ind}-th global iteration, "
			f"global model acc({self.g_para.global_info['g_acc'][global_ind]}), "
			f"global train time({round(self.g_para.global_info['total_local_train_time'][global_ind], 2)})"
		)

	def train_for_knowledge_distillation(self):
		"""
		Example method if you only want a knowledge distillation step.
		"""
		self.display_begin()
		self.client_selector.full_participation()
		transmission_gmodel_to_local_clients(self.g_para, self.server, self.clients)
		self.knowledge_distillation.distillation_run()
		self.server.high_capacity_gmodel_evaluation()
		# Optionally: self.client_manager.shutdown_workers() here if you only do KD
