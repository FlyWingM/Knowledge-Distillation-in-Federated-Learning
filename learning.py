# coding: utf-8
# Antonio (seoei2@gmail.com)
# ===================================================================
import numpy as np
import multiprocessing
from config_monitor.args import parse_args
from config_monitor.global_parameters import GlobalParameterClass
from datasets.dataset_selection import DatasetSelection
from datasets.dataset_augmentation import DataAugmentation
#from train_central.centralized_learning import CentralizedTraining
from train_fl.federated_learning import FederatedLearning

# Environment variable for CUDA
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def ML_FL_training(g_para, dataset):

	if g_para.data_distribution_type == "central":
		print("\n\nCentral Training")
		#trainer = CentralizedTraining(g_para, dataset)
		#trainer.train()

	elif g_para.data_distribution_type in ['IID', 'Dirichlet', 'rl', 'IID_rl']:
		if g_para.data_distribution_type in ['IID', 'Dirichlet']:
			print(f"\n\n{g_para.data_distribution_type}")
			trainer = FederatedLearning(g_para, dataset)

		elif g_para.data_distribution_type in ['rl', 'IID_rl']:
			print(f"\n\n{g_para.data_distribution_type}")
			#trainer = FederatedRLTraining(g_para, dataset)
		
		trainer.train()
		#trainer.train_for_knowledge_distillation()

	else:
		print(f"This does not support {g_para.data_distribution_type}; please correct it")


if __name__ == '__main__':

	# Initial Global Parameters
	args = parse_args()
	g_para = GlobalParameterClass(args)

	#Loadind the dataset
	dataset = DatasetSelection(g_para, DataAugmentation())

	multiprocessing.set_start_method('spawn', force=True)

	# Training
	ML_FL_training(g_para, dataset)