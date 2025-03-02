### global_parameters.py

import numpy as np
from util.utility import get_default_device

class GlobalParameterClass:

	def __init__(self, args):
		self.args = args
		self.initialize_default_parameters()
		self.update_parameters_with_args()

	def initialize_default_parameters(self):
		# Initialize default parameters
		self.dataset = {
			"data_name": "cinic10",
			"data_distribution_type": "Dirichlet",
			"data_aug": 'DAF',
			"iid_data": 0,
			"pretrained": False
		}

		self.data_name = 'cinic10'
		self.num_g_iter = 1400
		self.g_iter_index = 0
		self.hour_index = 0
		self.num_clients = 50
		self.data_distribution_type = "Dirichlet"
		self.pre_distr = ''

		self.num_of_label = 10
		self.configure_num_label = 0
		self.num_of_label_list = []

		self.label_drop = False
		self.label_extension = False
		self.num_label_ELS = 24
		self.label_ex = 0

		self.nn = {
			"name": "ResNet9",				# ResNet9
			"name_server": "densenet121",
			"pretrained": False,
			"trainable": False,
			"cont_train": False,
			"cont_train_filenames": [],
			"cont_train_rl": False,
			"cont_train_rl_filenames": {},
			"num_channel": 3
		}

		self.global_info = {
			"g_acc": [],
			"gmodel_name": '',
			"h_gmodel_name": '',
			"global_distribution": [],
			"g_acc_personal": [],
			"pca": [],
			"highest_acc": -1,
			"highest_iter": 0,
			"highest_acc_big": -1,
			"highest_iter_big": 0,
			"num_iter": 0,
			"current_iter": 0,
			"total_local_train_time": [],
			"local_carbon_total": [],
			"local_kl_avg": [],
			"entropy_ratio": [],
			"client_sel_n": []
		}

		self.client_select_param = {
			"criteria": "NA",
			"lr_carbon": True,
			"dynamic_sel": True,
			"softmax": False,
			"weighting": "NA",
			"c_ratio": 1.0,
			"learning_exploration": 'SS',
			"distribution_similarity": "KL"
		}

		self.rl = {
			"primary_ratio": 0.99,
			"carbon_ratio": 0.01,
			"s_train_counter": 0,
			"m_exploration_counter": 0,
			"exploitation_counter": 0
		}

		self.c_global_info = {
			"central_acc": np.zeros(self.num_g_iter),
			"c_highest_acc": 0,
			"c_highest_iter": 0,
			"c_prev_acc": 0,
			"c_prev_iter": 0,
			"unfrozen": False
		}

		self.device = get_default_device()

		self.h_param = {}
		self.avg_algo = 'FedAvg'
		self.Debug = {
			"data": False,
			"server": False,
			"model": False,
			"carbon": False,
			"d_train": False,
			"d_lr": False,
			"d_distribution": False,
			"rl_contin": False
		}

		self.data_dir = []
		self.shared_dir = "shared_data_synthe"
		self.environ = {"SBATCH": False}
		self.version = "v_2501"
		self.path = {
			"cinic10": "/proj/seo-220318/myfl/data/cinic10-fl",
			"cinic10_big": "/proj/seo-220318/myfl/data/cinic10-fl",
			"cifar10": "/proj/seo-220318/myfl/data/cifar10-fl",
			"cifar10_big": "/proj/seo-220318/myfl/data/cifar10-fl",
			"carbon": "/proj/seo-220318/myfl/data/info_carbon",
			"distribution": "/proj/seo-220318/myfl/data/info_distribution",
			"clients": "C:/Users/eunils/Dropbox/0. Research/2. ML_simulation/federated_learning/output/2409/cinic10_r1/rl_statistics/local_models"
		}
		self.path1 = {
			"cinic10": "/home/eunils/myfl/data/cinic10-fl",
			"cinic10_big": "/home/eunils/myfl/data/cinic10-fl",
			"cifar10": "/home/eunils/myfl/data/cifar10-fl",
			"cifar10_big": "/home/eunils/myfl/data/cifar10-fl",
			"carbon": "/home/eunils/myfl/data/info_carbon",
			"distribution": "/home/eunils/myfl/data/info_distribution"
		}

		self.output_dir = {
			"v_2408": "/proj/seo-220318/myfl/2408/output",
			"v_2409": "/proj/seo-220318/myfl/2409/output",
			"v_2501": "/proj/seo-220318/myfl/kfl/output"
		}
		self.root_dir = "../../output/2409/cinic10_r1"
		self.trained_model_dir = "./trained_models"
		self.gan_dir = {
			"generator_cifar10_dir": "/proj/seo-220318/myfl/data/cifar10-fl/gan_generator",
			"synthesized_images_cifar10_dir": "/proj/seo-220318/myfl/data/cifar10-fl/synthesized_images",
			"gen_name": "cifar10_generator_model"
		}

		self.l_model_target_acc = []
		self.carbon_ratio_client = {}

		self.selected_client_group = {}
		self.next_selected_client_group = {}
		self.nonselected_client_group = {}
		self.nonselected_client_group_list = []

		self.numEpoch = 3
		self.client_eval_with_iid = False
		self.cont_tr_iter = 0
		self.cont_tr_acc = 0.0

	def update_parameters_with_args(self):
		# Update parameters based on args
		self.data_name = self.args.data or self.data_name
		self.avg_algo = self.args.fl or self.avg_algo
		self.dataset['iid_data'] = self.args.num_s if self.args.num_s > 0 else self.dataset['iid_data']

		# Update NN parameters
		nn_mapping = {
			'nn': 'name',
			'pre_trained': 'pretrained',
			'nn_trainable': 'trainable',
			'cont_train': 'cont_train',
			'cont_train_rl': 'cont_train_rl'
		}
		self.nn.update({key: getattr(self.args, arg) for arg, key in nn_mapping.items() if getattr(self.args, arg) is not None})

		# Update client selection parameters
		cs_mapping = {
			'c_sel_cri': 'criteria',
			'lr_carbon': 'lr_carbon',
			'dynamic_sel': 'dynamic_sel',
			'l_explore': 'learning_exploration',
			'c_sel_weighting': 'weighting',
			'c_sel_rate': 'c_ratio'
		}
		self.client_select_param.update({key: getattr(self.args, arg) for arg, key in cs_mapping.items() if getattr(self.args, arg) is not None})

		if self.rl['primary_ratio'] + self.rl['carbon_ratio'] != 1.0:
			print(f"{'$'*30}\nConfiguration error: primary_ratio + carbon_ratio must equal 1.0\n{'$'*30}")
			self.rl['carbon_ratio'] = 1 - self.rl['primary_ratio']

		if self.client_select_param["learning_exploration"] in ['EO', 'EW'] and self.client_select_param["criteria"] == "NA":
			self.client_select_param["criteria"] = "acc"

		if self.client_select_param["criteria"] == "acc":
			self.client_eval_with_iid = True

		self.pre_distr = self.args.pre_distr or self.pre_distr
		self.data_distribution_type = self.args.distr_type or self.data_distribution_type
		if self.data_distribution_type in ['IID', 'IID_rl']:
			self.pre_distr = 'IID'

		self.cont_tr_iter = self.args.cont_tr_iter or self.cont_tr_iter
		self.cont_tr_acc = self.args.cont_tr_acc or self.cont_tr_acc

		self.label_extension = self.args.label_f or self.label_extension
		self.num_label_ELS = self.args.num_label_ELS or self.num_label_ELS
		self.label_ex = self.args.label_ex or self.label_ex

		self.h_param['learning_rate'] = 0.001 if self.pre_distr not in ['central', 'IID', '1d'] else 0.01

		self.environ['SBATCH'] = self.args.SBATCH or self.environ['SBATCH']
		print(f"SBATCH environment: {self.environ['SBATCH']}")

		self.num_clients = self.args.num_clients or self.num_clients
		self.num_g_iter = self.args.num_g_iter or self.num_g_iter

		self.l_iter_group = [self.args.l_iter_group] if self.args.l_iter_group else None
		self.numEpoch = self.num_g_iter if self.data_distribution_type == 'central' else (self.l_iter_group[0] if self.l_iter_group else self.numEpoch)

		self.global_model = [f"global{i}" for i in range(self.num_g_iter)]
		for index in range(self.num_clients):
			self.selected_client_group[index] = 1 / self.num_clients

		dataset_config = {
			'cifar10': 10,
			'cinic10': 10,
			'cinic10_big': 10,
			'cifar100': 100 if not self.label_extension else self.num_label_ELS + self.label_ex
		}
		num_labels = dataset_config.get(self.data_name)
		if num_labels is None:
			print(f"args.data({self.data_name}) is not supported")
			exit()

		self.num_of_label = num_labels
		self.num_label_ELS = num_labels

		self.h_param.update({
			"batch_size": 64,
			"learning_rate": 0.01,
			"momentum": 0.9,
			"weight_decay": 1e-4
		})

		if self.avg_algo == "Moon" and self.nn["name"] == "resnet18":
			self.nn["name"] = "resnet18_moon"

		self.dirichlet_min = self.args.dirich_min
		self.di_gen_flag = self.args.di_gen_flag

		self.global_info.update({
			"g_acc": [0] * self.num_g_iter,
			"global_distribution": np.full(10, 0.1),
			"total_local_train_time": [0] * self.num_g_iter,
			"local_carbon_total": [0] * self.num_g_iter,
			"local_kl_avg": [0] * self.num_g_iter,
			"entropy_ratio": [0] * self.num_g_iter,
			"client_sel_n": [0] * self.num_g_iter
		})