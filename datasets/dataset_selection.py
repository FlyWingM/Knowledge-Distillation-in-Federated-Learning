### dataset_selection.py

from datasets.cifar10 import load_cifar10, load_cifar10_global, load_cifar10_central
#from datasets.cifar100 import load_cifar100, load_cifar100_global, load_cifar100_central
from datasets.cinic10 import load_cinic10_client, load_cinic10_global, load_cinic10_central

class DatasetSelection:

	def __init__(self, g_para, data_aug):
		# Initialize data loaders
		self.g_para = g_para
		self.data_aug_t = data_aug
		self.Debug_1 = False
		self.shared_local_datasets = {}

	def partition_data_global(self, dataloader, server):
		required_keys = ['test_global', 'test_global_small', 'proxy_data_client', 'global_calibration']
		if all(key in dataloader for key in required_keys):
			server.dataloader_global_test = dataloader['test_global']
			server.dataloader_global_test_small = dataloader['test_global_small']
			server.dataloader_global_calibration = dataloader['global_calibration']
			server.dataloader_proxy_data_client = dataloader['proxy_data_client']
		else:
			missing = [key for key in required_keys if key not in dataloader]
			print(f"Global dataloader does not have the expected keys: {missing}")

	def partition_data_central(self, dataloader, c_server):
		if all(key in dataloader for key in ('train', 'valid', 'test')):
			c_server.dataloader_central_train = dataloader['train']
			c_server.dataloader_central_valid = dataloader['valid']
			c_server.dataloader_central_test = dataloader['test']
		else:
			missing = [key for key in ('train', 'valid', 'test') if key not in dataloader]
			print(f"Central dataloader is missing expected keys: {missing}")

	def load_client_dataset(self, client, server):
		# Define dataset loaders based on the global parameter settings
		dataset_local_loaders = {
			"cifar10":      lambda: load_cifar10(self.g_para),
			"cifar10_big":  lambda: load_cifar10(self.g_para),
			# "cifar100":     lambda: load_cifar100(self.g_para),
			"cinic10":      lambda: load_cinic10_client(self.g_para, server, client, self.data_aug_t),
			"cinic10_big":  lambda: load_cinic10_client(self.g_para, server, client, self.data_aug_t)
		}

		# Load and partition the dataset for the client
		load_local_func = dataset_local_loaders.get(self.g_para.data_name)
		if load_local_func is None:
			raise ValueError(f"Dataset {self.g_para.data_name} is not supported.")

		try:
			dataloader = load_local_func()
		except Exception as e:
			print(f"Failed to load dataset for client {client.id}. Error: {e}")
			return

		required_keys = ['train', 'valid', 'test', 'train_sample_size']
		if not all(key in dataloader for key in required_keys):
			missing = [key for key in required_keys if key not in dataloader]
			print(f"Dataloader for client {client.id} is missing expected keys: {missing}")
			return

		client.dataloaders_train = dataloader['train']
		client.dataloaders_valid = dataloader['valid']
		client.dataloaders_test = dataloader['test']

		server.server_local_info[client.id]['train_num_samples'] = dataloader['train_sample_size']

		if self.Debug_1:
			print(f"Successfully loaded datasets for client {client.id}:")
			print(f"  - Train set: {len(dataloader['train'])} batchs")
			print(f"  - Validation set: {len(dataloader['valid'])} batchs")
			print(f"  - Test set: {len(dataloader['test'])} batchs")
			print(f"  - Train sample size: {dataloader['train_sample_size']}")
			print(f"  - server.server_local_info[{client.id}]['train_num_samples']:{server.server_local_info[client.id]['train_num_samples']}")
			print("\n".join([f"server.server_local_info[{key}]['train_num_samples']: {server.server_local_info[key].get('train_num_samples', 'Not Found')}" for key in server.server_local_info]))

	def load_client_dataset_for_globally_sharing(self, client, server):
		# Define dataset loaders based on the global parameter settings
		dataset_local_loaders = {
			"cifar10":      lambda: load_cifar10(self.g_para),
			"cifar10_big":  lambda: load_cifar10(self.g_para),
			# "cifar100":     lambda: load_cifar100(self.g_para),
			"cinic10":      lambda: load_cinic10_client(self.g_para, server, client, self.data_aug_t),
			"cinic10_big":  lambda: load_cinic10_client(self.g_para, server, client, self.data_aug_t)
		}

		# Load and partition the dataset for the client
		load_local_func = dataset_local_loaders.get(self.g_para.data_name)
		if load_local_func is None:
			raise ValueError(f"Dataset {self.g_para.data_name} is not supported.")

		try:
			dataloader = load_local_func()
		except Exception as e:
			print(f"Failed to load dataset for client {client.id}. Error: {e}")
			return

		required_keys = ['train', 'valid', 'test', 'train_sample_size']
		if not all(key in dataloader for key in required_keys):
			missing = [key for key in required_keys if key not in dataloader]
			print(f"Dataloader for client {client.id} is missing expected keys: {missing}")
			return

		# Initialize shared_local_datasets for the client if not already initialized
		if client.id not in self.shared_local_datasets:
			self.shared_local_datasets[client.id] = {
				"dataloaders_train": None,
				"dataloaders_valid": None,
				"dataloaders_test": None
			}

		# Assign dataloaders
		self.shared_local_datasets[client.id]["dataloaders_train"] = dataloader['train']
		self.shared_local_datasets[client.id]["dataloaders_valid"] = dataloader['valid']
		self.shared_local_datasets[client.id]["dataloaders_test"] = dataloader['test']

		# Update server info
		server.server_local_info[client.id]['train_num_samples'] = dataloader['train_sample_size']

		if self.Debug_1:
			print(f"Successfully loaded datasets for client {client.id}:")
			print(f"  - Train set: {len(dataloader['train'])} samples")
			print(f"  - Validation set: {len(dataloader['valid'])} samples")
			print(f"  - Test set: {len(dataloader['test'])} samples")
			print(f"  - Train sample size: {dataloader['train_sample_size']}")
			print(f"  - server.server_local_info[{client.id}]['train_num_samples']:{server.server_local_info[client.id]['train_num_samples']}")
			print("\n".join([f"server.server_local_info[{key}]['train_num_samples']: {server.server_local_info[key].get('train_num_samples', 'Not Found')}" for key in server.server_local_info]))


	def load_dataset_global(self, server):
		# Check if global dataset is already loaded
		if server.dataloader_global_test is not None and server.dataloader_global_test_small is not None:
			return

		dataset_global_loaders = {
			"cifar10":      lambda: load_cifar10_global(self.g_para),
			"cifar10_big":  lambda: load_cifar10_global(self.g_para),
			#"cifar100":     lambda: load_cifar100_global(self.g_para),
			"cinic10":      lambda: load_cinic10_global(self.g_para),
			"cinic10_big":  lambda: load_cinic10_global(self.g_para),
		}
		load_global_func = dataset_global_loaders.get(self.g_para.data_name)
		if load_global_func is None:
			raise ValueError(f"Dataset {self.g_para.data_name} is not supported.")

		dataloader = load_global_func()
		self.partition_data_global(dataloader, server)


	def load_dataset_central(self, c_server):
		dataset_central_loaders = {
			"cifar10":      lambda: load_cifar10_central(self.g_para),
			"cifar10_big":  lambda: load_cifar10_central(self.g_para),
			#"cifar100":     lambda: load_cifar100_central(self.g_para),
			"cinic10":      lambda: load_cinic10_central(self.g_para),
			"cinic10_big":  lambda: load_cinic10_central(self.g_para),
		}
		load_central_func = dataset_central_loaders.get(self.g_para.data_name)
		if load_central_func is None:
			raise ValueError(f"Dataset {self.g_para.data_name} is not supported.")
		dataloader = load_central_func()
		self.partition_data_central(dataloader, c_server)
