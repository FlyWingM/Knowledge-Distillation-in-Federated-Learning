### cinic10.py
import os
import requests
import re
import random
import numpy as np
from zipfile import ZipFile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset, WeightedRandomSampler, Dataset
from collections import defaultdict
from scipy.stats import wasserstein_distance
from PIL import Image
from collections import Counter

from datasets.dataset_statistic import DataStatistic

# Custom IndexedSubset class that returns indices along with data and targets
class IndexedSubset(Subset):
	def __init__(self, dataset, indices):
		super().__init__(dataset, indices)
		self.indices = indices

	def __getitem__(self, idx):
		data, target = self.dataset[self.indices[idx]]
		return data, target, self.indices[idx]  # Return index

class DataMonitor:
	def __init__(self):
		self.data = {}

	def __str__(self):
		display_str = "\n\nDataMonitor Contents:\n"
		for key, value in self.data.items():
			display_str += f"Key: {key},\n Value: {value}\n\n"
		return display_str

class FederatedImageFolder(Dataset):
	def __init__(self, root_dir, label_set, transform=None):
		"""
		root_dir: Directory with all the images and subdirectories.
		label_set: Dictionary with class names as keys and corresponding numeric labels as values.
		transform: Optional transform to be applied on a sample.
		"""
		self.root_dir = root_dir
		self.label_set = label_set
		self.transform = transform
		self.classes = list(label_set.keys())  # Maintain a list of class names
		self.class_to_idx = label_set  # Map from class name to index
		self.samples = []
		self._load_samples()

	def _load_samples(self):
		for class_name, label in self.label_set.items():            # {"airplane": 0}
			class_dir = os.path.join(self.root_dir, class_name)
			if os.path.isdir(class_dir):
				for img_name in os.listdir(class_dir):
					img_path = os.path.join(class_dir, img_name)
					if os.path.isfile(img_path):
						self.samples.append((img_path, label))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		img_path, label = self.samples[idx]
		image = Image.open(img_path).convert('RGB')

		if self.transform:
			image = self.transform(image)

		return image, label

class CINIC10Loader:
	def __init__(self, g_para):
		self.g_para     = g_para
		self.dataset_dir = g_para.path[g_para.data_name]
		self.cinic10_url = "https://datashare.ed.ac.uk/download/DS_10283_3192.zip"
		self.zip_path = os.path.join(self.dataset_dir, "cinic-10.zip")
		self.label_set = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6 , "horse": 7, "ship": 8, "truck": 9}
		self.Debug_1, self.Debug_2 = False, True

	def download_cinic10(self):
		"""
		Download the CINIC-10 dataset.
		"""
		try:
			response = requests.get(self.cinic10_url, stream=True)
			response.raise_for_status()
			with open(self.zip_path, 'wb') as file:
				for chunk in response.iter_content(chunk_size=128):
					file.write(chunk)
			print(f"Downloaded CINIC-10 dataset to {self.zip_path}")
			return True
		except requests.RequestException as e:
			print(f"Failed to download the dataset. Error: {e}")
			return False

	def extract_zip(self):
		"""
		Extract the CINIC-10 zip file.
		"""
		with ZipFile(self.zip_path, 'r') as zip_ref:
			zip_ref.extractall(self.dataset_dir)
			print(f"Extracted CINIC-10 dataset to {self.dataset_dir}")

	def prepare_dataset(self):
		"""
		Prepare the CINIC-10 dataset by downloading and extracting it.
		"""
		if not os.path.exists(self.dataset_dir):
			os.makedirs(self.dataset_dir)

		if not os.path.exists(self.zip_path):
			if self.download_cinic10():
				self.extract_zip()

	#@staticmethod
	def create_transforms(self, image_size=32):
		"""
		Create data transformations.
		"""
		if self.g_para.g_iter_index == 0 and self.Debug_1: print(f"Loading the {image_size} data_training_type {self.g_para.data_name} begins")
		mean, std = [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]
		base_transforms = [
			transforms.RandomCrop(image_size, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(), 
			transforms.Normalize(mean, std)
			#transforms.RandomRotation(10),
			#transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
		]
		if image_size == 32:
			return transforms.Compose(base_transforms)
		elif image_size == 224:
			return transforms.Compose([transforms.Resize(224, 224)] + base_transforms)
		else:
			raise ValueError("Unsupported image size for transformations")

	# To see whether the directory exists
	def directory_exists(self, directory):
		return os.path.exists(directory) and os.path.isdir(directory)

	# To see whether the directory has images
	def has_images(self, directory):
		for root, dirs, files in os.walk(directory):
			for file in files:
				if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
					return True
		return False

	def emd_distance(self, P, Q):
		# Ensure P and Q are proper probability distributions
		P = np.array(P, dtype=np.float64)
		Q = np.array(Q, dtype=np.float64)
		P = P / np.sum(P)
		Q = Q / np.sum(Q)

		# Calculate the Earth Mover's Distance
		return round(wasserstein_distance(P, Q), 4)

	def kl_divergence(self, P, Q):
		P = np.array(P, dtype=np.float64)
		Q = np.array(Q, dtype=np.float64)
		P = P / np.sum(P)  # Ensure P is a proper probability distribution
		Q = Q / np.sum(Q)  # Ensure Q is a proper probability distribution

		epsilon_t = 1e-10
		Q = np.maximum(Q, epsilon_t)

		# Calculate the KL divergence, avoid division by zero or log(0)
		return round(np.sum(np.where(P != 0, P * np.log(P / Q), 0)), 4)

	def improve_kl_divergence_by_removal(self, indices_per_class):
		total_instances = sum(len(indices) for indices in indices_per_class.values())
		expected_count_per_class = total_instances / len(indices_per_class)
		
		# Calculate initial probabilities
		initial_prob = [len(indices_per_class[i]) / total_instances for i in range(len(indices_per_class))]
		initial_expected_counts = [len(indices_per_class[i]) for i in range(len(indices_per_class))]

		# IID distribution (uniform)
		iid_prob = [1 / len(indices_per_class)] * len(indices_per_class)
		
		match = re.search(r'KL(\d)', self.g_para.dataset["data_aug"])
		if match and match.group(1) == '0':
			upper_range = 10**6
		elif match:
			upper_range = int(match.group(1))
		else:
			upper_range = 2

		# Adjust the counts for each class by removing excess instances
		for class_index in indices_per_class:
			if len(indices_per_class[class_index]) > expected_count_per_class*upper_range:
				excess_count = len(indices_per_class[class_index]) - expected_count_per_class*upper_range
				#excess_count = len(indices_per_class[class_index])*upper_range     #The delete portion decrease
				indices_per_class[class_index] = indices_per_class[class_index][:-int(excess_count)]

		# Calculate adjusted probabilities
		adjusted_total_instances = sum(len(indices) for indices in indices_per_class.values())
		adjusted_prob = [len(indices_per_class[i]) / adjusted_total_instances for i in range(len(indices_per_class))]
		adjusted_expected_counts = [len(indices_per_class[i]) for i in range(len(indices_per_class))]

		# Calculate KL divergence
		initial_kl = self.kl_divergence(initial_prob, iid_prob)
		adjusted_kl = self.kl_divergence(adjusted_prob, iid_prob)


		print(f"\nexpected_count_per_class: {expected_count_per_class} from total {total_instances}\n", 
			  f"Initial KL divergence : {initial_kl} from {initial_expected_counts}\n",
			  f"Adjusted KL divergence: {adjusted_kl} from {adjusted_expected_counts}")
		
		return indices_per_class


	# Loading datasets for the clients
	def load_client_data(self, server, client, d_monitor, image_size, data_aug_t):
		self.prepare_dataset()
		transformations = self.create_transforms(image_size=image_size)
		dataloaders = {}
		data_directory = []
		full_size = True # True  False
		data_statistic = DataStatistic()
		data_directory = self.g_para.data_dir[:2] if self.g_para.dataset['iid_data'] > 0 and len(self.g_para.data_dir) >= 3 else self.g_para.data_dir[:1]

		for data_training_type in ['train', 'valid', 'test']:            
			datasets_list = []            

			for directory in reversed(data_directory):
				image_datasets = {}
				full_dataset = None
				dir_path = os.path.join(directory, data_training_type)
				client_name = os.path.basename(directory.rstrip('/'))
				if self.Debug_1: print(f"\n{dir_path} is a directory to load images, driven by {directory} for client/shared({client_name})")

				if not os.path.exists(dir_path) or not self.has_images(dir_path):
					print(f"{dir_path} does not exist or includes no images")
					continue

				#full_dataset = datasets.ImageFolder(dir_path, transform=transformations)
				full_dataset = FederatedImageFolder(dir_path, label_set=self.label_set, transform=transformations)
				indices_per_class = defaultdict(list)

				# Initialize indices_per_class with all possible labels, ensuring each label is represented
				indices_per_class = {label: [] for label in self.label_set.values()}    # label_set {"airplane": 0}, indices_per_class {0:[]}

				# Only load and remap indices for classes present in the dataset
				for idx, (_, label_index) in enumerate(full_dataset):
					class_name = full_dataset.classes[label_index]                  # class_name indicates "airplane"

					# Check if the class_name is in the predefined global label set
					if class_name in self.label_set:
						global_label_index = self.label_set[class_name]
						indices_per_class[global_label_index].append(idx)           # 0:[1, 100,], 1:[idx], ..., 0:[idx]
					else:
						print(f"Class {class_name} is not in the global label set.")

				# Improve KL divergence by adjusting the local data distribution
				if self.g_para.dataset["data_aug"] not in ['DAF', 'DAE']:
					indices_per_class = self.improve_kl_divergence_by_removal(indices_per_class)

				# It is extended to the more shared datasets.
				datasets_per_class = []
				entries_per_class_indices = []
				amount_per_label = {}                    
				if client_name == self.g_para.shared_dir and self.g_para.dataset['iid_data'] > 0:   # shared_data, shared_data_synthe
					selected_indices = []
					
					for label_index, class_indices in indices_per_class.items():
						num_images = max(1, min(self.g_para.dataset['iid_data'], 20))  # Assuming the max is 20
						selected_indices.extend(class_indices[:num_images])
						amount_per_label[label_index] = len(class_indices[:num_images])

					image_datasets[data_training_type] = Subset(full_dataset, selected_indices)
					#print(f"selected_indices({len(selected_indices)}):{selected_indices}")
					#print(f"amount_per_label:{amount_per_label}")

				else:                    
					expected_num_entries = sum(len(indices) for indices in indices_per_class.values())/len(indices_per_class)
					for label_index, class_indices in indices_per_class.items():            # indices_per_class {0:[]}, e.g., 0:[1, 100,], 1:[idx], ..., 9:[idx]
						if not full_size:
							# Limit the number of images per class if not full_size, with a minimum of 1
							num_images = max(1, min(len(class_indices), 10))  # Assuming you want up to 5 images if available
							entries_per_class_indices.extend(class_indices[:num_images])
							amount_per_label[label_index] = len(class_indices[:num_images])
						else:
							# If full_size is true, take all available images from the class
							entries_per_class_indices.extend(class_indices)
							amount_per_label[label_index] = len(class_indices)

							if self.g_para.dataset["data_aug"] not in ['DAF', 'DAE'] and 0 < len(class_indices) < expected_num_entries:
								#print(f"max number to lenght it by {expected_num_entries}")
								augmented_data = data_aug_t.data_augmentation(full_dataset, class_indices, expected_num_entries)
								datasets_per_class.extend(augmented_data)
								#data_aug_t.calculate_display_entries_per_label(datasets_per_class)

					if self.g_para.dataset["data_aug"] not in ['DAF', 'DAE'] and datasets_per_class:
						if self.Debug_1: print(f"\nAfter Data augmentation ----")
						image_datasets[data_training_type] =  ConcatDataset([Subset(full_dataset, entries_per_class_indices), datasets_per_class])
						data_aug_t.calculate_display_entries_per_label(image_datasets[data_training_type], self.Debug_1)
					else:
						image_datasets[data_training_type] = Subset(full_dataset, entries_per_class_indices)
						if self.Debug_1: print(f"\nNo Data Augmentation - Loaded {len(image_datasets[data_training_type])} images for {data_training_type}")
						data_aug_t.calculate_display_entries_per_label(image_datasets[data_training_type], self.Debug_1)

				if self.Debug_1: print(f"data_training_type({data_training_type}), image_datasets ({len(image_datasets[data_training_type])})")
				
				# Calcuate the value of KL and EMD
				data_statistic.kl_calculation(self.g_para, data_training_type, server, client, indices_per_class, amount_per_label, len(image_datasets[data_training_type]))

				datasets_list.append(image_datasets[data_training_type])
				if self.Debug_1: print(f"datasets_list({len(datasets_list)})")

			# Not for the combined dataset with the shared dataset yet.

			if len(data_directory) > 1:
				combined_dataset = ConcatDataset(datasets_list)
				shared_t = f"With"	
			else:
				combined_dataset = image_datasets[data_training_type]
				shared_t = f"Without"	

			if self.g_para.g_iter_index==0: # and self.g_para.Debug["data"]:
				if self.Debug_1: print(f"{shared_t} the shared dataset, the final dataset are, as follows----")
				data_aug_t.calculate_display_entries_per_label(combined_dataset, self.Debug_1)
			
			# ADJUSTMENT FOR OVERSAMPLING WHEN TOO SMALL
			if len(combined_dataset) < self.g_para.h_param['batch_size']:
				# Oversample using a WeightedRandomSampler with uniform weights:
				weights = [1.0] * len(combined_dataset)
				sampler = WeightedRandomSampler(
					weights, 
					num_samples=self.g_para.h_param['batch_size'],
					replacement=True
				)
				dataloaders[data_training_type] = DataLoader(
					combined_dataset,
					batch_size=self.g_para.h_param['batch_size'],
					sampler=sampler,
					drop_last=False,
					num_workers=2,
					pin_memory=True
				)
			else:
				dataloaders[data_training_type] = DataLoader(
					combined_dataset, 
					batch_size=self.g_para.h_param['batch_size'], 
					shuffle=True, 
					drop_last=False, 
					num_workers=2, 
					pin_memory=True
				)

			dataloaders["train_sample_size"] = len(combined_dataset)

			if self.Debug_1:
				print(f"M[{client.id}], the {data_training_type} dataloaders[batch:{len(dataloaders[data_training_type])}, sampel:{dataloaders['train_sample_size']}]")
			# End of for directory in reversed(data_directory):

		#End of for data_training_type in ['train', 'valid', 'test']:
		return dataloaders


	def load_data_global(self, d_monitor, image_size):
		"""
		Load the CINIC-10 data for global testing.
		"""
		self.prepare_dataset()
		transformations = self.create_transforms(image_size=image_size)
		dataloaders = {}

		# Determine the test data directory based on the parameters
		test_data_directory = self.g_para.data_dir[2:] if self.g_para.dataset['iid_data'] > 0 and len(self.g_para.data_dir) >= 3 else self.g_para.data_dir[1:]
		if not test_data_directory:
			raise ValueError(f"load_data_global: test_data_directory is empty as {test_data_directory}")
		else:
			print(f"The global test-dataset is being loaded from {test_data_directory}")

		dir_path = os.path.join(test_data_directory[0], 'test')			
		if not os.path.exists(dir_path) or not self.has_images(dir_path):
			print(f"{dir_path} does not exit or include no images")

		# Load dataset
		full_dataset = datasets.ImageFolder(dir_path, transform=transformations)

		# Shuffle the dataset indices
		dataset_indices = list(range(len(full_dataset)))
		random.shuffle(dataset_indices)

		indices_per_class = defaultdict(list)

		# Gather indices for each class
		for idx in dataset_indices:
			_, label_index = full_dataset[idx]
			indices_per_class[label_index].append(idx)

		# Define dataset portions for different labels
		dataset_labels = ['test_global', 'test_global_small', 'proxy_data_client', 'global_calibration']
		portions = [0.7, 0.15, 0.1, 0.05]
		assert sum(portions) <= 1.0

		# Keep a reference to the original indices for consistent portion calculation
		original_indices_per_class = {label_index: list(class_indices) for label_index, class_indices in indices_per_class.items()}		

		for current_label, portion in zip(dataset_labels, portions):
			selected_indices = []
			amount_per_label = {}

			for label_index, original_class_indices in original_indices_per_class.items():
				# Calculate portion size based on the original size of each class
				portion_size = int(len(original_class_indices) * portion)

				# Select the portioned indices and update the amount for this label
				selected_indices.extend(indices_per_class[label_index][:portion_size])
				amount_per_label[label_index] = min(portion_size, len(indices_per_class[label_index]))

				# Update indices_per_class by removing the selected portion
				indices_per_class[label_index] = indices_per_class[label_index][portion_size:]

			batch_size = 128 if current_label == 'test_global' else 100

			# Create and store the dataset and dataloader 
			if current_label == 'proxy_data_client':	# For the label soft for the knowledge distillation by using "IndexedSubset"
				image_dataset = IndexedSubset(full_dataset, selected_indices)
				dataloaders[current_label] = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=16, pin_memory=True)
			else:
				image_dataset = Subset(full_dataset, selected_indices)
				dataloaders[current_label] = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=16, pin_memory=True)

			# Store dataset information
			if self.g_para.Debug["data"]:
				d_monitor.data[current_label] = {
					"dataset_sizes": len(image_dataset),
					"class_names": tuple(full_dataset.classes),
					"num_entries_per_class": amount_per_label
				}

		if self.g_para.g_iter_index == 0 and self.g_para.Debug["data"]: print("{}".format(d_monitor))

		return dataloaders


	def load_data_central(self, d_monitor, image_size):
		self.prepare_dataset()
		transformations = self.create_transforms(image_size=image_size)

		data_directory = self.g_para.data_dir[0]
		dataloaders = {}
		
		for data_training_type in ['train', 'valid', 'test']:
			dir_path = os.path.join(data_directory, data_training_type)
			client_name = os.path.basename(data_directory.rstrip('/'))
			if not os.path.exists(dir_path) or not self.has_images(dir_path):
				print(f"{dir_path} does not exit or include no images")

			full_dataset = datasets.ImageFolder(dir_path, transform=transformations)
			
			# Counting images per class
			if self.g_para.Debug["data"]:
				print(f"{data_training_type} datasets are analized")
				indices_per_class = defaultdict(list)
				for idx, (_, label_index) in enumerate(full_dataset):
					indices_per_class[label_index].append(idx)

				# Extract and save the key values from the datasets
				dataset_sizes = len(full_dataset)
				class_names = set(label for _, label in full_dataset)

				# Count the number of entries per label
				label_counts = Counter(label for _, label in full_dataset)

				d_monitor.data[client_name+'-'+data_training_type] = {
					"dataset_sizes": dataset_sizes,
					"class_names": tuple(class_names),
					"label_counts": dict(label_counts)  # Convert Counter to a dictionary                    
				}

			dataloaders[data_training_type] = DataLoader(full_dataset, batch_size=self.g_para.h_param['batch_size'], shuffle=True, drop_last=False, num_workers=10, pin_memory=True)

		if self.g_para.g_iter_index==0 and self.g_para.Debug["data"]: print("{}".format(d_monitor))
		return dataloaders


def load_cinic10_client(g_para, server, client, data_aug_t=None):
	cinic_loader = CINIC10Loader(g_para)
	d_monitor     = DataMonitor()
	if g_para.data_name=="cinic10":
		return cinic_loader.load_client_data(server, client, d_monitor, 32, data_aug_t)
	elif g_para.data_name=="cinic10_big":
		return cinic_loader.load_client_data(server, client, d_monitor, 224, data_aug_t)


def load_cinic10_global(g_para):
	cinic_loader = CINIC10Loader(g_para)
	d_monitor     = DataMonitor()
	if g_para.data_name=="cinic10":
		return cinic_loader.load_data_global(d_monitor, 32)
	elif g_para.data_name=="cinic10_big":
		return cinic_loader.load_data_global(d_monitor, 224)


def load_cinic10_central(g_para):
	cinic_loader = CINIC10Loader(g_para)
	d_monitor     = DataMonitor()
	if g_para.data_name=="cinic10":
		return cinic_loader.load_data_central(d_monitor, 32)
	elif g_para.data_name=="cinic10_big":
		return cinic_loader.load_data_central(d_monitor, 224)


