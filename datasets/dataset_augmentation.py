
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from collections import defaultdict

class CustomDataset(Dataset):
	def __init__(self, data_list):
		self.data_list = data_list

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		return self.data_list[idx]

class DataAugmentation:

	def check_image_type(self, image):
		if isinstance(image, torch.Tensor):
			print("This is a Torch Tensor.")
		elif isinstance(image, Image.Image):
			print("This is a PIL image.")
		else:
			print("Unknown image type.")

	def calculate_display_entries_per_label(self, dataloader, Debug=False):
		# Print a few items to check the structure
		for item in dataloader:
			#print(item)
			break  # Print only the first item for structure checking
		#print(f"Type of dataloader in calculate_display_entries_per_label: {type(dataloader)}")

		label_counts = defaultdict(int)
		for item in dataloader:
			# Unpack images and labels
			images, labels = item
			
			if isinstance(labels, torch.Tensor):
				# If it's a single-element tensor, convert to integer
				if labels.dim() == 0:  # A single label
					labels = [labels.item()]
				else:  # A batch of labels
					labels = labels.tolist()

			# Ensure labels is always a list at this point
			if isinstance(labels, int):
				labels = [labels]

			for label in labels:
				label_counts[label] += 1

		info = []
		for label in sorted(label_counts.keys()):
			count = label_counts[label]
			info.append(f"({label}:{count})")
			#print(f"Label {label}: {count} entries")
		if Debug: print(', '.join(info))

	def data_augmentation_client(self, g_para, client_dataset, client_id):
		augmented_datasets = {}
		label_set = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6 , "horse": 7, "ship": 8, "truck": 9}

		# Initialize indices_per_class with all possible labels, ensuring each label is represented
		indices_per_class = {i: [] for i in label_set.values()}  # label_set {"airplane": 0}, indices_per_class {0:[]}
		client_dataset_list = []

		for images, labels in client_dataset:
			batch_tuples = list(zip(images, labels))
			client_dataset_list.extend(batch_tuples)

		if client_id in [0, 1]: 
			print(f"\nClient-{client_id} has the following original dataset")
			self.calculate_display_entries_per_label(DataLoader(CustomDataset(client_dataset_list), batch_size=g_para.h_param['batch_size'], shuffle=True))

		# Only load and remap indices for classes present in the dataset
		for idx, (_, label_index) in enumerate(client_dataset_list):
			label_index = label_index.item()
			if label_index in indices_per_class:
				indices_per_class[label_index].append(idx)
		#print(f"indices_per_class:\n{indices_per_class}")
		
		datasets_per_class = []
		expected_num_entries = sum(len(indices) for indices in indices_per_class.values())/len(indices_per_class)
		#print(f"expected_num_entries:({expected_num_entries}), len(indices_per_class):({len(indices_per_class)})")

		for label_index, class_indices in indices_per_class.items():   # indices_per_class {0:[]}, e.g., 0:[1, 100,], 1:[idx], ..., 9:[idx]
			#print(f"label_index:({label_index}) with {len(class_indices)}")
			if len(class_indices) > 0:
				augmented_data = self.data_augmentation(client_dataset_list, class_indices, expected_num_entries)
				#Ensure all labels in augmented_data are tensors
				augmented_data = [(img, lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl)) for img, lbl in augmented_data]
				datasets_per_class.extend(augmented_data)
	
		combined_dataset_list = client_dataset_list + datasets_per_class
		#print(f"combined_dataset_list:({type(combined_dataset_list)}), client_dataset_list:({type(client_dataset_list)}), datasets_per_class:({type(datasets_per_class)})")
		combined_dataset = CustomDataset(combined_dataset_list)

		augmented_dataloader = DataLoader(combined_dataset, batch_size=g_para.h_param['batch_size'], shuffle=True)
		if client_id in [0, 1]:
			print(f"Augmented by the following dataset")
			self.calculate_display_entries_per_label(augmented_dataloader)

		return augmented_dataloader


	def data_augmentation(self, full_dataset, class_indices, expected_num_entries):
		datasets_per_class = []
		max_times = 4
		length_class_indices = len(class_indices)
		if length_class_indices < expected_num_entries*0.5:
			size_portion = int(expected_num_entries) // length_class_indices
		else:
			size_portion = 0

		selected_image = Subset(full_dataset, class_indices)

		augment = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.1),
			transforms.ColorJitter(contrast=(0.9, 1.1)),
			transforms.ColorJitter(saturation=(0.8, 1.2)),
			transforms.ColorJitter(hue=0.08),
			transforms.RandomResizedCrop(32, scale=(0.7, 0.7)),
			transforms.Resize((32, 32)),
			transforms.ToTensor()  # Convert back to tensor
		])

		selected_iteration =  min(max_times, size_portion)
		label_t = 0
		for i in range(selected_iteration):
			for batch_images, batch_labels in DataLoader(selected_image, batch_size=4, shuffle=False):  # Set batch_size > 1
				# Process each image in the batch
				for img, lbl in zip(batch_images, batch_labels):
					img = img.squeeze(0)  # Remove batch dimension from individual images
					if i == 0: 
						#self.check_image_type(img)
						pass
					augmented_image = augment(img)
					datasets_per_class.append((augmented_image, lbl))
					label_t = lbl.item()

		#print(f"selected_iteration:{selected_iteration}={expected_num_entries}//{length_class_indices} by label:{label_t}, len of datasets_per_class({len(datasets_per_class)})")
		return datasets_per_class