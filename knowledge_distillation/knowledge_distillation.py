### knowledge_distillation.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

# Assuming you have the IndexedSubset class defined as follows
class IndexedSubset(Subset):
	def __init__(self, dataset, indices):
		super().__init__(dataset, indices)
		self.indices = indices

	def __getitem__(self, idx):
		data, target = self.dataset[self.indices[idx]]
		return data, target, self.indices[idx]  # Return index

class KnowledgeDistillation:

	def __init__(self, fl_params):
		self.fl_params = fl_params
		self.g_para = fl_params.g_para
		self.server = fl_params.server
		self.clients = fl_params.clients
		
		self.T = 4      # Experiment with higher temperatures
		self.Debug_1 = True

	def display(self):
		print(f"\nknowledge_extraction with the selected ({len(self.g_para.selected_client_group)}) clients")

	def average_soft_labels(self):
		selected_clients = self.g_para.selected_client_group
		aggregated_soft_labels = {}

		for client_id in selected_clients:
			client = self.clients[client_id]
			client_soft_labels = self.server.server_local_info[client.id]['local_soft_label']

			for data_id, soft_label in client_soft_labels.items():
				if data_id not in aggregated_soft_labels:
					aggregated_soft_labels[data_id] = []
				aggregated_soft_labels[data_id].append(soft_label)

		# Average the soft labels for each data_id
		for data_id in aggregated_soft_labels:
			stacked_labels = torch.stack(aggregated_soft_labels[data_id], dim=0)
			aggregated_soft_labels[data_id] = torch.mean(stacked_labels, dim=0)

		return aggregated_soft_labels  # Dictionary with data IDs as keys

	def knowledge_extraction(self):
		if self.Debug_1: self.display()
		selected_clients = self.g_para.selected_client_group

		# Train and evaluate for each selected client
		for client_id in selected_clients:
			client = self.clients[client_id]
			model = client.local_model.to(self.g_para.device)
			client_local_info = self.server.server_local_info[client.id]
			client_soft_labels = {}  # Use a dictionary to store soft labels with indices

			model.eval()
			with torch.no_grad():
				for data_batch in self.server.dataloader_proxy_data_client:
					data, target, indices = data_batch  # Correctly unpacking indices
					data = data.to(self.g_para.device)
					outputs = model(data)
					soft_labels = torch.nn.functional.softmax(outputs / self.T, dim=1)

					for idx, soft_label in zip(indices, soft_labels):
						client_soft_labels[idx.item()] = soft_label.cpu()

			client_local_info['local_soft_label'] = client_soft_labels

		# Optional: Clear cache if on GPU to manage memory better
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def knowledge_aggregation(self):
		# Aggregate soft labels
		aggregated_soft_labels = self.average_soft_labels()

		full_dataset = self.server.dataloader_proxy_data_client.dataset.dataset
		selected_indices = self.server.dataloader_proxy_data_client.dataset.indices

		# Split the selected_indices into training and validation indices
		total_size = len(selected_indices)
		indices = selected_indices.copy()
		np.random.shuffle(indices)

		val_split = 0.2  # 20% for validation
		split_idx = int(np.floor(val_split * total_size))
		train_indices, val_indices = indices[split_idx:], indices[:split_idx]

		# Create training and validation datasets using IndexedSubset
		train_dataset = IndexedSubset(full_dataset, train_indices)
		val_dataset = IndexedSubset(full_dataset, val_indices)

		# Create DataLoaders for training and validation
		batch_size = self.server.dataloader_proxy_data_client.batch_size  # Use the same batch size
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

		# Assign DataLoaders to the server
		self.server.train_loader = train_loader
		self.server.val_loader = val_loader

		# Initialize the model, optimizer, and loss function
		model = self.server.g_model_big.to(self.g_para.device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		criterion = nn.KLDivLoss(reduction='batchmean')
		model.train()

		# Early stopping parameters
		patience = 5  # Number of epochs with no improvement after which training stops
		best_val_loss = np.inf
		epochs_no_improve = 0
		n_epochs = 50  # Maximum number of epochs

		for epoch in range(n_epochs):
			# Training phase
			model.train()
			for data_batch in train_loader:
				data, target, indices = data_batch
				data = data.to(self.g_para.device)

				# Retrieve the corresponding aggregated soft labels
				target_soft_labels = [aggregated_soft_labels[idx.item()] for idx in indices]
				target_soft_labels = torch.stack(target_soft_labels).to(self.g_para.device)

				optimizer.zero_grad()
				outputs = model(data)
				student_logits = outputs / self.T
				student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=1)

				loss = criterion(student_log_probs, target_soft_labels)
				loss.backward()
				optimizer.step()

			# Validation phase
			model.eval()
			val_loss = 0.0
			with torch.no_grad():
				for data_batch in val_loader:
					data, target, indices = data_batch
					data = data.to(self.g_para.device)

					# Retrieve the corresponding aggregated soft labels
					target_soft_labels = [aggregated_soft_labels[idx.item()] for idx in indices]
					target_soft_labels = torch.stack(target_soft_labels).to(self.g_para.device)

					outputs = model(data)
					student_logits = outputs / self.T
					student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=1)

					loss = criterion(student_log_probs, target_soft_labels)
					val_loss += loss.item() * data.size(0)

			val_loss /= len(val_loader.dataset)
			print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {val_loss:.4f}")

			# Early stopping logic
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				epochs_no_improve = 0
				# Save the best model weights
				best_model_wts = model.state_dict()
			else:
				epochs_no_improve += 1
				if epochs_no_improve >= patience:
					print(f"Early stopping triggered at epoch {epoch+1}")
					break

		# Load the best model weights
		model.load_state_dict(best_model_wts)
		#self.server.g_model_big = model

		# Optional: Clear GPU cache after training
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def distillation_run(self):
		
		self.knowledge_extraction()
		self.knowledge_aggregation()

