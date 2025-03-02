### client.py        
import time
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR
from nn_architecture.nn_selection import nn_archiecture_selection
from train_fl.fl_util import get_dir_name, update_data_dir
from train_fl.utility import Utils

class Client:
	def __init__(self, g_para, server, c_index, dataset):
		"""
		The per-client object storing local model, dataset, etc.
		"""
		self.g_para = g_para
		self.server = server
		self.id = c_index
		self.dataset = dataset
		self.utils = Utils()
		self.device = self.g_para.device
		self.Debug_1 = True

		self.local_model = None
		self.local_optimizer = None
		self.local_criterion = None
		self.local_scheduler = None
		self.best_model_state = None

		# The caches:
		self.cached_batches_train = []
		self.cached_batches_valid = []

		# For optional early stopping
		self.best_val_acc = 0.0
		self.no_improvement_count = 0
		self.patience = 1  # or however many epochs to wait

		# Data loaders (client-specific)
		self.dataloaders_train = None
		self.dataloaders_valid = None
		self.dataloaders_test = None

		# 1) Load dataset (uncomment if needed)
		#self.create_local_dataset(self.server)

		# 2) Create local model/optimizer/criterion (uncomment if needed)
		self.create_local_moc()

	def convert_gradients_to_numpy(self, epoch_gradients):
		"""
		Convert PyTorch gradient tensors to a dictionary of NumPy arrays 
		(as an example, we take the mean of the gradients).
		"""
		numpy_gradients = {}
		for name, grad_tensor in epoch_gradients.items():
			if grad_tensor is not None:
				numpy_gradients[name] = grad_tensor.cpu().numpy().mean()
		return numpy_gradients

	def create_local_dataset(self, server):
		"""
		Load the dataset for this client.
		"""
		dir_name = get_dir_name(self.g_para)
		try:
			update_data_dir(self.g_para, dir_name, self.id)
			self.dataset.load_client_dataset(self, server)
			print(f"\nClient-{self.id} loaded the local dataset with "
		 		  f"train[S:{server.server_local_info[self.id]['train_num_samples']}, B:{len(self.dataloaders_train)}], "
		 		  f"valid[B:{len(self.dataloaders_valid)}], "
		 		  f"test[B:{len(self.dataloaders_test)}] by KV[{server.server_local_info[self.id]['kl_div'][0]}]")

			# If needed, also store it in a shared dataset object:
			# self.dataset.load_client_dataset_for_globally_sharing(self, server)
		except Exception as e:
			print(f"Failed to initialize client {self.id}: {e}")

	def create_local_moc(self):
		"""
		Initialize the local model and training-related modules.
		"""
		if self.local_model is not None:
			print(f"Client {self.id}: Model already initialized.")
			return

		print(f"\nClient {self.id}: Creating local model on {self.device}")
		self.local_model = nn_archiecture_selection(self.g_para, self.device)
		self.local_optimizer = torch.optim.SGD(
			self.local_model.parameters(),
			lr=self.g_para.h_param['learning_rate'],
			momentum=self.g_para.h_param['momentum'],
			weight_decay=self.g_para.h_param['weight_decay']
		)
		self.local_criterion = nn.CrossEntropyLoss()
		self.local_scheduler = CosineAnnealingLR(optimizer=self.local_optimizer, T_max=self.g_para.numEpoch, eta_min=0)
		#self.local_scheduler = torch.optim.lr_scheduler.StepLR(
		#	optimizer=self.local_optimizer,
		#	step_size=1,
		#	gamma=0.1
		#)
		#self.local_model.to(self.device)

	def preload_local_model_dataset_onGPU(self, device):

		if self.local_model is None:
			# If the model hasn't been created yet, do so.
			self.create_local_moc()
		if next(self.local_model.parameters()).device != device:
			self.local_model.to(device)

		"""
		Preloads the client's local dataset into GPU memory (or pinned memory).
		This should be called before _train_once().
		"""
		if not self.dataloaders_train:
			print(f"\n{'#'*10} Client {self.id} has not local datasets loaded, it might be an error {'#'*10}\n")
			self.create_local_dataset(self.server)
			#raise RuntimeError(f"Client-{self.id} local dataset not yet created. Call create_local_dataset() first.")

		# Move each batch to GPU (or pinned memory) in advance
		for batch_idx, (data, labels) in enumerate(self.dataloaders_train):
			# Using non_blocking=True + pinned memory can overlap CPU-GPU transfer
			data_gpu = data.to(device, non_blocking=True)
			labels_gpu = labels.to(device, non_blocking=True)
			self.cached_batches_train.append((data_gpu, labels_gpu))

		# Move each batch to GPU (or pinned memory) in advance
		for batch_idx, (data, labels) in enumerate(self.dataloaders_valid):
			data_gpu = data.to(device, non_blocking=True)
			labels_gpu = labels.to(device, non_blocking=True)
			self.cached_batches_valid.append((data_gpu, labels_gpu))
		return self.id

	def clear_local_model_and_cached_batches_onGPU(self):

		if self.local_model is not None:
			self.local_model.to("cpu")  # Move the model to CPU
			torch.cuda.empty_cache()   # Release GPU memory
			#print("Local model moved to CPU and GPU memory freed.")

		"""			
		Clears the space allocated by cached_batches_train and cached_batches_valid
		to free GPU memory.
		"""
		# Clear the training cached batches
		for batch in self.cached_batches_train:
			del batch  # Remove tensor references
		self.cached_batches_train.clear()  # Clear the list

		# Clear the validation cached batches
		for batch in self.cached_batches_valid:
			del batch  # Remove tensor references
		self.cached_batches_valid.clear()  # Clear the list

		# Release GPU memory
		torch.cuda.empty_cache()
		#print("Cached batches cleared and GPU memory released.")

	def check_model_update(self, before_state, after_state):
		"""
		Raises ValueError if no params changed during training.
		"""
		updated_params = []
		for param_name in before_state:
			if not torch.equal(before_state[param_name], after_state[param_name]):
				updated_params.append(param_name)

		if not updated_params:
			raise ValueError(
				f"Client-{self.id}: No model parameters updated during training."
			)

	def train_epoch(self, epoch):
		"""
		A single epoch of training. Returns train_acc, train_loss.
		"""
		if self.g_para.Debug["d_train"]:
			# Debugging placeholder
			pass

		self.local_model.train()
		train_loss, correct, total = 0.0, 0, 0
		num_batches = 0
		epoch_gradients = {}  # Dictionary to store accumulated gradients for the epoch

		for batch_idx, (inputs, targets) in enumerate(self.cached_batches_train):
			# self.cached_batches_train is already stored on GPU.
			self.local_optimizer.zero_grad()
			outputs = self.local_model(inputs)
			loss = self.local_criterion(outputs, targets)
			loss.backward()

			self.local_optimizer.step()

			# Step the scheduler only if it is intended for per-batch updates
			if hasattr(self.local_scheduler, "step"):
				self.local_scheduler.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			num_batches = batch_idx + 1

			# Debug printing for non-Slurm environments
			if not self.g_para.environ.get("SBATCH", False) and self.g_para.Debug["d_train"]:
				self.utils.progress_bar(
					f"#{self.id} Model-{epoch}/{self.g_para.numEpoch}",
					batch_idx,
					len(self.cached_batches_train),
					f'Loss: {train_loss/(batch_idx+1):.3f} '
					f'| Acc: {100.*correct/total:.3f}% ({correct}/{total})'
				)

		for name, param in self.local_model.named_parameters():
			if param.grad is not None:
				if name not in epoch_gradients:
					epoch_gradients[name] = param.grad.detach().clone()
				else:
					epoch_gradients[name] += param.grad.detach()

		# Average the accumulated gradients over the number of batches
		if num_batches > 0:
			for name in epoch_gradients:
				epoch_gradients[name] /= num_batches

		# Convert epoch gradients to numpy
		self.server.server_local_gradients[self.id] = self.convert_gradients_to_numpy(epoch_gradients)

		train_acc = float(correct) / total if total > 0 else 0.0
		avg_loss = train_loss / num_batches if num_batches > 0 else 0.0

		return {
			"train_acc": train_acc,
			"train_loss": avg_loss
		}


	def validate_epoch(self, epoch):
		"""
		Validate for one epoch. Returns val_acc, val_loss.
		"""
		self.local_model.eval()
		valid_loss, correct, total = 0.0, 0, 0
		num_batches = 0

		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(self.cached_batches_valid):
				# self.cached_batches_valid is suppose to be already stored on GPU.
				outputs = self.local_model(inputs)
				loss = self.local_criterion(outputs, targets)

				valid_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()
				num_batches = batch_idx + 1

		val_acc = float(correct) / total if total > 0 else 0.0
		avg_loss = valid_loss / num_batches if num_batches > 0 else 0.0

		return {
			"val_acc": val_acc,
			"val_loss": avg_loss
		}


	def should_early_stop(self, val_acc):
		"""
		Checks if validation accuracy has not improved for 'patience' epochs.
		Returns True if we should stop early.
		"""
		if val_acc >= self.best_val_acc:
			self.best_val_acc = val_acc
			self.no_improvement_count = 0
		else:
			self.no_improvement_count += 1
			if self.no_improvement_count >= self.patience:
				return True
		return False


	def train_model(self):
		"""
		A typical multi-epoch training loop.
		"""
		history = []
		initial_state = {k: v.clone() for k, v in self.local_model.state_dict().items()}
		local_model_best = None
		self.best_val_acc = 0.0

		for epoch in range(self.g_para.numEpoch):
			train_result = self.train_epoch(epoch)
			val_result = self.validate_epoch(epoch)

			# Combine train/val metrics
			result = {**train_result, **val_result}
			history.append(result)
			#if result['val_acc'] > self.best_val_acc and self.g_para.numEpoch > 1:
			#	local_model_best = deepcopy(self.local_model)

			if self.Debug_1 and self.g_para.environ.get("SBATCH", False):
				id_t = int(self.id)
				id_t_str = f"{id_t:>2}"  
				train_num_samples_t = self.server.server_local_info[self.id]['train_num_samples']
				train_num_samples_t_str = f"{train_num_samples_t:>4}" 
				dataloaders_train_length = len(self.cached_batches_train)		# len(self.dataloaders_train)
				dataloaders_train_length_str = f"{dataloaders_train_length:>3}"
				if self.Debug_1: 	#if dataloaders_train_length == 1:
					print(
						f"#{id_t_str}, S[{train_num_samples_t_str}], B[{dataloaders_train_length_str}], E {epoch+1}/{self.g_para.numEpoch}, "
						f"TLoss: {result['train_loss']:.4f}, "
						f"TAcc: {result['train_acc']:.4f}, "
						f"VLoss: {result['val_loss']:.4f}, "
						f"VAcc: {result['val_acc']:.4f}"
					)

			# Possibly do early stopping, logging, etc.
			'''
			if self.should_early_stop(result["val_acc"]):
				if local_model_best is not None:
					self.local_model.load_state_dict(local_model_best.state_dict())
					del local_model_best
					torch.cuda.empty_cache()   # Release GPU memory explicitly
					if self.Debug_1: print(f"[Client-{self.id}] Early stop triggered at epoch {epoch+1}")
				break
			'''

		# Check if anything changed
		final_state = {k: v.clone() for k, v in self.local_model.state_dict().items()}
		
		self.check_model_update(initial_state, final_state)
		return history

	def _train_once(self, global_state=None):
		if global_state is not None:
			self.local_model.load_state_dict(global_state)
			self.local_model.to(self.device)

		start_time = time.time()
		history = self.train_model()

		training_time = round(time.time() - start_time, 4)
		final_acc = history[-1]["train_acc"] if history else 0.0

		return history, training_time

	
	
	#############################################
	# Integrated Model Evaluation Functionality #
	#############################################

	@torch.no_grad()
	def _evaluate_localmodel_global_small(self, model, test_loader):
		"""
		Evaluate the local model on a given test loader.
		Returns dict with 'test_acc' and 'test_loss'.
		"""
		criterion = nn.CrossEntropyLoss()
		test_loss, correct, total = 0.0, 0, 0

		for batch_idx, (inputs, targets) in enumerate(test_loader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			outputs = model(inputs)  # Generate predictions
			loss = criterion(outputs, targets)  # Calculate loss

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			# For debug
			if not self.g_para.environ.get("SBATCH", False) and self.g_para.Debug["d_train"]:
				self.utils.progress_bar(
					f"#{self.id}-Test",
					batch_idx,
					len(test_loader),
					f'Loss: {test_loss / (batch_idx + 1):.3f} | '
					f'Acc: {100. * correct / total:.3f}% ({correct}/{total})'
				)

		return {
			'test_acc': round(correct / total, 4) if total > 0 else 0.0,
			'test_loss': round(test_loss / (batch_idx + 1), 4) if batch_idx >= 0 else 0.0
		}

	def evaluate_client_model(self, dataloader_global_test_small, history):
		"""
		Evaluate this client's local model. Depending on self.g_para.client_select_param["criteria"], 
		we either:
		- Evaluate against a global test dataset (IID test dataset), or
		- Use the most recent validation accuracy in the training history.
		"""
		if self.g_para.client_select_param["criteria"] == "acc":
			result = self._evaluate_localmodel_global_small(self.local_model, dataloader_global_test_small)
			accuracy_table = result["test_acc"]
			if self.g_para.Debug["d_lr"]:
				print(
					f"{self.id}th local model shows the acc {accuracy_table} "
					f"with server test dataset (IID test datasets):"
				)
		else:
			# e.g., use last validation accuracy
			accuracy_table = history[-1]['val_acc'] if self.g_para.numEpoch > 0 else 0
			if self.g_para.Debug["model"]:
				print(
					f"{self.id}-th model, epoch:({self.g_para.numEpoch}), "
					f"accuracy_table:({accuracy_table})"
				)
		return accuracy_table

