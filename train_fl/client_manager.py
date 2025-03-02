### client_manager.py - ongoing
import torch
import random
import statistics

from concurrent.futures import (
	ThreadPoolExecutor,
	as_completed,
	FIRST_COMPLETED,
	wait
)

def train_single_client(
	g_para, 
	server, 
	client, 
	ratio_t_increase, 
	Debug_1,
	use_stream=False
):
	"""
	Worker function for parallel processing. 
	This runs the training and evaluation for a single client on a separate CUDA stream (if use_stream=True).
	"""
	g_index = g_para.g_iter_index
	client_local_info = server.server_local_info[client.id]

	# Optional: create a stream for GPU concurrency
	if use_stream and g_para.device.type == 'cuda':
		stream = torch.cuda.Stream(device=g_para.device)
		with torch.cuda.stream(stream):
			# Perform training
			history, training_time = client._train_once(None) 
			# Evaluate accuracy
			if g_para.client_select_param["criteria"] == "acc":
				accuracy_table = client.evaluate_client_model(
					server.cached_batches_global_test_small,
					history
				)
				accuracy_table = round(accuracy_table, 4)
			else:
				accuracy_table = history['val_acc']
		# Block until this client's stream finishes (ensures correctness)
		stream.synchronize()
	else:
		# Fall back to standard training on the default stream
		history, training_time = client._train_once(None)
		if g_para.client_select_param["criteria"] == "acc":
			accuracy_table = client.evaluate_client_model(
				server.cached_batches_global_test_small,
				history
			)
			accuracy_table = round(accuracy_table, 4)
		else:
			accuracy_table = history['val_acc']

	accuracy_table = round(accuracy_table, 4)
	client_local_info["local_train_time"][g_index] = training_time * ratio_t_increase

	# Calculate carbon (same as before)
	time_of_complete = (
		client_local_info["local_offset_time"][g_index] 
		+ client_local_info["local_train_time"][g_index]
	)
	hour_index = int(time_of_complete // 3600)
	carbon_intensity_at_hour = g_para.carbon_ratio_client[int(client.id)][hour_index]
	carbon_emission = round(
		client_local_info["local_train_time"][g_index] 
		* ratio_t_increase 
		* carbon_intensity_at_hour, 
		2
	)
	client_local_info["local_carbon"][g_index] = carbon_emission
	client_local_info["local_carbon_intensity"][g_index] = carbon_intensity_at_hour

	if Debug_1:
		print(
			f"Client-{client.id}: offset=({client_local_info['local_offset_time'][g_index]}), "
			f"train=({client_local_info['local_train_time'][g_index]}), acc=({accuracy_table})"
		)

	client.local_model.to('cpu')
	client.cached_batches_train = []
	client.cached_batches_valid = []

	# Return final info
	return {
		"client_id": client.id,
		"model_state": {k: v.cpu() for k, v in client.local_model.state_dict().items()},
		"train_time": client_local_info["local_train_time"][g_index],
		"carbon_emission": client_local_info["local_carbon"][g_index],
		"acc": accuracy_table,
	}


class ClientManager:
	def __init__(self, fl_params):
		self.fl_params = fl_params
		self.g_para = fl_params.g_para
		self.dataset = fl_params.dataset  # shared dataset object
		self.server = fl_params.server
		self.clients = fl_params.clients  # dict of {client_id: Client instance}

		self.global_dataset_future = False
		self.global_small_test_future = False
		self.global_test_future = False

		self.chunk_size   = self.g_para.num_clients
		if self.g_para.nn['name'] == "ResNet9":
			self.max_workers = 50
			self.threshold_portion = 0.7
		else:
			self.max_workers = 10
			self.threshold_portion = 0.92
		
		self.ratio_t_increase = 1
		self.Debug_1, self.Debug_2 = False, True


	def train_clients(self):
		g_index = self.g_para.g_iter_index
		global_info = self.g_para.global_info
		selected_client_group = self.g_para.selected_client_group
		nonselected_client_group = self.g_para.nonselected_client_group

		if self.g_para.nn['name'] == "ResNet9" and g_index == 1:
			self.max_workers = 16
		elif g_index == 1:
			self.max_workers = 10

		accuracy_table_total = []
		local_kl_avg_t = []
		selected_ids = list(selected_client_group.keys())

		# Reset aggregator for this round
		global_info["total_local_train_time"][g_index] = 0.0
		global_info["local_carbon_total"][g_index] = 0.0

		results_from_all_clients = []
		total_tasks = len(selected_ids)
		threshold_count = int(self.threshold_portion * total_tasks)

		# Flag & future to preload the full (large) global test dataset
		global_full_test_data_future_flag = False
		global_full_test_data_future = None

		# Preload smaller test set if needed
		if (self.g_para.client_select_param["criteria"] == "acc" 
				and not self.server.cached_batches_global_test_small):
			self.server.preload_global_small_test_dataset_onGPU(self.g_para.device)
			self.global_small_test_future = True
		else:
			self.global_small_test_future = False

		# ------------------------------------------------------------------
		# STEP 0: Helper wrappers for the three main stages
		# ------------------------------------------------------------------
		def create_dataset_wrapper(c_id):
			"""
			Create the local dataset (if not already present).
			Return c_id for chaining.
			"""
			client = self.clients[c_id]
			if not client.dataloaders_train:
				client.create_local_dataset(self.server)
			return c_id

		def preload_wrapper(c_id):
			"""
			Preload local model & dataset onto GPU.
			Return c_id for chaining.
			"""
			self.clients[c_id].preload_local_model_dataset_onGPU(self.g_para.device)
			return c_id

		def train_wrapper(c_id):
			"""
			Actually run the training for a single client.
			Returns the result dict from train_single_client().
			"""
			# Wait for the small test dataset if needed
			if self.global_small_test_future and not self.server.cached_batches_global_test_small:
				self.server.preload_global_small_test_dataset_onGPU(self.g_para.device)

			# Load global model from CPU
			cpu_global_state = {
				k: v.cpu() for k, v in self.server.g_model.state_dict().items()
			}
			self.clients[c_id].local_model.load_state_dict(cpu_global_state)

			# Now train
			return train_single_client(
				self.g_para,
				self.server,
				self.clients[c_id],
				self.ratio_t_increase,
				self.Debug_1,
				use_stream=True
			)

		def preload_global_fulltest_wrapper(c_id):
			"""
			Preload the large global test dataset.
			Return c_id for chaining.
			"""
			self.server.preload_global_test_dataset(self.g_para.device)
			return c_id

		# ------------------------------------------------------------------
		# Use a single ThreadPoolExecutor to schedule tasks dynamically.
		# ------------------------------------------------------------------
		futures_map = {}  
		with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
			# Schedule the large global test dataset preload once we reach threshold_count,
			# but do not .result() until we actually need it (after training).

			# ---------------------------------------------------------------
			# STEP 1: Submit the initial tasks for all selected clients.
			#         BUT if the dataset is already created, skip directly
			#         to the preload_wrapper stage.
			# ---------------------------------------------------------------
			for c_id in selected_ids:
				client = self.clients[c_id]
				if not client.dataloaders_train:
					# Dataset is not created yet; schedule create_dataset_wrapper first
					fut = executor.submit(create_dataset_wrapper, c_id)
					futures_map[fut] = {"client_id": c_id, "stage": "dataset"}
				else:
					# If already loaded, skip dataset creation; go right to preload
					fut = executor.submit(preload_wrapper, c_id)
					futures_map[fut] = {"client_id": c_id, "stage": "preload"}

			# ---------------------------------------------------------------
			# STEP 2: As tasks finish, schedule the next stage
			# ---------------------------------------------------------------
			while futures_map:
				done_set, _ = wait(futures_map.keys(), return_when=FIRST_COMPLETED)

				for done_future in done_set:
					info = futures_map.pop(done_future)
					c_id = info["client_id"]
					stage = info["stage"]

					if stage == "dataset":
						# Now that dataset is created, proceed to preload
						try:
							done_future.result()  # might raise
							fut = executor.submit(preload_wrapper, c_id)
							futures_map[fut] = {"client_id": c_id, "stage": "preload"}
						except Exception as e:
							print(f"[ERROR] create_dataset_wrapper failed for Client-{c_id}: {e}")

					elif stage == "preload":
						# Preload is done, schedule train
						try:
							done_future.result()  # might raise
							fut = executor.submit(train_wrapper, c_id)
							futures_map[fut] = {"client_id": c_id, "stage": "train"}
						except Exception as e:
							print(f"[ERROR] preload_wrapper failed for Client-{c_id}: {e}")

					elif stage == "train":
						# Final stage for this client
						try:
							res = done_future.result()  # might raise
							results_from_all_clients.append(res)

							# Possibly schedule large test dataset preload if threshold is reached
							if (not global_full_test_data_future_flag
								and len(results_from_all_clients) >= threshold_count):
								global_full_test_data_future = executor.submit(
									preload_global_fulltest_wrapper, 1000
								)
								global_full_test_data_future_flag = True

						except Exception as e:
							print(f"[ERROR] train_wrapper failed for Client-{c_id}: {e}")

			# All clients have finished training by now.
			# Ensure the large global test set is loaded if we scheduled it
			if global_full_test_data_future and not global_full_test_data_future_flag:
				global_full_test_data_future.result()

		# ---------------------------------------------------------------
		# STEP 3: Process final results (same as in the original code)
		# ---------------------------------------------------------------
		for res in results_from_all_clients:
			client_id = res["client_id"]
			cpu_model_state = res["model_state"]
			local_acc = res["acc"]
			local_train_time = res["train_time"]
			carbon_emission = res["carbon_emission"]

			client = self.clients[client_id]
			client_local_info = self.server.server_local_info[client_id]

			# Save local models on the server
			self.server.server_local_models[client_id] = {
				key: value.to(self.g_para.device)
				for key, value in cpu_model_state.items()
			}
			client_local_info['local_select'][g_index] = True

			client_local_info["local_acc"].append(local_acc)
			client_local_info["local_train_time"][g_index] = local_train_time
			client_local_info["local_carbon"][g_index] = carbon_emission

			global_info["total_local_train_time"][g_index] += local_train_time
			global_info["local_carbon_total"][g_index] += carbon_emission

			kl_div = client_local_info["kl_div"][0]
			local_kl_avg_t.append(kl_div)
			accuracy_table_total.append(local_acc)

			if self.Debug_1:
				print(
					f"[train_clients] Finished Client-{client_id}, "
					f"g_index({g_index}), train_time={local_train_time}, "
					f"carbon={carbon_emission}, acc={local_acc}, kl_div={kl_div}"
				)

		num_selected_clients = len(selected_ids)
		if num_selected_clients > 0:
			avg_accuracy = sum(accuracy_table_total) / num_selected_clients
			avg_carbon = (
				global_info["local_carbon_total"][g_index] / num_selected_clients
			)
		else:
			avg_accuracy = 0.0
			avg_carbon = 0.0

		# Fill placeholders for non-selected clients
		for client_id in nonselected_client_group:
			client_local_info = self.server.server_local_info[client_id]
			estimated_accuracy = avg_accuracy * (random.random() * 0.2 + 0.9)
			client_local_info["local_acc"].append(round(estimated_accuracy, 4))

			carbon_for_non_client = avg_carbon * (random.random() * 0.2 + 0.9)
			client_local_info["local_carbon"][g_index] = round(carbon_for_non_client, 4)

		# Determine largest local train time among selected
		if num_selected_clients > 0:
			client_with_largest_time = max(
				selected_ids,
				key=lambda cid: self.server.server_local_info[cid]["local_train_time"][g_index]
			)
			largest_time = self.server.server_local_info[client_with_largest_time]["local_train_time"][g_index]
		else:
			largest_time = 0.0

		# Offset time for the next global iteration
		if g_index < self.g_para.num_g_iter - 1:
			for cid in self.clients:
				self.server.server_local_info[cid]["local_offset_time"][g_index + 1] = (
					self.server.server_local_info[cid]["local_offset_time"][g_index]
					+ largest_time
				)

		self.g_para.hour_index = int(
			self.server.server_local_info[0]["local_offset_time"][g_index + 1] // 3600
		)

		if self.g_para.Debug["carbon"]:
			for cid in self.clients:
				print(
					f"[train_clients] Client-{cid}'s Carbon emission: "
					f"{self.server.server_local_info[cid]['local_carbon'][g_index]}"
				)



	def set_averaged_weights_in_global_model(self, ms_index=0):
		selected_client_group = self.g_para.selected_client_group
		le_param = self.g_para.client_select_param.get("learning_exploration")

		def aggregate_weights():
			new_params = {
				name: torch.zeros_like(param.data)
				for name, param in self.server.g_model.named_parameters()
			}

			# Calculate the total number of samples across all selected clients
			total_samples = sum(
				self.server.server_local_info[index_selected]['train_num_samples']
				for index_selected in selected_client_group
				if index_selected in self.clients
			)
			if total_samples == 0:
				print("Warning: Total number of samples is zero.")
				return new_params

			count_t = 0

			if self.Debug_2:
				print("\n--- Client Weighting Information ---")
				print("Client ID | Num Samples | Weight")
				print("-----------------------------------")

			num_clients= len(selected_client_group)
			# Aggregate weights from the selected clients in the group
			for index_selected in selected_client_group:
				if index_selected in self.clients:
					try:
						local_state_dict = self.server.server_local_models[index_selected]
						client_samples = self.server.server_local_info[index_selected]['train_num_samples']
						# Example: uniform weighting or data-based weighting
						client_weight = 1.0 / num_clients
						# Example alternative: client_weight = client_samples / total_samples

						if self.Debug_2 and count_t < 4:
							print(f"{index_selected:<9} | {client_samples:<11} | {client_weight:.4f}")

						for name, param in local_state_dict.items():
							if name in new_params:
								if param.shape == new_params[name].shape:
									new_params[name] += param * client_weight
							elif any(x in name for x in [
									"running_mean", "running_var", "num_batches_tracked"
							]):
								# For BatchNorm buffers that may not exist in the global model
								new_params[name] = torch.zeros_like(param) + param * client_weight
								continue

						count_t += 1
					except KeyError as e:
						print(f"KeyError: {e}. Skipping client {index_selected}.")
				else:
					print(f"Client {index_selected} not found in clients. Skipping this client.")

			if self.Debug_2:
				print("\n-----------------------------------")
				print(f"Aggregated models from {count_t} clients.")

			return new_params

		try:
			# Handle special conditions if needed
			if le_param in ["EO", "EW"] and ms_index == 1:
				self.server.g_model_deposit.load_state_dict(self.server.g_model.state_dict())
			elif le_param in ["EO", "EW"]:
				self.server.g_model.load_state_dict(self.server.g_model_deposit.state_dict())

			averaged_params = aggregate_weights()
			try:
				self.server.g_model.load_state_dict(averaged_params)
			except RuntimeError as e:
				print(f"RuntimeError loading state dict: {e}")

		except Exception as e:
			print(f"Unexpected error during aggregation: {e}")

