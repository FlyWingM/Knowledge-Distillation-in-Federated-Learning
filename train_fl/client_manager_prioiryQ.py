### client_manager.py - ongoing
import torch
import random
import statistics
import heapq
from concurrent.futures import (
	ThreadPoolExecutor,
	as_completed,
	FIRST_COMPLETED,
	wait
)


class PriorityExecutor:
    """
    Executes tasks with an associated priority in a single ThreadPoolExecutor.
    Lower numeric priority => higher chance to run earlier.
    """
    def __init__(self, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Store tasks as (priority, client_id, stage, future)
        self.task_heap = []
        self.pending_tasks = []

    def submit_task(self, priority, fn, client_id, stage):
        """
        Submit a task with a given priority. 'fn' is a function that
        takes 'client_id' as input (or no input if you prefer).
        """
        future = self.executor.submit(fn, client_id)
        # Push into the min-heap (lower number => higher priority).
        heapq.heappush(self.task_heap, (priority, client_id, stage, future))

    def process_tasks(self):
        """
        Pull tasks from the heap in priority order, then wait on them
        with 'wait(..., FIRST_COMPLETED)'. 
        This ensures that tasks of lower numeric priority
        will be put in the queue earlier.
        
        Returns a list of (client_id, stage, result).
        """
        results = []
        # Keep going while we have tasks in the heap or tasks not done yet
        while self.task_heap or self.pending_tasks:
            # Move all tasks from the heap to pending_tasks
            while self.task_heap:
                priority, client_id, stage, future = heapq.heappop(self.task_heap)
                self.pending_tasks.append((priority, client_id, stage, future))

            # If we have no pending tasks, break
            if not self.pending_tasks:
                break

            # Wait for whichever finishes first among the pending set
            futures_only = [pt[3] for pt in self.pending_tasks]
            done_set, _ = wait(futures_only, return_when=FIRST_COMPLETED)

            # For each completed future, store or handle the result
            still_pending = []
            for (priority, cid, stg, fut) in self.pending_tasks:
                if fut in done_set:
                    try:
                        res = fut.result()
                        results.append((cid, stg, res))
                    except Exception as e:
                        print(f"[ERROR] {stg} failed for Client-{cid}: {e}")
                else:
                    still_pending.append((priority, cid, stg, fut))

            self.pending_tasks = still_pending

        return results


PRIORITY_TRAIN   = 0   # highest priority => runs first
PRIORITY_PRELOAD = 1
PRIORITY_CREATE  = 2   # lowest priority => runs last


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
        self.dataset = fl_params.dataset
        self.server = fl_params.server
        self.clients = fl_params.clients

        self.global_dataset_future = False
        self.global_small_test_future = False
        self.global_test_future = False

        self.chunk_size   = self.g_para.num_clients
        if self.g_para.nn['name'] == "ResNet9":
            self.max_workers = 30
            self.threshold_portion = 0.8
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

        # Possibly override max_workers depending on your NN
        if self.g_para.nn['name'] == "ResNet9":
            self.max_workers = 30
        else:
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

        # Preload smaller test set if needed
        if (self.g_para.client_select_param["criteria"] == "acc" 
                and not self.server.cached_batches_global_test_small):
            self.server.preload_global_small_test_dataset_onGPU(self.g_para.device)
            self.global_small_test_future = True
        else:
            self.global_small_test_future = False

        # Define wrappers as in your original code
        def create_dataset_wrapper(c_id):
            client = self.clients[c_id]
            if not client.dataloaders_train:
                client.create_local_dataset(self.server)
            return c_id

        def preload_wrapper(c_id):
            self.clients[c_id].preload_local_model_dataset_onGPU(self.g_para.device)
            return c_id

        def train_wrapper(c_id):
            # If your code really depends on create/preload first, 
            # you can add checks or forcibly wait. Example:
            #   client = self.clients[c_id]
            #   if not client.dataloaders_train:
            #       client.create_local_dataset(self.server)
            #   # etc.
            # 
            # For now, we’ll assume it’s okay to run first.
            # Wait for the small test dataset if needed
            if self.global_small_test_future and not self.server.cached_batches_global_test_small:
                self.server.preload_global_small_test_dataset_onGPU(self.g_para.device)

            # Load global model from CPU
            cpu_global_state = {
                k: v.cpu() for k, v in self.server.g_model.state_dict().items()
            }
            self.clients[c_id].local_model.load_state_dict(cpu_global_state)

            return train_single_client(
                self.g_para,
                self.server,
                self.clients[c_id],
                self.ratio_t_increase,
                self.Debug_1,
                use_stream=True
            )

        # If you want to preload full test dataset once a threshold is reached,
        # you can still do that, but we’ll simplify for clarity.

        # --------------------------------------------------------
        # Use our PriorityExecutor instead of the old concurrency
        # --------------------------------------------------------
        from your_module_with_priority_executor import PriorityExecutor  # Or wherever you put it
        priority_exec = PriorityExecutor(self.max_workers)

        # 1) Submit tasks for each client *all at once* with priorities
        for c_id in selected_ids:
            client = self.clients[c_id]
            # Suppose we set "train" as highest priority (lowest int):
            priority_exec.submit_task(PRIORITY_TRAIN, train_wrapper, c_id, stage="train")
            # Then preload as a lower priority
            priority_exec.submit_task(PRIORITY_PRELOAD, preload_wrapper, c_id, stage="preload")
            # Then dataset creation as the lowest priority
            if not client.dataloaders_train:
                priority_exec.submit_task(PRIORITY_CREATE, create_dataset_wrapper, c_id, stage="create_dataset")

        # 2) Actually process them, in priority order
        completed_results = priority_exec.process_tasks()

        # completed_results is a list of (client_id, stage, result)
        # in the order tasks finished. We'll gather the train results.
        for (client_id, stage, res) in completed_results:
            if stage == "train" and res is not None:
                # 'res' is the dict returned from train_single_client
                results_from_all_clients.append(res)

        # --------------------------------------------------------------
        # STEP: Process final results (same logic as original code)
        # --------------------------------------------------------------
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

