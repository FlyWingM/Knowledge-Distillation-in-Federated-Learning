### model_aggregation_transmission.py

import torch
import torch.cuda

Debug_1, Debug_2 = True, True

def set_averaged_weights_in_global_model(g_para, server, clients, ms_index=0):
	selected_client_group = g_para.selected_client_group

	# Aggregation function with data-weighted averaging
	def aggregate_weights():
		new_params = {name: torch.zeros_like(param.data) for name, param in server.g_model.named_parameters()}

		# Calculate the total number of samples across all selected clients
		total_samples = sum(
			server.server_local_info[index_selected]['train_num_samples']
			for index_selected in selected_client_group
			if index_selected in clients
		)
		if total_samples == 0:
			print("Warning: Total number of samples is zero.")
			return new_params

		count_t = 0

		if Debug_1:
			print("\n--- Client Weighting Information ---")
			print("Client ID | Num Samples | Weight")
			print("-----------------------------------")

		num_clients= len(selected_client_group)
		# Aggregate weights from the selected clients in the group
		for index_selected in selected_client_group:
			if index_selected in clients:
				try:
					#local_state_dict = clients[index_selected].local_model.state_dict()
					local_state_dict = server.server_local_models[index_selected]
					client_samples = server.server_local_info[index_selected]['train_num_samples']
					#client_weight = client_samples / total_samples
					client_weight = 1.0 / num_clients

					# Display weighting information for each client
					if Debug_1: print(f"{index_selected:<9} | {client_samples:<11} | {client_weight:.4f}")

					for name, param in local_state_dict.items():
						if name in new_params:
							if param.shape == new_params[name].shape:
								new_params[name] += param * client_weight
						elif "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
						#	print(f"Initializing missing BatchNorm parameter '{name}' in the global model.")
						#	new_params[name] = torch.zeros_like(param) + param * client_weight
							continue
						#else:
						#	print(f"Parameter '{name}' not found in the global model. Skipping.")
					count_t += 1
				except KeyError as e:
					print(f"KeyError: {e}. Skipping client {index_selected}.")
			else:
				print(f"Client {index_selected} not found in clients. Skipping this client.")

		if Debug_1:
			print("\n-----------------------------------")
			print(f"Aggregated models from {count_t} clients.")

		return new_params
	
	def recalibrate_batchnorm(server):
		print("Recalibrating BatchNorm statistics...")
		server.g_model.train()  # Set model to training mode for recalibration
		with torch.no_grad():
			for inputs, _ in server.dataloader_global_calibration:
				inputs = inputs.to(g_para.device)
				server.g_model(inputs)  # Forward pass to update BatchNorm statistics
		print("BatchNorm recalibration completed.")

	try:
		# For FedAvg, perFedAvg, or other algorithms
		if g_para.client_select_param.get("learning_exploration") in ["EO", "EW"] and ms_index == 1:
			server.g_model_deposit.load_state_dict(server.g_model.state_dict())
		elif g_para.client_select_param.get("learning_exploration") in ["EO", "EW"]:
			server.g_model.load_state_dict(server.g_model_deposit.state_dict())

		averaged_params = aggregate_weights()
		try:
			server.g_model.load_state_dict(averaged_params)
		except RuntimeError as e:
			print(f"RuntimeError loading state dict: {e}")
		
		# Recalibrate BatchNorm statistics after aggregation
		# recalibrate_batchnorm(server)

	except Exception as e:
		print(f"Unexpected error during aggregation: {e}")


def idential_from_gmodel_to_lmodels(g_para, server, clients):
	selected_client_group = g_para.selected_client_group
	state_dict_global = server.g_model.state_dict()
	identical_for_all_flag = True

	for index in selected_client_group:
		if index in clients:
			state_dict_local = clients[index].local_model.state_dict()
			for key in state_dict_global.keys():
				if key in state_dict_local:
					# Compare tensors element-wise
					if not torch.equal(state_dict_global[key], state_dict_local[key]):
						identical_for_all_flag = False
						print(f"Mismatch found in parameter: {key} for client {index}.")
						break
				else:
					print(f"Parameter '{key}' is missing in the local model of client {index}.")
					identical_for_all_flag = False
					break
		else:
			print(f"Client {index} not found in clients. Skipping this client.")
			identical_for_all_flag = False

		# Compare keys in an example client model and the global model
		client_index = 0  # Example client index; adjust as needed
		keys_for_client = set(clients[client_index].local_model.state_dict().keys()) - set(server.g_model.state_dict().keys())
		keys_for_server = set(server.g_model.state_dict().keys()) - set(clients[client_index].local_model.state_dict().keys())
		if keys_for_client or keys_for_server:
			print(f"Keys in client model but not in global model: {keys_for_client}")
			print(f"Keys in global model but not in client model: {keys_for_server}")

	print(f"  All {len(selected_client_group)} clients are {'identical' if identical_for_all_flag else 'not identical'}.")


def transmission_gmodel_to_local_clients(g_para, server, clients):
	print(f"Starting transmission of global model to {len(g_para.selected_client_group)} clients...")
	selected_client_group = g_para.selected_client_group
	with torch.no_grad():
		for index in selected_client_group:
			if index in clients:
				try:
					clients[index].local_model.load_state_dict(server.g_model.state_dict())
				except RuntimeError as e:
					print(f"Error loading state dict for client {index}: {e}")
			else:
				print(f"Client {index} not found in clients. Skipping this client.")

	idential_from_gmodel_to_lmodels(g_para, server, clients)
	print("  Global model transmission completed.")
