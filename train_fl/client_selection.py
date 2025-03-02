### client_selection.py

import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score

class ClientSelection:
	def __init__(self, fl_params):
		self.fl_params = fl_params
		self.g_para = fl_params.g_para
		self.server = fl_params.server
		self.clients = fl_params.clients

		self.Debug_1 = True
		self.Debug_2 = True

	def get_model_weights(self, model):
		weights = np.concatenate([param.data.cpu().numpy().flatten() for param in model.parameters()])
		return weights

	def compare_weights(self, global_weights, local_weights_list):
		differences = [np.mean(np.abs(global_weights - local_weights)) 
					for local_weights in local_weights_list.values()]
		return differences

	def apply_pca_and_find_distances(self, weights_list, n_components=0.95):
		pca = PCA(n_components=n_components)
		reduced_weights = pca.fit_transform(weights_list)
		centroid = np.mean(reduced_weights, axis=0)
		distances = [euclidean(point, centroid) for point in reduced_weights]
		return distances

	def apply_cosine_similarity_and_amplify_distances(self, weights_list, exponent=2):
		# Calculate the centroid (mean) of the weight vectors
		centroid = np.mean(weights_list, axis=0)
		
		# Normalize the centroid to have unit length
		centroid_norm = centroid / np.linalg.norm(centroid)
		
		# Normalize all the weights in the list for cosine similarity calculation
		weights_list_norm = weights_list / np.linalg.norm(weights_list, axis=1, keepdims=True)
		
		# Calculate the cosine similarity between each model's weights and the centroid
		similarities = cosine_similarity(weights_list_norm, centroid_norm.reshape(1, -1)).flatten()
		
		# Clip similarities to ensure they stay within the valid range [-1, 1]
		similarities = np.clip(similarities, -1, 1)
		
		# Convert similarities to distances (1 - similarity)
		distances = 1 - similarities
		
		# Amplify small distances using exponentiation
		amplified_distances = distances ** exponent
		
		# Optionally, print debug information if Debug_2 flag is enabled
		if self.Debug_2:
			print(f"Cosine similarities: {similarities}")
			print(f"Original distances: {distances}")
			print(f"Amplified distances (exponent={exponent}): {amplified_distances}")

		return amplified_distances.tolist()

	def entropy(self, X):
		"""Calculate entropy of a discrete variable."""
		value, counts = np.unique(X, return_counts=True)
		prob = counts / len(X)
		return -np.sum(prob * np.log(prob + 1e-9))  # Adding a small value to avoid log(0)

	def calculate_reversed_nmi_between_centroid_and_models(self, weights_list, bins=10):
		# Step 1: Calculate the centroid (mean of the weight vectors)
		centroid = np.mean(weights_list, axis=0)
		
		# Step 2: Discretize the centroid and the weights in weights_list
		bins_edges = np.linspace(np.min(weights_list), np.max(weights_list), bins)  # Bin edges for discretization
		
		# Discretize the centroid
		centroid_binned = np.digitize(centroid, bins_edges)
		
		reversed_nmi_scores = []
		
		# Step 3: For each model in the weights_list, calculate the reversed NMI with the centroid
		for i, model_weights in enumerate(weights_list):
			# Discretize the model's weights
			model_binned = np.digitize(model_weights, bins_edges)
			
			# Calculate mutual information between the centroid and model weights
			mi = mutual_info_score(centroid_binned, model_binned)
			
			# Calculate entropy of the centroid and the model weights
			h_centroid = self.entropy(centroid_binned)
			h_model = self.entropy(model_binned)
			
			# Calculate NMI (normalized mutual information)
			nmi = mi / np.sqrt(h_centroid * h_model)
			
			# Reverse the NMI score (1 - NMI) to align with the desired behavior
			reversed_nmi = 1 - nmi
			reversed_nmi_scores.append(reversed_nmi)
			
			# Optionally, print debug information
			#print(f"Reversed NMI (1 - NMI) between centroid and model {i}: {reversed_nmi}")

		if self.Debug_1:
			print(f"reversed_nmi_scores: {reversed_nmi_scores}")

		return reversed_nmi_scores


	def apply_tsne_and_find_distances(self, weights_list, n_components=2, perplexity=30.0, learning_rate=200.0):
		tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
		reduced_weights = tsne.fit_transform(weights_list)
		centroid = np.mean(reduced_weights, axis=0)
		distances = [euclidean(point, centroid) for point in reduced_weights]
		return distances

	def get_indices_of_smallest(self, number_of_clients, key="local_pca"):
		# Extract the PCA distances for each client.id at the given iteration
		number_of_clients = min(number_of_clients, len(self.clients))
		distances = [self.server.server_local_info[client.id][key][self.g_para.g_iter_index] for client in self.clients.values()]
		client_ids = list(self.clients.keys())
		distance_id_pairs = list(zip(distances, client_ids))

		# Sort the pairs based on the distances
		sorted_pairs = sorted(distance_id_pairs, key=lambda x: x[0])

		# Get the specified number of self.clients with the smallest distances
		selected_client_ids = [client_id for _, client_id in sorted_pairs[:number_of_clients]]
		return selected_client_ids

	def assign_weights_to_models(self, distances, client_ids, method='inverse', alpha=1, beta=1):
		if method == 'inverse':
			weights = [1 / (d**alpha) if d != 0 else 1.0 for d in distances]
		elif method == 'exponential':
			weights = [np.exp(-beta * d) for d in distances]
		else:
			raise ValueError("Unsupported method. Choose 'inverse' or 'exponential'.")
		total_weight = sum(weights)
		normalized_weights = [w / total_weight for w in weights]
		normalized_weights_dict = {client_id: weight for client_id, weight in zip(client_ids, normalized_weights)}
		return normalized_weights_dict

	def pca_carbon_list(self, global_ind):
		self.local_pca_list = []
		self.local_weight_list = []
		self.local_carbon_list = []     
		self.local_carbon_intensity_list = []     
		for client_id, client in self.clients.items():
			pca = self.server.server_local_info[client.id]["local_pca"][global_ind]
			weight = self.server.server_local_info[client.id]["local_weight"][global_ind]
			carbon = self.server.server_local_info[client.id]["local_carbon"][global_ind]
			carbon_intensity = self.server.server_local_info[client.id]["local_carbon_intensity"][global_ind]
			self.local_pca_list.append(pca)
			self.local_weight_list.append(weight)
			self.local_carbon_list.append(carbon)
			self.local_carbon_intensity_list.append(carbon_intensity)

	def calculate_percentiles(self, values, percentile):
		return np.percentile(values, percentile)

	def by_kl_evaluation(self):
		#print(f"by_acc_evaluation with argument({self.g_para.client_select_param['criteria']})")
		g_index = self.g_para.g_iter_index
		client_group = np.zeros((self.g_para.num_clients, 6))
		client_group[:, 0] = range(len(self.clients))
		selection_rate = float(self.g_para.client_select_param["c_ratio"])

		for client_id, client in self.clients.items():
			client_group[client.id, 4] = self.server.server_local_info[client.id]["kl_div"][0]
			client_group[client.id, 5] = self.server.server_local_info[client.id]['train_num_samples']
		sort_col = 4
		ascending = False

		#sort_col = 1 # Accuracy
		client_group_df = pd.DataFrame(client_group, columns=['index', 'acc', 'pca', 'pca_carbon', 'kl', 'data_size'])
		client_group_df.sort_values(by=client_group_df.columns[sort_col], ascending=ascending, inplace=True)
		#print(f"\nclient_group, after sorting:\n{client_group_df}")

		num_sel_client = int(self.g_para.num_clients * selection_rate)
		final_client_group = {}
		selected_indices = client_group_df.iloc[:num_sel_client, 0].astype(int).tolist()
		for idx in selected_indices:
			final_client_group[idx] = round(1 / num_sel_client, 6)

		self.g_para.next_selected_client_group = final_client_group
		#print(f"\nNewly selected final client_group:\n{final_client_group}")
		#print(f"####The next selected clients by order of acc/carbon: {list(self.g_para.next_selected_client_group.keys())}")

	# So far, the standard FL calls this funciton yet.
	def by_acc_evaluation(self):
		#print(f"by_acc_evaluation with argument({self.g_para.client_select_param['criteria']})")
		g_index = self.g_para.g_iter_index
		client_group = np.zeros((self.g_para.num_clients, 4))
		client_group[:, 0] = range(len(self.clients))
		selection_rate = float(self.g_para.client_select_param["c_ratio"])

		for client_id, client in self.clients.items():
			client_group[client.id, 1] = self.server.server_local_info[client.id]["local_acc"][g_index]
		sort_col = 1
		ascending = False

		#sort_col = 1 # Accuracy
		client_group_df = pd.DataFrame(client_group, columns=['index', 'acc', 'pca', 'pca_carbon'])
		client_group_df.sort_values(by=client_group_df.columns[sort_col], ascending=ascending, inplace=True)
		#print(f"\nclient_group, after sorting:\n{client_group_df}")

		num_sel_client = int(self.g_para.num_clients * selection_rate)
		final_client_group = {}
		selected_indices = client_group_df.iloc[:num_sel_client, 0].astype(int).tolist()
		for idx in selected_indices:
			final_client_group[idx] = round(1 / num_sel_client, 6)

		self.g_para.next_selected_client_group = final_client_group
		#print(f"\nNewly selected final client_group:\n{final_client_group}")
		#print(f"####The next selected clients by order of acc/carbon: {list(self.g_para.next_selected_client_group.keys())}")

	def by_pca_tsne_evaluation(self, mode='pca'):
		g_index = self.g_para.g_iter_index
		# Extract weights from each client's model selected from the previous evaluatioin and flatten them
		local_weights_list = {client_id: self.get_model_weights(client.local_model)
								for client_id, client in self.clients.items()
								if client_id in self.g_para.selected_client_group}

		# List of model IDs
		client_ids = list(local_weights_list.keys())

		# Apply PCA or t-SNE and find distances from the centroid in the reduced-dimensional space
		# It doesn't apply the reverse formation yet, the smaller the better quality
		distances = []
		if self.g_para.client_select_param["criteria"] in ['pca', 'carbon']:
			#TMP
			#distances = self.apply_pca_and_find_distances(list(local_weights_list.values()))
			#distances = self.apply_cosine_similarity_and_amplify_distances(list(local_weights_list.values()))
			distances = self.calculate_reversed_nmi_between_centroid_and_models(list(local_weights_list.values()))
		elif self.g_para.client_select_param["criteria"] == 'tsne':
			distances = self.apply_tsne_and_find_distances(list(local_weights_list.values()))
		#elif self.g_para.client_select_param["criteria"] == 'consine':
		#	distances = self.apply_cosine_similarity_and_find_distances(list(local_weights_list.values()))

		#print(f"self.g_para.client_select_param['criteria']:({self.g_para.client_select_param['criteria']}), distances ({len(distances)}):({distances})")

		# Average distance used for non-selected clients
		pca_avg = sum(distances) / len(distances) if distances else 0

		# Assigning PCA or t-SNE distances to selected and non-selected clients
		id_to_distance = dict(zip(client_ids, distances))
		for s_client_name in self.g_para.selected_client_group.keys():
			# Update distance information for selected clients
			if s_client_name in id_to_distance:
				self.clients[s_client_name].local_info["local_pca"][self.g_para.g_iter_index] = id_to_distance[s_client_name]
			if self.Debug_1: print(f"A selected client ({s_client_name}) with the PCA values:({self.clients[s_client_name].local_info['local_pca'][g_index]})")

		for non_s_client_name in self.g_para.nonselected_client_group.keys():
			# Estimate distance for non-selected clients based on the average and random factor
			random_factor = 0.9 + 0.2 * random.random()
			pca_for_non_client = pca_avg * random_factor
			self.clients[non_s_client_name].local_info["local_pca"][self.g_para.g_iter_index] = round(pca_for_non_client, 6)
			if self.Debug_1: print(f"Non-selected client ({non_s_client_name}) with the PCA values:({self.clients[non_s_client_name].local_info['local_pca'][g_index]}) based on avg({pca_avg})")

		# Update the new selected clients by distance of PCA or TNSE
		## RL will decide the newly selected client by his neural network, not here.

		if self.g_para.client_select_param["weighting"] == 'weighting_inverse':
			self.g_para.selected_client_group = self.assign_weights_to_models(distances, client_ids, method='inverse', alpha=2)
			#print(f"\nModels with weighting_inverse:\n{final_client_group}")
		elif self.g_para.client_select_param["weighting"] == 'weighting_exponential':
			self.g_para.selected_client_group = self.assign_weights_to_models(distances, client_ids, method='exponential', beta=1)
			#print(f"\nModels with weighting_exponential:\n{final_client_group}")

		if mode == 'rl' or mode == 'carbon':
			pass        # RL already determined the set of the selected clients. Or Carbon selection will be based on the PCA values
		else:
			#if self.g_para.client_select_param["c_ratio"] != 1.0: # Client selection based on local_info[]["local_pca"], instead of distances that is from only the selected clinets
				selection_rate = float(self.g_para.client_select_param["c_ratio"])

				###Q Ensure that the small distance is better
				selected_client_ids = self.get_indices_of_smallest(int(self.g_para.num_clients * selection_rate), "local_pca")
				#Otherwise self.get_indices_of_largest_values

				self.g_para.next_selected_client_group = {client_id: round(1 / len(selected_client_ids), 6) for client_id in selected_client_ids}                    
				#print(f"\n{len(self.g_para.next_selected_client_group)} Models with distance of PCA and TNSE:\n{self.g_para.next_selected_client_group.keys()}")
				#print(f"The next selected client group list by order:\n{list(self.g_para.next_selected_client_group.keys())}")

	# So far, the standard FL calls this funciton yet.
	def by_carbon_evaluation(self):
		#print(f"by_acc_evaluation with argument({self.g_para.client_select_param['criteria']})")
		g_index = self.g_para.g_iter_index
		client_group = np.zeros((self.g_para.num_clients, 4))
		client_group[:, 0] = range(len(self.clients))
		selection_rate = float(self.g_para.client_select_param["c_ratio"])

		# Calcuate the pca values
		self.pca_carbon_list(g_index)
		percentilea = [40, 70, 100]
		thresholds = [self.calculate_percentiles(self.local_carbon_intensity_list, percentile) for percentile in percentilea]
		if self.Debug_1: print(f"\nthresholds:\n{thresholds}")

		for client_id, client in self.clients.items():
			for i, threshold in enumerate(thresholds):
				carbon_inten_t = self.server.server_local_info[client.id]["local_carbon_intensity"][g_index]
				# Flag to check if a threshold was found
				threshold_found = False

				if carbon_inten_t == 0:
					self.server.server_local_info[client.id]["local_carbon_intensity_level"][g_index] = 1
					threshold_found = True
					break	# Exit the inner loop without executing the rest of statements
				# No invert, the inversion should be applied when it is used.
				elif 0 < carbon_inten_t <= threshold:
					self.server.server_local_info[client.id]["local_carbon_intensity_level"][g_index] = (0.8 + 0.2*i)
					threshold_found = True
					break	# Exit the inner loop without executing the rest of statements
			if not threshold_found:
				self.server.server_local_info[client.id]["local_carbon_intensity_level"][g_index] = 1.2
				print(f"!!!!by_carbon_evaluation->Wired!!! carbon_inten_t({carbon_inten_t}) > threshold({thresholds})")
				#raise ValueError(f"!!!!by_carbon_evaluation->Wired!!! carbon_inten_t({carbon_inten_t}) > threshold({threshold})")

		for client_id, client in self.clients.items():
			current_pca = self.server.server_local_info[client.id]["local_pca"][g_index]
			incentive_by_carbon = self.server.server_local_info[client.id]["local_carbon_intensity_level"][g_index]
			carbon_intensity = self.server.server_local_info[client.id]["local_carbon_intensity"][g_index]
			client_group[client.id, 3] =  current_pca*incentive_by_carbon
			if self.Debug_2 and client.id < 3:
				print(f"{client.id}-client has the carbon-based pca evaluation({client_group[client.id, 3]}) = ({current_pca}) X ({incentive_by_carbon}) of ({carbon_intensity})")

		sort_col = 3
		ascending = True

		#sort_col = 4 # pca_carbon
		client_group_df = pd.DataFrame(client_group, columns=['index', 'acc', 'pca', 'pca_carbon'])
		if self.Debug_1: print(f"\nby_carbon_evaluation-client_group, before sorting:\n{client_group_df}")

		client_group_df.sort_values(by=client_group_df.columns[sort_col], ascending=ascending, inplace=True) #ascending:False, the higher pca, the better
		if self.Debug_1: print(f"\nby_carbon_evaluation-client_group, after sorting:\n{client_group_df}")

		num_sel_client = int(self.g_para.num_clients * selection_rate)
		final_client_group = {}
		selected_indices = client_group_df.iloc[:num_sel_client, 0].astype(int).tolist()
		for idx in selected_indices:
			final_client_group[idx] = round(1 / num_sel_client, 6)

		self.g_para.next_selected_client_group = final_client_group


	def by_weight_evaluation(self):
		g_index = self.g_para.g_iter_index
		global_weights = self.get_model_weights(self.server.g_model)
		#local_weights_list = {int(client_id.split('-')[1]): self.get_model_weights(client.local_model) for client_id, client in self.clients.items()}
		local_weights_list = {client_id: self.get_model_weights(client.local_model) 
								for client_id, client in self.clients.items() if client_id in self.g_para.selected_client_group}
		# List of model IDs
		client_ids = list(local_weights_list.keys())

		weight_differences = self.compare_weights(global_weights, local_weights_list)
		weight_avg = sum(weight_differences) / len(weight_differences) if weight_differences else 0             # Average distance used for non-selected clients

		id_to_distance = dict(zip(client_ids, weight_differences))
		for s_client_name in self.g_para.selected_client_group.keys(): # Update distance information for selected clients            
			if s_client_name in id_to_distance:
				self.clients[s_client_name].local_info["local_weight"][g_index] = id_to_distance[s_client_name]
				#print(f"A selected client ({s_client_name}) with the weight values:({id_to_distance[s_client_name]})==({self.clients[s_client_name].local_info['local_weight'][g_index]})")

		for non_s_client_name in self.g_para.nonselected_client_group.keys():
			# Estimate distance for non-selected clients based on the average and random factor
			random_factor = 0.9 + 0.2 * random.random()
			weight_avg_for_non_client = weight_avg * random_factor
			self.clients[non_s_client_name].local_info["local_weight"][g_index] = round(weight_avg_for_non_client, 6)
			#print(f"A non-selected client ({non_s_client_name}) with the weight values:({elf.clients[non_s_client_name].local_info['local_weight'][g_index]})")

		client_ids = list(local_weights_list.keys())
		if self.g_para.client_select_param["weighting"] in ['weighting_inverse']:
			self.g_para.selected_client_group = self.assign_weights_to_models(list(weight_differences.values()), client_ids, method='inverse', alpha=1)
		elif self.g_para.client_select_param["weighting"] in ['weighting_exponential']:
			self.g_para.selected_client_group = self.assign_weights_to_models(list(weight_differences.values()), client_ids, method='exponential', beta=1)

		if self.g_para.data_distribution_type == 'rl':
			pass        # RL already determined the set of the selected clients.
		else:
			if self.g_para.client_select_param["c_ratio"] != 1.0:
				selection_rate = float(self.g_para.client_select_param["c_ratio"])
				selected_client_ids = self.get_indices_of_smallest(int(self.g_para.num_clients * selection_rate), "local_weight")
				self.g_para.next_selected_client_group = {client_id: round(1 / len(selected_client_ids), 6) for client_id in selected_client_ids}                    
				#print(f"\n{len(self.g_para.next_selected_client_group)} Models with distance of weight:\n{self.g_para.next_selected_client_group.keys()}")


	def determine_selected_clients_by_next_selected_client(self):
		# Initialize selected_client_group and nonselected_client_group as empty dictionaries
		self.g_para.selected_client_group = {}
		self.g_para.nonselected_client_group = {}

		# Check if the next selected client group is empty
		if self.g_para.client_select_param["c_ratio"] == 1.0 and self.g_para.next_selected_client_group:
			if self.g_para.num_clients == len(self.g_para.next_selected_client_group):
				# Assign the next selected client group to selected_client_group; its client list is sorted by the requirment, such as acc, pca, carbon, etc.
				self.g_para.selected_client_group = self.g_para.next_selected_client_group
		elif self.g_para.next_selected_client_group:
			self.g_para.selected_client_group = self.g_para.next_selected_client_group
			# Identify non-selected clients
			all_clients = [i for i in self.clients.keys()]
			self.g_para.nonselected_client_group = {client_id: 0 for client_id in all_clients if client_id not in self.g_para.selected_client_group}
		else:
			# Populate selected_client_group with all clients equally without no priority
			equal_weight = 1 / self.g_para.num_clients
			self.g_para.selected_client_group = {index: equal_weight for index in self.clients.keys()}

		##print(f"###. The selected clients by order of acc/carbon: {list(self.g_para.selected_client_group.keys())}")
		# Append the keys of nonselected_client_group to nonselected_client_group_list
		self.g_para.nonselected_client_group_list.append(list(self.g_para.nonselected_client_group.keys()))

		self.g_para.next_selected_client_group = {}

		# Print the last 5 non-selected clients or all if fewer than 5
		recent_nonselected_clients = self.g_para.nonselected_client_group_list[-5:]
		clients_to_print = "\n".join(map(str, recent_nonselected_clients))
		#print(f"Clients not selected (last 5):\n{clients_to_print}")


	def compute_gradient_magnitude(self, included_gradients):

		# To store the gradient magnitude for each set of gradients
		all_gradients = [] 
		display_num = 5
		#print(f"\n\nselected_gradients({type(included_gradients)}): {included_gradients}")

		for key, value in included_gradients.items():
			if isinstance(value, (float, int, np.float32, np.float64)):
				all_gradients.append(value)
			else:
				# Handle unexpected non-float/int values with a message
				if display_num > 0:
					display_num -= 1
					print(f"Unexpected value for {key}: {value}, type: {type(value)}")

		# Convert the list to a NumPy array for efficient computation
		if all_gradients:
			all_gradients = np.array(all_gradients)
			#print(f"all_gradients: {all_gradients}")

			# Compute the L2 norm (Euclidean norm) of the gradients
			gradient_magnitude = np.linalg.norm(all_gradients)
			if self.Debug_1: print(f"gradient_magnitude: {gradient_magnitude}")
		else:
			print("No valid gradient values found.")
			gradient_magnitude = 0
		
		return gradient_magnitude


	def min_max_normalize(self, values):
		"""
		Normalize the values using min-max normalization to the range [0, 1].
		"""
		min_val = np.min(values)
		max_val = np.max(values)
		if max_val - min_val == 0:
			return np.zeros(len(values))  # Avoid division by zero
		return (values - min_val) / (max_val - min_val)

	# Define the objective function g(a), which should be customized based on your specific use case
	def objective_function(self, a, gradients_dict, carbon_intensities_dict):
		alpha = 0.5
		# Extract the keys (client indices) from the dictionary
		included_clients = [i for i in range(len(a)) if a[i] == 1]

		if self.Debug_1: print(f"\n\nobjective_function- {a}, leading to {included_clients}")

		# Select gradients and carbon intensities for the selected clients
		#included_gradients = np.array([self.compute_gradient_magnitude(gradients_dict[i]) for i in included_clients])
		included_gradients = np.array([gradients_dict[i] for i in included_clients])
		included_carbon = np.array([carbon_intensities_dict[i] for i in included_clients])

		if self.Debug_1: print(f"included_gradients: {included_gradients}")
		if self.Debug_1: print(f"included_carbon: {included_carbon}")

		# Calculate utility as a weighted sum of gradients and carbon intensities
		utility = alpha * np.sum(included_gradients) - (1 - alpha) * np.sum(included_carbon)

		if self.Debug_1: print(f"Computed Utility: {utility}\n")
		return utility


	def randomized_double_greedy(self, local_gradients_dict, carbon_intensities_dict):
		# Initialize a_e and a_f
		a_e = np.zeros(len(local_gradients_dict), dtype=int)
		a_f = np.ones(len(local_gradients_dict), dtype=int)

		# Iteratively process each client
		for j in range(len(local_gradients_dict)):
			if self.Debug_1: print(f"\n\n\n {j}-client, the selected client group, a_e({a_e})")

			# Calculate the gain of including client j in a_e
			if self.Debug_1: print(f"\n the gain of including client j in np.append(a_e[:j], [1]) ({np.append(a_e[:j], [1])}) Vs a_e:({a_e})")
			u_j = self.objective_function(np.append(a_e[:j], [1]), local_gradients_dict, carbon_intensities_dict) - \
				  self.objective_function(a_e, local_gradients_dict, carbon_intensities_dict)

			# Calculate the loss of excluding client j from a_f
			if self.Debug_1: print(f"\n the loss of excluding client j from a_f ({a_f}) Vs np.append(a_f[:j], [0]):({np.append(a_f[:j], [0])})")
			v_j = self.objective_function(a_f, local_gradients_dict, carbon_intensities_dict) - \
				  self.objective_function(np.append(a_f[:j], [0]), local_gradients_dict, carbon_intensities_dict)

			if self.Debug_1: print(f"gain:{u_j} Vs loss({v_j})")
			# Define u_j^+ and v_j^+
			u_j_plus = max(u_j, 0)
			v_j_plus = max(v_j, 0)

			# Calculate the probability p_j
			if u_j_plus + v_j_plus > 0:
				p_j = u_j_plus / (u_j_plus + v_j_plus)
			else:
				p_j = 0.5  # If both are 0, default to 0.5 probability

			# Random decision to update a_e or a_f
			if np.random.rand() < p_j:
				a_e[j] = 1
			else:
				a_f[j] = 0

			if self.Debug_1: print(f"max gain,u_j_plus :({u_j_plus}), max loss, v_j_plus:({v_j_plus}), probaility for client j:({p_j}), selection({a_e[j]})")


		# The selected client group is determined for the aggregration of the global model.
		final_client_group = {idx: 0 for idx, val in enumerate(a_e) if val}

		#self.g_para.selected_client_group = final_client_group
		self.g_para.selected_client_group = {client_id: client.local_model
			for client_id, client in self.clients.items()
			if client_id in list(final_client_group.keys())
		}
		print(f"by_random_double_evaluation--- \n{len(self.g_para.selected_client_group)} clients are selected as {list(self.g_para.selected_client_group.keys())}")

		self.g_para.next_selected_client_group = {i: 0 for i in range(len(local_gradients_dict))}
		print(f"However, the next selected client group is reset as {len(self.g_para.next_selected_client_group)}")

	# Example usage:
	def by_random_double_evaluation(self):

		### "self.server.server_local_info[client.id]["client_gradient"]" has a dictionary type with full of gradiens, a lot, 'dense4.14.conv1.weight': 1.5531725e-05, 'dense4.14.bn2.weight': 3.5715857e-09
		local_gradients_dict = {client.id: self.server.server_local_info[client.id]["client_gradient"] for client in self.clients}

		max_carbon_value = max(self.server.server_local_info[client.id]["local_carbon_intensity"][0] for client in self.clients)

		### carbon_intensities_dict is like {0: 0.47056502006182993, 1: 0.08814049858580543, 2: 1.0, 3: 0.08814049858580543, 4: 1.0}
		carbon_intensities_dict = {client.id: self.server.server_local_info[client.id]["local_carbon_intensity"][0]/max_carbon_value for client in self.clients}

		#print(f"\n\n by_random_double_evaluation-local_gradients_dict:\n{local_gradients_dict}")
		#print(f"carbon_intensities_dict:\n{carbon_intensities_dict}\n\n")

		# Filter out non-numeric values from the local_gradients_dict
		gradient_magnitude_values = np.array([self.compute_gradient_magnitude(local_gradients_dict[client.id]) for client in self.clients])

		if len(gradient_magnitude_values) > 0:
			normalized_gradients = self.min_max_normalize(gradient_magnitude_values)
		else:
			normalized_gradients = np.array([])  # Handle case where there are no valid values

		if self.Debug_1: print(f"\nby_random_double_evaluation-gradient_magnitude_values:\n{gradient_magnitude_values}")
		if self.Debug_1: print(f"normalized_gradients:\n{normalized_gradients}\n\n")

		# Update the dictionary with normalized gradients
		local_gradients_dict = {client.id: normalized_gradients[client.id] for client in self.clients}

		# Extract and normalize the carbon intensities
		max_carbon_value = max(self.server.server_local_info[client.id]["local_carbon_intensity"][0] for client in self.clients)
		carbon_intensities_dict = {client.id: self.server.server_local_info[client.id]["local_carbon_intensity"][0] / max_carbon_value for client in self.clients}

		if self.Debug_1: print(f"Normalized Gradients: {local_gradients_dict}")
		if self.Debug_1: print(f"Normalized Carbon Intensities: {carbon_intensities_dict}")
							
		self.randomized_double_greedy(local_gradients_dict, carbon_intensities_dict)

	def determine_next_selected_clients(self):
		if self.g_para.client_select_param["criteria"] in ['acc']:
			self.by_acc_evaluation()
		elif self.g_para.client_select_param["criteria"] in ['weight']:
			self.by_weight_evaluation()
		elif self.g_para.client_select_param["criteria"] in ['pca', 'tsne', 'consine']:
			self.by_pca_tsne_evaluation()
		elif self.g_para.client_select_param["criteria"] in ['carbon']:
			self.by_pca_tsne_evaluation(mode='carbon')
			self.by_carbon_evaluation()
		elif self.g_para.client_select_param["criteria"] in ['random_double']:
			self.by_random_double_evaluation()
		elif self.g_para.client_select_param["criteria"] in ['kl']:
			self.by_kl_evaluation()

	def update_selected_client_group_by_portion(self, selected_client_group_multi_sel, c_ratio):
		selected_clients_list = list(selected_client_group_multi_sel.items())
		num_clients_to_select = int(self.g_para.num_clients * c_ratio)
		selected_clients_slice = selected_clients_list[:num_clients_to_select]
		weighting_value = 1 / len(selected_clients_slice)
		updated_selected_clients = {client_id: value + weighting_value for client_id, value in selected_clients_slice}
		self.g_para.selected_client_group = updated_selected_clients

	def full_participation(self):
		fully_selected_clients = {client_id: 1 for client_id, client in self.clients.items()}
		self.g_para.selected_client_group = fully_selected_clients