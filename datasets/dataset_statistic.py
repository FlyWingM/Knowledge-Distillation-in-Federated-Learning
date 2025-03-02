import os
import re
import json
import numpy as np
from scipy.stats import wasserstein_distance

class DataStatistic:

	def kl_divergence(self, P, Q):
		epsilon_t = 1e-10

		if len(P) == 0:
			P = np.full(10, 0.1)
			print(f"P, the global data distribution is null somehow, it set as by force")
		else:
			P = np.array(P, dtype=np.float64)
			P = P / np.sum(P)  # Ensure P is a proper probability distribution

		Q = np.array(Q, dtype=np.float64)
		Q = Q / np.sum(Q)  # Ensure Q is a proper probability distribution
		Q = np.maximum(Q, epsilon_t)

		# Calculate the KL divergence, avoid division by zero or log(0)
		return round(np.sum(np.where(P != 0, P * np.log(P / Q), 0)), 4)

	def emd_distance(self, P, Q):
		# Ensure P and Q are proper probability distributions
		P = np.array(P, dtype=np.float64)
		Q = np.array(Q, dtype=np.float64)
		P = P / np.sum(P)
		Q = Q / np.sum(Q)

		# Calculate the Earth Mover's Distance
		return round(wasserstein_distance(P, Q), 4)

	def kl_calculation(self, g_para, data_training_type, server, client, indices_per_class, amount_per_label, dataset_sizes):

		if g_para.client_select_param["criteria"] in ["acc", "pca", "tsne", "weight", "carbon"] and data_training_type == 'train' and g_para.dataset['iid_data'] <= 0:
			local_distributions = []

			for class_t in range(10):
				if class_t in indices_per_class.keys():
					local_distributions.append(amount_per_label[class_t] / dataset_sizes)
				else:
					local_distributions.append(0)

			#print(f"global:({g_para.global_info['global_distribution']}), local({local_distributions})")
			if g_para.client_select_param["distribution_similarity"] == "KL":
				server.server_local_info[client.id]["kl_div"][0] = self.kl_divergence(g_para.global_info['global_distribution'], local_distributions)
				#print(f"{client.id}-client has KL divergence: {server.server_local_info[client.id]['kl_div'][0]}")

			elif g_para.client_select_param["distribution_similarity"] == "EMD":
				server.server_local_info[client.id]["emd"][0] = self.emd_distance(g_para.global_info['global_distribution'], local_distributions)
				print(f"{client.id}-client has EMD: {server.server_local_info[client.id]['emd'][0]}")


		# Extract and save the key values from the datasets
		if g_para.Debug["data"] and g_para.Debug["d_distribution"] and data_training_type == "train":  # Or 'test', depending on the dataset type
			data_to_save = {}
			data_to_save[client.id] = amount_per_label

			file_name = f"{client.id}_{g_para.pre_distr}.json"
			if not os.path.exists(g_para.path["distribution"]):
				try:
					os.makedirs(g_para.path["distribution"])
				except OSError as e:
					print(f"Error creating directory {g_para.path['distribution']}: {e}")
			file_path = os.path.join(g_para.path["distribution"], file_name)
			# Write the data to a JSON file
			with open(file_path, 'w') as file:
				json.dump(data_to_save, file, indent=4)
			print(f"Data distribution saved to {file_path}")     
