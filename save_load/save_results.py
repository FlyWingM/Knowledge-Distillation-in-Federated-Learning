import os
import numpy as np
import pandas as pd
import util.utility as util 

def generate_file_suffix(g_para, mode, client_name=None):
	if mode in ["central_acc", "global_model_results", "carbon", "per_model_results", "local_model_results", "gmodel", "gmodel_name_server"]:
		# Individual components of the suffix
		if mode == "gmodel_name_server":
			name = f"{g_para.nn['name_server']}"
		else:
			name = f"{g_para.nn['name']}"

		pretrained_status = 'T' if g_para.nn['pretrained'] else 'F'
		trainable_status = 'T' if g_para.nn['trainable'] else 'F'
		pre = f"P{pretrained_status}{trainable_status}"

		d_key = g_para.data_distribution_type   # central, IID, IID_rl, Dirichlet, Dirich_rl
		distribution_type = 'Dir' if d_key == 'Dirichlet' else 'rl' if d_key == 'Dirich_rl' else d_key
		data_size = "B" if g_para.data_name in ["cifar10_big", "cinic10_big"] else "b"
		sdata_type = {"shared_data": "rd", "shared_data_synthe": "ls", "shared_data_synthe_progressive_GAN": "qs"}.get(g_para.shared_dir, "ns")
		
		data = f"{data_size}_{sdata_type}_{distribution_type}_{g_para.pre_distr}_iid{g_para.dataset['iid_data']}_{g_para.client_select_param['distribution_similarity']}_{g_para.dataset['data_aug']}"
		if distribution_type == 'rl' and g_para.pre_distr == 'IID':
			data = f"{data_size}_{sdata_type}_IIDrl_iid{g_para.dataset['iid_data']}_{g_para.client_select_param['distribution_similarity']}_{g_para.dataset['data_aug']}"

		avg_algo = f"{g_para.avg_algo}"
		#iter_config = f"{g_para.num_g_iter}-{g_para.numEpoch}-{g_para.cont_tr_iter}"
		iter_config = f"i{g_para.num_g_iter}_e{g_para.numEpoch}"
		num_clients = f"c{g_para.num_clients}"

		client_select_criteria = g_para.client_select_param['criteria']
		weighting_type = g_para.client_select_param['weighting']
		weighting = 'WE' if weighting_type == 'weighting_exponential' else 'WI' if weighting_type == 'weighting_inverse' else weighting_type

		client_ratio_num = g_para.client_select_param.get('c_ratio', 1)
		client_ratio = f"CR{client_ratio_num}"
		client_selection = f"{g_para.client_select_param['learning_exploration']}"
		selection_primary_num = int(g_para.rl['primary_ratio']*100)
		selection_primary = f"P{selection_primary_num}"
		dynamic_flag = 'T' if g_para.client_select_param.get('dynamic_sel') else 'F'        
		dynamic = f"d{dynamic_flag}"

		carbon_flag = 'T' if g_para.client_select_param['lr_carbon'] else 'F'
		carbon = f"c{carbon_flag}"

		selection = f"{client_select_criteria}_{client_selection}_{selection_primary}_{weighting}_{client_ratio}_{carbon}_{dynamic}"

		# Assemble all parts into the list
		suffix_parts = [
			name, pre, data, avg_algo, iter_config, num_clients, selection
		]

		if mode == "global_model_results":
			suffix_parts.extend([
				#f"lr({g_para.h_param['learning_rate']})",
			])
			return "_".join(suffix_parts) + ".csv"

		elif mode == "carbon":
			suffix_parts.extend([
				"carbon"
			])
			return "_".join(suffix_parts) + ".csv"

		elif mode == "per_model_results":
			suffix_parts.append("per_train_by_lmodel")
			return "_".join(suffix_parts) + ".csv"

		elif mode == "local_model_results":
			suffix_parts.extend([
				client_name
			])
			return "_".join(suffix_parts) + ".csv"

		elif mode in ["gmodel", "gmodel_name_server"]:
			suffix_parts.extend([
				#f"lr({g_para.h_param['learning_rate']})"
			])
			return "_".join(suffix_parts)

		elif mode == "central_acc":
			return "_".join(suffix_parts) + ".csv"

	else:
		return "Unsupported mode from generate_file_suffix"


def construct_merged_data_path(g_para, dir_name, client_n, iid=None, shared_suffix=None):
	if shared_suffix is None:
		path = os.path.join(g_para.path[g_para.data_name], f'{g_para.pre_distr}-{g_para.num_clients}', f'client-{client_n}/')
	elif shared_suffix is not None and shared_suffix in ['cinic10_orig/']:
		path = os.path.join(g_para.path[g_para.data_name], shared_suffix)
	else:
		path = os.path.join(g_para.path[g_para.data_name], f'{g_para.pre_distr}-{g_para.num_clients}', shared_suffix)
	return path


def construct_csv_path(g_para, additional_path='', file_suffix=''):
	base_path = f"{g_para.output_dir[g_para.version]}/{g_para.data_name}/{g_para.data_distribution_type}"

	if g_para.data_distribution_type =='Dirichlet':
		base_path = f"{base_path}/{g_para.pre_distr}/{g_para.avg_algo}"
	elif g_para.data_distribution_type in ['rl']:
		base_path = f"{base_path}/{g_para.pre_distr}"        
	elif g_para.data_distribution_type in ['IID', 'IID_rl']:
		base_path = f"{base_path}/{g_para.avg_algo}"
	else: 	# 'central' and else
		base_path = f"{base_path}"

	if additional_path:
		base_path = f"{base_path}/{additional_path}"
	util.mkdir_p(base_path)
	return os.path.join(base_path, file_suffix)


def save_dataframe_to_csv(data_frame, csv_file, g_para):
	try:
		data_frame.to_csv(csv_file, mode='w')
	except PermissionError as e:
		print(f"Permission error: {e}")
	except Exception as e:
		print(f"Unexpected error: {e}")


def save_results(g_para, server=None, clients=None, mode=None):

	if mode == "central_acc":
		file_suffix = generate_file_suffix(g_para, mode)
		csv_file = construct_csv_path(g_para, file_suffix=file_suffix)
		DF_central = pd.DataFrame(np.array(g_para.c_global_info['central_acc']))
		save_dataframe_to_csv(DF_central, csv_file, g_para)

	elif mode == "global_model_results":
		file_suffix = generate_file_suffix(g_para, mode)
		csv_file = construct_csv_path(g_para, file_suffix=file_suffix)

		# Create a dictionary where each key-value pair will become a column in the DataFrame
		data = {
			'g_acc': g_para.global_info["g_acc"],
			'l_train_time': g_para.global_info["total_local_train_time"],
			'l_carbon': g_para.global_info["local_carbon_total"],
			'l_kl_avg': g_para.global_info["local_kl_avg"],
			'l_entropy_ratio': g_para.global_info["entropy_ratio"],
			'sel_client_num': g_para.global_info["client_sel_n"]
		}
		DF_global = pd.DataFrame(data)
		save_dataframe_to_csv(DF_global, csv_file, g_para)
  
	elif mode == "carbon":        
		file_suffix = generate_file_suffix(g_para, mode)
		csv_file = construct_csv_path(g_para, file_suffix=file_suffix)
		data = {client.id: server.server_local_info[client.id]["local_carbon"] for client in clients}
		DF_global = pd.DataFrame(data.items(), columns=['client_name', 'carbon_value'])
		save_dataframe_to_csv(DF_global, csv_file, g_para)


	elif mode == "local_model_results":
		additional_path = "local_models"
		for client in clients:
			local_name = f"client{client.id}"
			file_suffix = generate_file_suffix(g_para, mode, client_name=local_name)

			client_local_info = server.server_local_info[client.id]

			# Extracting values
			local_acc = client_local_info["local_acc"]
			length = len(local_acc)  # Get the length of local_acc

			global_acc = g_para.global_info["g_acc"][:length]
			local_training_time = client_local_info["local_train_time"][:length]  # Adjust to match the length of local_acc
			local_pca = client_local_info["local_pca"][:length]  
			local_weight = client_local_info["local_weight"][:length] 
			local_carbon = client_local_info["local_carbon"][:length]  
			local_carbon_intensity = client_local_info["local_carbon_intensity"][:length]  
			local_state_1 = client_local_info["local_state_1"][:length]  
			local_state_2 = client_local_info["local_state_2"][:length]  
			local_new_state_1 = client_local_info["local_new_state_1"][:length]  
			local_new_state_2 = client_local_info["local_new_state_2"][:length]  
			avg_primary = client_local_info["avg_primary"][:length]  
			avg_carbon = client_local_info["avg_carbon"][:length]  
			primary_reward = client_local_info["primary_reward"][:length]  
			carbon_reward = client_local_info["carbon_reward"][:length]  
			local_reward = client_local_info["local_reward"][:length]  
			local_select = client_local_info["local_select"][:length]  
			local_data_size = server.server_local_info[client.id]['train_num_samples'] * length  # Repeat the single element to match the length of local_acc
			kl_div = client_local_info["kl_div"] * length  
			emd = client_local_info["emd"] * length  
			#country = client_local_info["country_carbon"] * length  

			data = {
				'g_acc': global_acc,
				'local_acc': local_acc,
				#'country': country,
				'local_training_time': local_training_time,
				'local_pca': local_pca,
				'local_weight': local_weight,
				'local_carbon': local_carbon,
				'local_carbon_intensity': local_carbon_intensity,
				'local_state_1': local_state_1,
				'local_state_2': local_state_2,
				'local_new_state_1': local_new_state_1,
				'local_new_state_2': local_new_state_2,
				'avg_primary': avg_primary,
				'avg_carbon': avg_carbon,
				'primary_reward': primary_reward,
				'carbon_reward': carbon_reward,
				'local_reward': local_reward,
				'local_select': local_select,
				'local_data_size': local_data_size,
				'kl_div': kl_div,
				'emd': emd,
			}
			DF_local = pd.DataFrame(data)

			local_csv_file = construct_csv_path(g_para, additional_path, file_suffix=file_suffix)
			save_dataframe_to_csv(DF_local, local_csv_file, g_para)

