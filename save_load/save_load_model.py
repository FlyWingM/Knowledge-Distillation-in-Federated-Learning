import torch
import torch.cuda
import os
#import util.utility as util 
from save_load.save_results import generate_file_suffix


def construct_model_file_path(g_para, suffix, is_high_accuracy=False, is_directory=False):
	"""Constructs a file path for the model based on given parameters."""
	base_path = f"{g_para.output_dir[g_para.version]}/{g_para.data_name}/{g_para.data_distribution_type}"

	if g_para.data_distribution_type =='Dirichlet':
		base_path = f"{base_path}/{g_para.pre_distr}/{g_para.avg_algo}"
	elif g_para.data_distribution_type in ['rl']:
		base_path = f"{base_path}/{g_para.pre_distr}"        
	elif g_para.data_distribution_type in ['IID', 'IID_rl']:
		base_path = f"{base_path}/{g_para.avg_algo}"
	else: 	# 'central' and else
		base_path = f"{base_path}"	

	if suffix['addition_info'] == "high_capacity":
		file_suffix = generate_file_suffix(g_para, 'gmodel_name_server')
	else:
		file_suffix = generate_file_suffix(g_para, 'gmodel')

	acc = 'H_' if is_high_accuracy else ''
	
	s_train = f"_ST{g_para.rl['s_train_counter']}"
	m_explore = f"_ER{g_para.rl['m_exploration_counter']}"
	s_explore = f"_ET{g_para.rl['exploitation_counter']}"

	addition_info = f"_{suffix['addition_info']}" if suffix.get('addition_info') else ''
	file_path = (f"{file_suffix}_i{suffix['iter']}{s_train}{m_explore}{s_explore}{addition_info}_acc{acc}{suffix['acc']:.4f}.pth")

	if is_directory:
		return base_path
	else:   
		return os.path.join(base_path, file_path) 


def create_directory_if_not_exists(directory):
	"""Create a directory if it doesn't already exist."""
	if not os.path.exists(directory):
		os.makedirs(directory)


def gmodel_save(g_para, server, global_ind, mode=''):
	acc_new_t = g_para.global_info["g_acc"][global_ind]
	save_directory = construct_model_file_path(g_para, {'iter': 0, 'acc': 0, 'addition_info': mode}, is_high_accuracy=False, is_directory=True)
	create_directory_if_not_exists(save_directory)

	is_high_accuracy = g_para.global_info["highest_acc"] <= acc_new_t or g_para.global_info["highest_iter"] == global_ind
	#TMP
	#suffix = {'iter': global_ind, 'acc': acc_new_t, 'addition_info':mode}
	suffix = {'iter': global_ind+g_para.cont_tr_iter, 'acc': acc_new_t, 'addition_info':mode}
	save_path = construct_model_file_path(g_para, suffix, is_high_accuracy=is_high_accuracy)

	directory = os.path.dirname(save_path)
	# Check if the directory exists, if not try to create it
	if not os.path.exists(directory):
		try:
			os.makedirs(directory)
		except OSError as e:
			print(f"Error creating directory {directory}: {e}")

	# Now try to save the model
	try:
		if mode == "high_capacity":
			torch.save(server.g_model_big.state_dict(), save_path)
			print(f"\nhigh_capacity Model successfully saved to {save_path}")
		else:
			torch.save(server.g_model.state_dict(), save_path)
			print(f"\nModel successfully saved to {save_path}")

	except Exception as e:
		print(f"Failed to save the model: {e}")

	if suffix.get('addition_info') == 'stagnation':
		return  #Without removing the previous ones, it keeps running

	if is_high_accuracy: 
		if os.path.exists(g_para.global_info["h_gmodel_name"]): 
			os.remove(g_para.global_info["h_gmodel_name"])
		g_para.global_info["h_gmodel_name"] = save_path
		print(f"\n==The best accuracy({acc_new_t:.4f}) has been achieved at the {global_ind} global iteration")
	else:
		if os.path.exists(g_para.global_info["gmodel_name"]): 
			os.remove(g_para.global_info["gmodel_name"])
		g_para.global_info["gmodel_name"] = save_path


def construct_global_model_filename(g_para):
	"""Loads the global model."""

	if g_para.nn['cont_train_filenames']:
		if len(g_para.nn['cont_train_filenames']) == 1:
			global_model_filename = f"{g_para.trained_model_dir}/{g_para.nn['cont_train_filenames'][0]}"
			print(f"construct_global_model_filename: {global_model_filename}")	
		else:
			global_model_filename = f"{g_para.trained_model_dir}/{g_para.nn['cont_train_filenames'][0]}"
			print(f"construct_global_model_filename with the first one out of serveral: {global_model_filename}")
	else:
		load_directory = construct_model_file_path(g_para, {'iter': g_para.cont_tr_iter, 'acc': g_para.cont_tr_acc}, is_high_accuracy=False, is_directory=True)
		if not os.path.exists(load_directory):
			raise FileNotFoundError(f"Loading the global model fails in the non-existing directory, {load_directory}")
		global_model_filename = construct_model_file_path(g_para, {'iter': g_para.cont_tr_iter, 'acc': g_para.cont_tr_acc}, is_high_accuracy=False)
		print(f"==The previous model with the {g_para.cont_tr_acc:.4f} and {g_para.cont_tr_iter} global iteration is being loaded")

	return global_model_filename


def construct_central_model_file_path(csv_path, g_para, epoch, accuracy):
	"""Constructs a file path for the central model based on given parameters."""
	return os.path.join(csv_path, f"{g_para.nn['name']}-{'T' if g_para.nn['pretrained'] else 'F'}-({g_para.numEpoch}-{epoch})-{accuracy:.4f}.pth")


def cmodel_save(g_para, server, epoch): 
	csv_path = os.path.join(g_para.output_dir[g_para.version], g_para.data_name, 'central')
	create_directory_if_not_exists(csv_path)

	save_path = construct_central_model_file_path(csv_path, g_para, epoch, g_para.c_global_info['central_acc'][epoch])

	directory = os.path.dirname(save_path)
	# Check if the directory exists, if not try to create it
	if not os.path.exists(directory):
		try:
			os.makedirs(directory)
		except OSError as e:
			print(f"Error creating directory {directory}: {e}")

	# Now try to save the model
	try:
		torch.save(server.cen_model.state_dict(), save_path)
		print(f"Model successfully saved to {save_path}")
	except Exception as e:
		print(f"Failed to save the model: {e}")

	if g_para.c_global_info['central_acc'][epoch] > g_para.c_global_info['c_highest_acc']:
		# Delete the previous highest-accuracy model
		previous_model = construct_central_model_file_path(csv_path, g_para, g_para.c_global_info['c_highest_iter'], g_para.c_global_info['c_highest_acc'])
		if os.path.exists(previous_model):
			os.remove(previous_model)

		g_para.c_global_info['c_highest_iter'] = epoch
		g_para.c_global_info['c_highest_acc'] = g_para.c_global_info['central_acc'][epoch]
		print(f"***The best accuracy has been achieved by {g_para.c_global_info['c_highest_acc']:.4f} at the {epoch} iteration\n")
	else:
		# Delete the previous model every 5 epoch
		previous_model = construct_central_model_file_path(csv_path, g_para, g_para.c_global_info['c_prev_iter'], g_para.c_global_info['c_prev_acc'])
		if os.path.exists(previous_model):
			os.remove(previous_model)

		g_para.c_global_info['c_prev_iter'] = epoch    
		g_para.c_global_info['c_prev_acc'] = g_para.c_global_info['central_acc'][epoch]

def cmodel_load(g_para):
	csv_path = os.path.join(g_para.output_dir[g_para.version], g_para.data_name, 'central')
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"Loading the centralized model fails in the non-existing directory, {csv_path}")

	loading_path = construct_central_model_file_path(csv_path, g_para, g_para.cont_tr_iter, g_para.cont_tr_acc)
	return loading_path

