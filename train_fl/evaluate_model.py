import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
#from train_fl.utility import progress_bar
from train_fl.utility import Utils

class ModelEvaluator:

	def __init__(self, fl_params):
		self.fl_params = fl_params
		self.g_para = fl_params.g_para
		self.server = fl_params.server
		self.clients = fl_params.clients
		self.utils = Utils()

	def cmodel_evaluate(self):
		try:
			result = [self.evaluate_model(self.server.cen_model, self.server.central_test)]
			print(f"\nThe Accuracy of centralized on Test Dataset: acc({result[0]['test_acc']:6.4f})")
			return result[0]["test_acc"]
		except Exception as e:
			print(f"An error occurred: {e}")



