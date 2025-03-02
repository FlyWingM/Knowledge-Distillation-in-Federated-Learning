### train_cen_model

import torch
from train_fl.utility import Utils

class Model_Cen_Trainer:

	def __init__(self, fl_params):
		self.fl_params = fl_params
		self.g_para = fl_params.g_para
		self.server = fl_params.server
		self.utils = Utils()

	def train_epoch(self, epoch):
		if self.g_para.Debug["d_train"]:
			print(f"Epoch: {epoch+1}/{self.g_para.numEpoch}")

		self.server.cen_model.train()
		train_loss, correct, total = 0, 0, 0

		for batch_idx, (inputs, targets) in enumerate(self.server.central_train):
			inputs, targets = inputs.to(self.g_para.device), targets.to(self.g_para.device)
			self.server.cen_optimizer.zero_grad()
			outputs = self.server.cen_model(inputs)

			loss = self.server.cen_criterion(outputs, targets)
			loss.backward()

			self.server.cen_optimizer.step()
			self.server.cen_scheduler.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			if not self.g_para.environ["SBATCH"] and self.g_para.Debug["d_train"] and self.g_para.Debug["d_lr"]:
				self.utils.progress_bar(
					"Training",
					batch_idx,
					len(self.server.central_train),
					f'Loss: {train_loss/(batch_idx+1):.3f} | '
					f'Acc: {100.*correct/total:.3f}% ({correct}/{total})'
				)

		return {'train_acc': correct/total, 'train_loss': train_loss/(batch_idx+1)}

	def validate_epoch(self, epoch):
		self.server.cen_model.eval()
		valid_loss, correct, total = 0, 0, 0

		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(self.server.central_valid):
				inputs, targets = inputs.to(self.g_para.device), targets.to(self.g_para.device)
				outputs = self.server.cen_model(inputs)
				loss = self.server.cen_criterion(outputs, targets)

				valid_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

				if not self.g_para.environ["SBATCH"] and self.g_para.Debug["train"] and self.g_para.Debug["d_lr"]:
					self.utils.progress_bar(
						"Validation",
						batch_idx,
						len(self.server.central_valid),
						f'Loss: {valid_loss/(batch_idx+1):.3f} | '
						f'Acc: {100.*correct/total:.3f}% ({correct}/{total})'
					)

		return {'val_acc': correct/total, 'val_loss': valid_loss/(batch_idx+1)}

	def train_cen_model(self):
		history = []

		for epoch in range(self.g_para.numEpoch):
			train_result = self.train_epoch(epoch)
			val_result = self.validate_epoch(epoch)
			result = {**train_result, **val_result}
			history.append(result)

			print(
				f"Epoch {epoch+1}/{self.g_para.numEpoch}, "
				f"TrainLoss: {result['train_loss']:.4f}, "
				f"TrainAcc: {result['train_acc']:.4f}, "
				f"ValidLoss: {result['val_loss']:.4f}, "
				f"ValidAcc: {result['val_acc']:.4f}"
			)

		return history
