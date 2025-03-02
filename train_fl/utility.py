import sys
import time
import numpy as np
import torch

class Utils:
	TERM_WIDTH = 100
	TOTAL_BAR_LENGTH = 20.0

	def __init__(self):
		self.last_time = time.time()
		self.begin_time = self.last_time

	def progress_bar(self, model_n, current, total, msg=None):
		if current == 0:
			self.begin_time = time.time()  # Reset for new bar.
		sys.stdout.write(model_n)

		cur_len = int(self.TOTAL_BAR_LENGTH * current / total)
		rest_len = int(self.TOTAL_BAR_LENGTH - cur_len) - 1

		sys.stdout.write(' [')
		for i in range(cur_len):
			sys.stdout.write('=')
		sys.stdout.write('>')
		for i in range(rest_len):
			sys.stdout.write('.')
		sys.stdout.write(']')

		cur_time = time.time()
		step_time = cur_time - self.last_time
		self.last_time = cur_time
		tot_time = cur_time - self.begin_time

		L = []
		L.append('  Step: %s' % self.format_time(step_time))
		L.append(' | Tot: %s' % self.format_time(tot_time))
		if msg:
			L.append(' | ' + msg)

		msg = ''.join(L)
		sys.stdout.write(msg)
		for i in range(self.TERM_WIDTH - int(self.TOTAL_BAR_LENGTH) - len(msg) - 3):
			sys.stdout.write(' ')

		# Go back to the center of the bar.
		for i in range(self.TERM_WIDTH - int(self.TOTAL_BAR_LENGTH / 2) + 2):
			sys.stdout.write('\b')
		sys.stdout.write(' %d/%d ' % (current + 1, total))

		if current < total - 1:
			sys.stdout.write('\r')
		else:
			sys.stdout.write('\n')
		sys.stdout.flush()

	@staticmethod
	def format_time(seconds):
		days = int(seconds / 3600 / 24)
		seconds = seconds - days * 3600 * 24
		hours = int(seconds / 3600)
		seconds = seconds - hours * 3600
		minutes = int(seconds / 60)
		seconds = seconds - minutes * 60
		secondsf = int(seconds)
		seconds = seconds - secondsf
		millis = int(seconds * 1000)

		f = ''
		i = 1
		if days > 0:
			f += str(days) + 'D'
			i += 1
		if hours > 0 and i <= 2:
			f += str(hours) + 'h'
			i += 1
		if minutes > 0 and i <= 2:
			f += str(minutes) + 'm'
			i += 1
		if secondsf > 0 and i <= 2:
			f += str(secondsf) + 's'
			i += 1
		if millis > 0 and i <= 2:
			f += str(millis) + 'ms'
			i += 1
		if f == '':
			f = '0ms'
		return f

	@staticmethod
	def cal_elements(values, num_display=2):
		# Handle dictionary
		if isinstance(values, dict):
			elements = {k: values[k] for i, k in enumerate(values) if i < num_display}

			# Handle NumPy arrays
		elif isinstance(values, list):
			if len(values) < num_display:
				elements = values[:]
			else:
				elements = values[:num_display]

		# Handle NumPy arrays
		elif isinstance(values, np.ndarray):
			if values.size < num_display:
				elements = values[:]
			else:
				elements = values[:num_display]
		
		# Handle PyTorch tensors
		elif isinstance(values, torch.Tensor):

			# Flatten if values are multi-dimensional (2D or higher)
			if values.dim() > 1:
				flattened_values = values.view(-1)  # Flatten the tensor
			else:
				flattened_values = values

			# Handle cases where the number of elements is less than num_display
			if flattened_values.numel() < num_display:
				elements = flattened_values[:]
			else:
				elements = flattened_values[:num_display]		
		# Unsupported type
		else:
			raise TypeError("Unsupported type for 'values'. Expected dict, np.ndarray, or torch.Tensor.")
		
		return elements

