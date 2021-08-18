import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

class Conv1d(nn.Module):
	"""Extended nn.Conv1d for incremental dilated convolutions
	"""

	def __init__(self, cin, cout, kernel_size, stride=1, padding=1, residual=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv_block = nn.Sequential(
							nn.Conv1d(cin, cout, kernel_size, stride, padding),
							nn.BatchNorm1d(cout)
							)
		self.act = nn.ReLU()
		self.residual = residual

	def forward(self, x):
		out = self.conv_block(x)
		if self.residual:
			out += x
		return self.act(out)


class Conv2d(nn.Module):
	def __init__(self, cin, cout, kernel_size, stride=1, padding=1, residual=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv_block = nn.Sequential(
							nn.Conv2d(cin, cout, kernel_size, stride, padding),
							nn.BatchNorm2d(cout)
							)
		self.act = nn.ReLU()
		self.residual = residual

	def forward(self, x):
		out = self.conv_block(x)
		if self.residual:
			out += x
		return self.act(out)


class Conv3d(nn.Module):
	"""Extended nn.Conv1d for incremental dilated convolutions
	"""

	def __init__(self, cin, cout, kernel_size, stride=1, padding=1, residual=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv_block = nn.Sequential(
							nn.Conv3d(cin, cout, kernel_size, stride, padding),
							nn.BatchNorm3d(cout)
							)
		self.act = nn.ReLU()
		self.residual = residual

	def forward(self, x):
		out = self.conv_block(x)
		if self.residual:
			try:
				out += x
			except:
				print(out.size())
				print(x.size())
		return self.act(out)

class Conv1dTranspose(nn.Module):
	"""Extended nn.Conv1d for incremental dilated convolutions
	"""

	def __init__(self, cin, cout, kernel_size, stride, padding=1, output_padding=1, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv_block = nn.Sequential(
							nn.ConvTranspose1d(cin, cout, kernel_size, stride, padding, output_padding),
							nn.BatchNorm1d(cout)
							)
		self.act = nn.ReLU()

	def forward(self, x):
		out = self.conv_block(x)
		return self.act(out)

class Conv2dTranspose(nn.Module):
	"""Extended nn.Conv1d for incremental dilated convolutions
	"""

	def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv_block = nn.Sequential(
							nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
							nn.BatchNorm2d(cout)
							)
		self.act = nn.ReLU()

	def forward(self, x):
		out = self.conv_block(x)
		return self.act(out)


class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()


		self.audio_encoder = nn.Sequential(
			Conv1d(257, 600, kernel_size=3, stride=1),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1, residual=True),
			Conv1d(600, 600, kernel_size=3, stride=1)
		) 


		self.face_encoder = nn.Sequential(
			Conv2d(3, 32, kernel_size=5, stride=(2,2), padding=2),             # Bx32x5x48x48
			Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),      

			Conv2d(32, 64, kernel_size=3, stride=(2,2), padding=1),            # Bx64x5x24x24
			Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

			Conv2d(64, 128, kernel_size=3, stride=(2,2), padding=1),           # Bx128x5x12x12
			Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

			Conv2d(128, 256, kernel_size=3, stride=(2,2), padding=1),          # Bx256x5x6x6
			Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

			Conv2d(256, 512, kernel_size=3, stride=(2,2), padding=1),          # Bx512x5x3x3
			Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
			
			Conv2d(512, 512, kernel_size=3, stride=(3,3), padding=1),          # Bx512x5x1x1
			Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
		)

		
		self.time_upsampler = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='nearest'),
			Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.Upsample(scale_factor=2, mode='nearest'),
			Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
		)

		self.mag_decoder = nn.Sequential(
			Conv1d(1112, 1024, kernel_size=3, stride=1),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			nn.Conv1d(1024, 257, kernel_size=1, stride=1, padding=0)
		)

		self.phase_decoder = nn.Sequential(
			Conv1d(1026, 1024, kernel_size=3, stride=1),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			Conv1d(1024, 1024, kernel_size=3, stride=1, residual=True),
			nn.Conv1d(1024, 257, kernel_size=1, stride=1, padding=0)
		)


	def forward(self, mag, phase, face_sequence):	

		# -------------------------- Face model ------------------------------- #
		# print("Face input: ", face_sequence.size())						# Bx25x96x96x3

		face_sequence = face_sequence.permute(0, 1, 4, 2, 3)				# Bx25x3x96x96

		B = face_sequence.size(0)
		face_sequence = torch.cat([face_sequence[:, i] for i in range(face_sequence.size(1))], dim=0)
		# print("Reshaped face sequence: ", face_sequence.size())			# (B*25)x3x96x96

		face_enc = self.face_encoder(face_sequence)							# (B*25)x512x1x1

		face_enc = torch.split(face_enc, B, dim=0) 
		face_enc = torch.stack(face_enc, dim=2) 
		# print("Face encoder output: ", face_enc.size())					# Bx512x25x1x1

		face_enc = face_enc.view(-1, face_enc.size(1), face_enc.size(2))	# Bx512x25

		face_output = self.time_upsampler(face_enc)
		# print("Face output: ", face_output.size())						# Bx512x100
		# -------------------------------------------------------------------- #

		# -------------------------- Mag model ------------------------------- #
		# print("Mag input: ", mag.size())							# Bx100x257		

		mag_permuted = mag.permute(0, 2, 1)							# Bx257x100

		mag_enc = self.audio_encoder(mag_permuted)					# Bx600x100

		# Concatenate face network output and magnitude network output
		concatenated = torch.cat([mag_enc, face_output], dim=1)		# Bx1112x100

		mask = self.mag_decoder(concatenated)						# Bx257x100

		mask = mask.permute(0, 2, 1)								# Bx100x257

		mag_output = mask + mag
		mag_output = torch.sigmoid(mag_output)
		# print("Magnitude output: ", mag_output.size())			# Bx100x257
		# -------------------------------------------------------------------- #

		
		# -------------------------- Phase model ----------------------------- #
		# print("Phase input: ", phase.size())						# Bx100x257

		face_output = face_output.permute(0,2,1)
		concatenated = torch.cat([phase, mag_output, face_output], dim=2)
		concatenated = concatenated.permute(0, 2, 1)
		# print("Concatenated: ", concatenated.size())				# Bx1026x100

		mask = self.phase_decoder(concatenated)						# Bx257x100

		mask = mask.permute(0, 2, 1)								# Bx100x257

		phase_output = mask + phase
		phase_output = torch.sigmoid(phase_output)
		# print("Phase output: ", phase_output.size())				# Bx100x257
		# -------------------------------------------------------------------- #

		return mag_output, phase_output