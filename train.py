import numpy as np
import os
import argparse
from model import *
import audio.hparams as hparams 
from scripts.data_loader import *
from tqdm import tqdm
import librosa
import torch
import torch.optim as optim

def train(device, model, train_loader, test_loader, optimizer, epoch_resume, args):

	l1_loss = nn.L1Loss()
	cosine_sim = nn.CosineSimilarity()

	for epoch in range(epoch_resume+1, args.epochs+1):

		print("Epoch %d" %epoch)
		
		# Train
		model.train()

		# Initialize the variables to store losses
		total_mag_loss, total_phase_loss, total_loss = 0.0, 0.0, 0.0

		# Loop the train loader
		progress_bar = tqdm(enumerate(train_loader))
		for step, (x_mag, x_phase, x_image, y_mag, y_phase) in progress_bar:

			# Reset the optimizer
			optimizer.zero_grad()						

			# Transfer the data variables to the appropriate device (GPU if available)
			x_mag = x_mag.to(device)									# Bx100x257
			x_phase = x_phase.to(device)								# Bx100x257
			x_image = x_image.to(device)								# Bx25x96x96x3
			y_mag = y_mag.to(device)									# Bx100x257
			y_phase = y_phase.to(device)								# Bx100x257

			# Forward pass
			mag_output, phase_output = model(x_mag, x_phase, x_image)
			
			# Compute the magnitude and phase losses
			mag_loss = l1_loss(mag_output, y_mag)
			phase_loss = torch.mean(1.0 - cosine_sim(phase_output, y_phase))
			loss = mag_loss + phase_loss
			
			# Backward pass
			loss.backward()
			optimizer.step()

			# Keep track of the losses
			total_mag_loss += mag_loss.item()
			total_phase_loss += phase_loss.item()
			total_loss += loss.item()

			# Display the progress			
			progress_bar.set_description('Mag: %.4f, Phase: %.4f, Total loss: %.4f' % 
										(total_mag_loss / (step + 1),
										 total_phase_loss / (step + 1),
										 total_loss / (step + 1)))
			progress_bar.refresh()


		# Save the checkpoint
		if epoch % args.ckpt_freq == 0:
			save_checkpoint(model, optimizer, (total_loss/(step+1)), folder, step, epoch)

		# Validation loop to analyse the performance
		if epoch % args.validation_interval == 0:	
			validate(device, model, test_loader)


def validate(device, model, test_loader):

	print('\nEvaluating for {} steps'.format(len(test_loader)))

	l1_loss = nn.L1Loss()
	cosine_sim = nn.CosineSimilarity()

	# Validate
	model.eval()

	# Initialize the los s variable
	losses = []	

	for step, (x_mag, x_phase, x_image, y_mag, y_phase) in enumerate(test_loader):

		# Transfer the data variables to the appropriate device (GPU if available)
		x_mag = x_mag.to(device)
		x_phase = x_phase.to(device)
		x_image = x_image.to(device)
		y_mag = y_mag.to(device)
		y_phase = y_phase.to(device)

		# Generate the magnitude and phase
		with torch.no_grad():
			mag_output, phase_output = model(x_mag, x_phase, x_image)
		
		# Compute the losses
		mag_loss = l1_loss(mag_output, y_mag)
		phase_loss = torch.mean(1.0 - cosine_sim(phase_output, y_phase))
		loss = mag_loss + phase_loss
	
		# Keep track of loss
		losses.append(loss.item())

		
	# Find the average loss
	averaged_loss = sum(losses) / len(losses)
	print("Validation loss: ", averaged_loss)



def save_checkpoint(model, optimizer, train_loss, checkpoint_dir, step, epoch):

	checkpoint_path = join(
		checkpoint_dir, "checkpoint_epoch_{:05d}.pt".format(epoch))

	torch.save({
		'epoch': epoch,
		'step' : step,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': train_loss
	}, checkpoint_path)

	print("Saved checkpoint:", checkpoint_path)


def load_checkpoint(args, model, optimizer):

	if not torch.cuda.is_available():
		checkpoint = torch.load(args.checkpoint, map_location='cpu')
	else:
		checkpoint = torch.load(args.checkpoint)
	
	epoch_resume = checkpoint['epoch']
	
	s = checkpoint["model_state_dict"]
	new_s = {}
	for k, v in s.items():
		if torch.cuda.device_count() > 1:
			if not k.startswith('module.'):
				new_s['module.'+k] = v
			else:
				new_s[k] = v
		else:
			new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	if args.learning_rate is None:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loss = checkpoint['loss']

	print("Model resumed for training...")
	print("Epoch: ", epoch_resume)
	print("Loss: ", loss)

	return model, epoch_resume

def print_network(model, summary=False):

	if summary:
		print("Model summary: ", model)
	
	num_params = 0
	for param in model.parameters():
		num_params += param.numel()

	print('Total number of parameters: %f M' % (int(num_params)/1000000.0))
	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-i', '--img_size', type=int, default=96, required=False, \
						help='Input image size')
	parser.add_argument('-nf', '--num_frames', type=int, default=25, required=False, \
						help='Num of frames to consider')
	parser.add_argument('-b', '--batch_size', type=int, default=16, required=False, \
						help='Batch size to train the model')
	parser.add_argument('-e', '--epochs', type=int, default=500000, required=False, \
						help='No of epochs to train the model')
	parser.add_argument('-w', '--num_workers', type=int, default=16, help='No of workers')
	parser.add_argument('-g', '--n_gpu', type=int, default=1, required=False, help='No of GPUs')
	parser.add_argument('-cf', '--ckpt_freq', type=int, default=1, required=False, \
						help='Frequency of saving the model')
	parser.add_argument('-ckpt', '--checkpoint', default=None, \
						help='Path of the pre-trained model checkpoint to resume training')
	parser.add_argument('-md', '--model_directory', default='saved_models/', \
						help='Path to save the model')
	parser.add_argument('-lr', '--learning_rate', default=None, type=float,\
						help='learning rate')
	parser.add_argument('-vi', '--validation_interval', default=2, type=int, \
						help='Validation interval')
			
	args = parser.parse_args()


	args.batch_size = args.n_gpu * args.batch_size	
	hp = hparams.hparams

	# Load the train and test data
	train_loader = load_data(img_size=args.img_size, num_frames=args.num_frames, hparams=hp, \
							num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

	total_batch = len(train_loader)
	print("Total train batch: ", total_batch)

	test_loader = load_data(img_size=args.img_size, num_frames=args.num_frames, hparams=hp, \
							num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, test=True)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize the model
	model = Model()

	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs for model!")
		model = nn.DataParallel(model)
	else:
		print("Using single GPU for model!")
	model.to(device)

	# Print the model and its parameters
	print_network(model)

	# Set the optimizer
	if args.learning_rate is not None:
		lr = args.learning_rate
	else:
		lr = hp.initial_learning_rate

	optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

	# Resume the pre-trained model (if available) 
	epoch_resume = 0
	if args.checkpoint is not None:
		model, epoch_resume = load_checkpoint(args, model, optimizer)

	# Create the folder to save the checkpoints
	folder = args.model_directory
	if not os.path.exists(folder):
		os.makedirs(folder)

	# Train!
	train(device, model, train_loader, test_loader, optimizer, epoch_resume, args)

	print("Finished")


	
