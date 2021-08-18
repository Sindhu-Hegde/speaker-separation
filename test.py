from os import listdir, path
import numpy as np
import scipy, cv2, os, argparse, audio
import subprocess
import librosa
from tqdm import tqdm
import audio.audio_utils as audio
from model import *
import audio.hparams as hp 
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 	


# Function to extract the frames from the video
def get_frames(file):

	video_stream = cv2.VideoCapture(file)
		
	frames = []
	while 1:
		still_reading, frame = video_stream.read()

		if not still_reading:
			video_stream.release()
			break

		# Mask out the specified regions
		if args.mask == 'r':
			index = frame.shape[1]//2
			frame = frame[:, :index]
		elif args.mask == 'l':
			index = frame.shape[1]//2
			frame = frame[:, index:]

		frames.append(frame)

	return frames


# Functon to load the wav file
def load_wav(args):

	wav_file  = 'tmp.wav';

	subprocess.call('ffmpeg -hide_banner -loglevel panic -threads 1 -y -i %s -async 1 -ac 1 -vn \
					-acodec pcm_s16le -ar 16000 %s' % (args.input, wav_file), shell=True)

	wav = audio.load_wav(wav_file, 16000)

	os.remove("tmp.wav")

	return wav


# Function to extract the spectrogram from the wav file
def get_spec(wav):

	# Extract the STFT
	stft = librosa.stft(y=wav, n_fft=hp.hparams.n_fft, hop_length=hp.hparams.hop_size, win_length=hp.hparams.win_size).T
	stft = stft[:-1]
	# print("STFT: ", stft.shape)                                       # 100x257

	# Decompose STFT into magnitude and phase components
	mag = np.abs(stft)
	mag = audio.db_from_amp(mag)
	phase = audio.angle(stft)

	# Normalize the magnitude and phase components
	norm_mag = audio.normalize_mag(mag)
	norm_phase = audio.normalize_phase(phase)
		
	# Concatenate the magnitude and phase components
	spec = np.concatenate((norm_mag, norm_phase), axis=1)               # 100x514
	
	return spec


# Function to segment the spectrograms
def get_window_spec(spec_ip, idx):
	
	frame_num = idx
	start_idx = int((hp.hparams.spec_step_size / hp.hparams.fps) * frame_num)
	end_idx = start_idx+hp.hparams.spec_step_size
	
	spec_window = spec_ip[start_idx:end_idx, :]						# 100x514

	return spec_window


# Function to obtain the window of images
def get_window_images(window_images):
	window = []
	for img in window_images:
		if img is None:
			raise FileNotFoundError('Missing frames!')

		img = cv2.resize(img, (hp.hparams.img_size, hp.hparams.img_size))		# 3x96x96
		window.append(img)	

	x_image = np.asarray(window) / 255. 

	return x_image


# Function to reconstruct the audio and generate the output video 
def generate_video(mag, phase, args):

	denorm_mag = audio.unnormalize_mag(mag)
	denorm_phase = audio.unnormalize_phase(phase)
	recon_mag = audio.amp_from_db(denorm_mag)
	complex_arr = audio.make_complex(recon_mag, denorm_phase)
	wav = librosa.istft(complex_arr, hop_length=hp.hparams.hop_size, \
						win_length=hp.hparams.win_size)
	print("Generated wav: ", wav.shape)

	
	# Create the folder to save the results
	result_dir = args.result_dir
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# Save the wav file
	audio_output = os.path.join(result_dir, 'pred_'+args.input.rsplit('/')[-1].split('.')[0] + '.wav')
	librosa.output.write_wav(audio_output, wav, 16000)

	# Save the video output file
	no_sound_video = os.path.join(result_dir, args.input.rsplit('/')[-1].split('.')[0] + '_nosouund.mp4')
	subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -c copy -an -strict -2 %s' % (args.input, no_sound_video), shell=True)

	video_output_mp4 = os.path.join(result_dir, 'pred_'+args.input.rsplit('/')[-1].split('.')[0] + '.mp4')
	if os.path.exists(video_output_mp4):
		os.remove(video_output_mp4)
	
	subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 %s' % 
					(audio_output, no_sound_video, video_output_mp4), shell=True)

	os.remove(no_sound_video)

	print("Successfully generated the output video: ", video_output_mp4)


# Function to load the model
def load_model(args):

	model = Model()

	if not torch.cuda.is_available():
		checkpoint = torch.load(args.model_ckpt, map_location='cpu')
	else:
		checkpoint = torch.load(args.model_ckpt)

	# model.load_state_dict(checkpoint['model_state_dict'])
	if torch.cuda.device_count() > 1:
		model.load_state_dict(checkpoint['model_state_dict'])
	else:
		ckpt = {}
		for key in checkpoint['model_state_dict'].keys():
			k = key.split('module.', 1)[1]
			ckpt[k] = checkpoint['model_state_dict'][key]
		model.load_state_dict(ckpt)	
	model = model.to(device)

	print("Loaded model from: ", args.model_ckpt)

	return model.eval()


# Function to obtain the predictions
def predict(args):

	# Extract the frames from the given input video
	faces = get_frames(args.input)
	total_frames = len(faces)

	if len(faces) < args.num_frames: 
		print("No of frames is less than {}!".format(args.num_frames))
		return
	print("Total no of frames = ", total_frames)

	# Obtain a window for frames
	id_windows = [range(i, i + args.num_frames) for i in range(0, total_frames, 
				args.num_frames - hp.hparams.overlap) if (i + args.num_frames <= total_frames)]
	print("ID windows: ", id_windows)

	all_images = [[faces[i] for i in window] for window in id_windows]
	print("All images: ", len(all_images))

	inp_wav = load_wav(args)
	spec_ip = get_spec(inp_wav)
	print("Noisy spec inp: ", spec_ip.shape)


	# Load the model
	model = load_model(args)


	for i, window_images in enumerate(tqdm(all_images)):

		images = get_window_images(window_images)

		if(images.shape[0] != args.num_frames):
			continue
		image_batch = np.expand_dims(images, axis=0)			#1x25x15x48x96

		# Get the corresponding input noisy melspectrograms
		idx = id_windows[i][0]
		spec_window = get_window_spec(spec_ip, idx)

		if(spec_window.shape[0] != hp.hparams.spec_step_size):
			continue
		spec_batch = np.expand_dims(np.array(spec_window), axis=0)
		
		x_mag = torch.FloatTensor(spec_batch)[..., :257].to(device)
		x_phase = torch.FloatTensor(spec_batch)[..., 257:].to(device)
		x_image = torch.FloatTensor(image_batch).to(device)

		# Predict the spectrograms for the corresponding window
		with torch.no_grad():
			pred_mag, pred_phase = model(x_mag, x_phase, x_image)


		pred_mag = pred_mag.cpu().numpy()
		pred_mag = np.squeeze(pred_mag, axis=0).T
		
		pred_phase = pred_phase.cpu().numpy()
		pred_phase = np.squeeze(pred_phase, axis=0).T
		
				
		# Concatenate the melspectrogram windows to generate the complete spectrogram	
		if i == 0:
			generated_mag = pred_mag[:, :80]
			generated_phase = pred_phase[:, :80]
		else:
			generated_mag = np.concatenate((generated_mag, pred_mag[:, :80]), axis=1)
			generated_phase = np.concatenate((generated_phase, pred_phase[:, :80]), axis=1)


	print("Output mag: ", generated_mag.shape)
	print("Output phase: ", generated_phase.shape)

	# Reconstruct the audio and generate the output video
	generate_video(generated_mag, generated_phase, args)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-i', '--input', type=str, required=True, help='Filepath of noisy video')
	parser.add_argument('-m', '--model_ckpt', type=str,  required=True, \
						help='Name of saved checkpoint to load weights from')
	parser.add_argument('-sr', '--sampling_rate', type=int, default=16000)
	parser.add_argument('-r', '--result_dir', default='results', required=False, \
						help='Name of the directory to save the results')
	parser.add_argument('-f', '--fps', type=float, default=25., required=False, \
						help='FPS of input video, ignore if image', )
	parser.add_argument('-g', '--n_gpu', default=1, type=int, help='Number of GPUs to use', )
	parser.add_argument('-mask', '--mask', default=None, help='Mask left (l) or right (r) speaker')
	

	args = parser.parse_args()
	args.img_size = 96
	args.num_frames = 25

	predict(args)