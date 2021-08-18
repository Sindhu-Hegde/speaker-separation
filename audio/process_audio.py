import sys
import numpy as np
import argparse
import os
from os.path import exists, isdir, basename, join, splitext
from glob import glob
import subprocess
from tqdm import tqdm

def preporocess(files, args):

	progress_bar = tqdm(range(len(files)))

	for i in progress_bar:

		# print("File: ", files[i])

		# Create the folder to save the results
		sub_folder = files[i].rsplit('/', 1)[0]
		folder = os.path.join(sub_folder, files[i].rsplit('/', 1)[1].split('.')[0])
		# print("Folder: ", folder)
		if not os.path.exists(folder):
			os.makedirs(folder)

		wav_file = os.path.join(folder, 'audio.wav') 
		# print("Wav file: ", wav_file)
		subprocess.call('ffmpeg -hide_banner -loglevel panic -threads 1 -y -i %s -async 1 -ac 1 -vn \
					-acodec pcm_s16le -ar 16000 %s' % (files[i], wav_file), shell=True)

		video_file = os.path.join(folder, 'video.mp4')
		# print("Video file: ", video_file) 
		subprocess.call('mv %s %s' % (files[i], video_file), shell=True)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-p', '--data_path', required=True, help='Path containing the audio files')		
	# parser.add_argument('-sp', '--save_path', required=True, help='Path to save audio and video files')	

	args = parser.parse_args()

	filelist = glob(os.path.join("{}/*/*/*.mp4".format(args.data_path)))
	print("No of files: ", len(filelist))

	preporocess(filelist, args)
	