import os, argparse
import subprocess
import librosa
from tqdm import tqdm

def load_wav(file):

	if file.rsplit('.', 1)[1] == '.wav':
		wav = librosa.load(file, sr=16000)[0]
	else:
		wav_file  = 'tmp.wav';

		subprocess.call('ffmpeg -hide_banner -loglevel panic -threads 1 -y -i %s -async 1 -ac 1 -vn \
						-acodec pcm_s16le -ar 16000 %s' % (file, wav_file), shell=True)

		wav = librosa.load(wav_file, sr=16000)[0]

		os.remove("tmp.wav")

	return wav


def create_mixed_file(args):

	# Load the audio files
	clean_wav = load_wav(args.clean_file)
	random_wav = load_wav(args.random_file)

	# Mix the files
	min_length = min(len(clean_wav), len(random_wav))
	noisy_wav = clean_wav[:min_length] + random_wav[:min_length]
	print("Created the noisy wav: ", noisy_wav.shape)

	# Save the noisy wav
	mixed_audio_file = args.output+'.wav'
	librosa.output.write_wav(mixed_audio_file, noisy_wav, 16000)

	# Save the video output
	no_sound_video = os.path.join(args.output + '_nosouund.mp4')
	subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -c copy -an -strict -2 %s' % (args.clean_file, no_sound_video), shell=True)

	video_output_mp4 = os.path.join(args.output + '.mp4')
	subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -shortest -strict -2 -q:v 1 %s' % 
					(mixed_audio_file, no_sound_video, video_output_mp4), shell=True)

	os.remove(no_sound_video)

	print("Successfully mixed and saved the noisy video")

if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-f1', '--clean_file', type=str, required=True, help='Clean video file')
	parser.add_argument('-f2', '--random_file', type=str, required=True, help='Random audio/video file to mix with clean file')
	parser.add_argument('-o', '--output', default='output', required=False, \
						help='Name of the output file to save the mixed file')	

	args = parser.parse_args()


	create_mixed_file(args)