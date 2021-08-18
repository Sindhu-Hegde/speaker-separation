import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../audio/")
import hparams as hp
import librosa
import librosa.display

def plot(file, fname='spec'):

	wav = librosa.load(file, sr=16000)[0]
	stft = librosa.stft(y=wav, n_fft=hp.hparams.n_fft, hop_length=hp.hparams.hop_size, win_length=hp.hparams.win_size)
	print("STFT: ", stft.shape)

	D = np.abs(stft)
	librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),y_axis='log', x_axis='time')
	plt.title('Power spectrogram')
	plt.colorbar(format='%+2.0f dB')
	plt.tight_layout()
	plt.show()
	plt.savefig(fname+".jpg")
	plt.clf()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--file', type=str, required=True, help='Wav file input')
	args = parser.parse_args()

	plot(args.file)

