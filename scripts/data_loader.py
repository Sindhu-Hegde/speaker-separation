from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import audio.audio_utils as audio
import audio.hparams as hparams
import random
import os
from glob import glob
from os.path import dirname, join, basename, isfile
import librosa
from decord import VideoReader
from decord import cpu, gpu

     
class DataGenerator(Dataset):

    def __init__(self, data_path, img_size, num_frames, sampling_rate, split):

        self.data_path = data_path
        self.img_size = img_size
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.split = split
        self.files = hparams.get_filelist(data_path, split) 
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        while(1):
            index = random.randint(0, len(self.files) - 1)
            vid_file = self.files[index]

            # Extract the frames
            fid, frames = self.extract_frames(vid_file)

            # Condition checks
            if frames is None:
                continue

            if len(frames) != self.num_frames:
                continue

            # Extract the corresponding speech spectrograms
            stft, y = self.process_audio(vid_file, fid)

            if stft is None or y is None:
                continue

            # Decompose into magnitude and phase representations and convert to Torch tensors
            x_mag = torch.FloatTensor(np.array(stft)[..., :257])
            x_phase = torch.FloatTensor(np.array(stft)[..., 257:])
            x_image = torch.FloatTensor(np.array(frames))
            y_mag = torch.FloatTensor(np.array(y)[..., :257])
            y_phase = torch.FloatTensor(np.array(y)[..., 257:])

            return x_mag, x_phase, x_image, y_mag, y_phase


    def extract_frames(self, video):

        # Read the video
        try:
            vr = VideoReader(video, ctx=cpu(0), width=self.img_size, height=self.img_size)
        except:
            return None, None

        # Obtain the frames
        try:
            fid = random.choice(range(self.num_frames//2, len(vr)-self.num_frames//2-1))
            selected_frames = range(fid-self.num_frames//2, fid+self.num_frames//2+1)
            frames = vr.get_batch(selected_frames).asnumpy()
        except:
            return None, None

        # Convert to array
        frames = np.asarray(frames) / 255. 

        return fid, frames


    def process_audio(self, vid_file, fid):

        # Load the corresponding audio file
        try:
            audio_file = vid_file.replace('.mp4', '.wav')
            gt_wav = audio.load_wav(audio_file, self.sampling_rate)
        except:
            return None, None

        # Get the random file to mix with the clean ground truth file
        random_file = random.choice(self.files)

        # Load the random audio file
        try:
            random_audio_file = random_file.replace('.mp4', '.wav')
            random_wav = audio.load_wav(random_audio_file, self.sampling_rate)
        except:
            return None, None

        # Mix the GT speech with another randomly sampled speech to obtain the input noisy speech
        try:
            idx = random.randint(0, len(random_wav) - len(gt_wav) - 1)
            noisy_wav = gt_wav + random_wav[idx:idx + len(gt_wav)]
        except:
            return None, None

        # Extract the corresponding GT and noisy audio segments
        gt_seg_wav, noisy_seg_wav = self.crop_audio_window(gt_wav, noisy_wav, fid)
        
        if gt_seg_wav is None or noisy_seg_wav is None:
            return None, None
        

        # -----------------------------------STFTs--------------------------------------------- #
        # Get the STFT, normalize and concatenate the mag and phase of GT and noisy wavs
        noisy_spec = self.get_spec(noisy_seg_wav) 
        # print("Noisy spec: ", noisy_spec.shape)                                     # 100x514

        gt_spec = self.get_spec(gt_seg_wav)
        # print("GT spec: ", gt_spec.shape)                                           # 100x514                             

        # ------------------------------------------------------------------------------------- #
        
        # Input to the model: Noisy spectrogram array
        inp_stft = np.array(noisy_spec)
        # print("Input STFT: ", inp_stft.shape)                                       # 100x514

        # GT to train the model
        gt_stft = np.array(gt_spec)
        # print("GT stft: ", gt_stft.shape)                                           # 100x514

        return inp_stft, gt_stft


    def crop_audio_window(self, gt_wav, noisy_wav, center_frame):

        if gt_wav.shape[0] - hparams.hparams.wav_step_size < 16000: 
            return None, None

        start_frame_id = center_frame - self.num_frames//2
        if start_frame_id < 0:
            return None, None

        start_idx = int(start_frame_id * (self.sampling_rate/hparams.hparams.fps))
        end_idx = start_idx + hparams.hparams.wav_step_size

        gt_seg_wav = gt_wav[start_idx : end_idx]
        if len(gt_seg_wav) != hparams.hparams.wav_step_size: 
            return None, None

        noisy_seg_wav = noisy_wav[start_idx : end_idx]
        if len(noisy_seg_wav) != hparams.hparams.wav_step_size: 
            return None, None

        return gt_seg_wav, noisy_seg_wav

    

    def get_spec(self, wav):

        stft = librosa.stft(y=wav, n_fft=hparams.hparams.n_fft, hop_length=hparams.hparams.hop_size, win_length=hparams.hparams.win_size).T
        stft = stft[:-1]
        # print("STFT: ", stft.shape)                                       # 100x257

        mag = np.abs(stft)
        mag = audio.db_from_amp(mag)
        phase = audio.angle(stft)

        norm_mag = audio.normalize_mag(mag)
        norm_phase = audio.normalize_phase(phase)
            
        spec = np.concatenate((norm_mag, norm_phase), axis=1)               # 100x514
        
        return spec


def load_data(data_path, img_size, num_frames, num_workers, batch_size=4, split='train', sampling_rate=16000, shuffle=True):
    
    train_data = DataGenerator(data_path, img_size, num_frames, sampling_rate, split)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return train_loader