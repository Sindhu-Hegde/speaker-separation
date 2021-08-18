


# Audio-Visual Speaker Separation

This code is for separating two speakers talking simultaneously in a cocktail-party like environment. The enhanced speech for the desired target speaker is isolated and generated as output.  

------
**Features**
--------
- Separate the target speaker's speech by utilizing the lip movements along with the mixed speech.
- Works for any speaker in any language.
- Complete training code and inference codes available. 

----
Prerequisites
---
- `Python 3.7.4` (Code has been tested with this version)
- ffmpeg: `sudo apt-get install ffmpeg`
- Video loading and reading is done using [Decord](https://github.com/dmlc/decord): `pip install decord`
- Install necessary packages using `pip install -r requirements.txt`

-----
Pre-trained checkpoint
-----
The pre-trained model trained on VoxCeleb2 dataset can be downloaded from the following link: [Link](https://drive.google.com/file/d/1B1HVaWZS8OhbZoTY8XPWS8dizONj2aGm/view?usp=sharing)

---
Separating the target speaker's speech using the pre-trained model (Inference)
----
To isolate the speech of the target speaker from a mixed (two speaker) video, download the pre-trained checkpoint and run the following command:

    python test.py --input=<mixed-video-file> --checkpoint=<trained-model-ckpt-path> 

The result is saved (by default) in `results/pred_<input_file-name>.mp4`. The result directory can be specified in arguments, similar to several other available options. The input file needs to be a **video file**: `*.mp4`, `*.avi`, containing two speakers talking. If multiple speakers are present in the input video, then the target speaker needs to be specified by masking the other regions of the face (argument: `--mask=r` indicating to mask the right half of the video and consider the left speaker as the desired target speaker). Note that the input video must contain only two speakers talking, as this project only works for separating two speakers.

# Training

We illustrate the training process using the [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset. Adapting for other datasets would involve small modifications to the code.

##### VoxCeleb2 train dataset folder structure

```
data_root 
├── mp4
│   ├── id
│	│	├── subfolders
│	│	│		├── five-digit numbered video files (.mp4)
│	│	│		├── corresponding five-digit numbered audio files (.wav)
```

## Train!
 
To train the speaker separation model, run the following command: 

    python train.py --data_root_vox2_train=<path-of-VoxCeleb2-train-set> --checkpoint_dir=<path-to-save-the-trained-model> 
    
The model can be resumed for training as well. Look at `python train.py --help` for more details. Also, additional less commonly-used hyper-parameters can be set using the arguments.

Note that the mixed speech input during the training is obtained on-the-fly by mixing the clean speech signal with the random speech signal from the dataset itself.

---
Acknowledgements
---
Parts of this code has been modified using our [Denoising repository](https://github.com/Sindhu-Hegde/pseudo-visual-speech-denoising). The audio functions and parameters are taken from this [TTS repository](https://github.com/r9y9/deepvoice3_pytorch). We thank the authors for their wonderful code. 
