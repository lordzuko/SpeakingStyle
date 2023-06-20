# Aligning latent space of speaking style with human perception using a re-embedding strategy

To generate appropriate speech in context, we must not only consider what is said but also how it is said. Prosody is used to influence a particular meaning, to indicate attitude and emotion, and more. Modelling prosody in Text-To-Speech (TTS) is a highly complex task, as prosody manifests itself in multiple hierarchies in speech, from phone-level acoustic features to suprasegmental effects spanning multiple sentences. As perception of certain prosodic features is highly subjective and prosodically labelled speech corpora are scarce, researchers have looked to unsupervised methods of modelling prosody in TTS.

Many propose to jointly train a reference encoder to model a latent representation of prosody or speaking style [e.g., 1, 2, 3]. The reference encoder is conditioned on the target speech during training and learns to model features that are perceptually important for the acoustic model in reconstructing the target speech [1, 2]. Many recent unsupervised style-modelling methods have focused on disentangling non-prosodic features from the latent prosodic representations [e.g., 3, 4, 8], but there is little known about how well this latent space aligns with human perception of style. Understanding this space could help solve the issue of feature-entanglement without any modifications to the model.

Here, we propose to manipulate the latent prosody space using light-supervision from human annotators. Such strategies have already been proposed for speaker-modelling [5, 6] and speaker-adaptation [9] but not for speaking-style modelling. To achieve this we suggest to re-embed synthesised utterances that human annotators have tuned acoustically to match a target speaking style. After re-embedding the annotated utterances we aim to tune a fixed bank of style tokens [2]. This could not only further understanding the latent prosody space but also align it with human perception improving the quality of style generation.

* [1] [Towards end-to-end prosody transfer for expressive speech synthesis with tacotron](http://proceedings.mlr.press/v80/skerry-ryan18a.html)
* [2] [Style tokens: Unsupervised style modeling, control and transfer in end-to-end speech synthesis](http://proceedings.mlr.press/v80/wang18h.html?ref=https://githubhelp.com)
* [3] [Daft-exprt: Robust prosody transfer across speakers for expressive speech synthesis](https://arxiv.org/abs/2108.02271)
* [4] [Copycat: Many-to-many fine-grained prosody transfer for neural text-to-speech](https://arxiv.org/abs/2004.14617)
* [5] [Perceptual-similarity-aware deep speaker representation learning for multi-speaker generative modeling](https://ieeexplore.ieee.org/iel7/6570655/9289074/09354556.pdf)
* [6] [DNN-based Speaker Embedding Using Subjective Inter-speaker Similarity for Multi-speaker Modeling in Speech Synthesis](https://arxiv.org/pdf/1907.08294)
* [7] [Fastspeech: Fast, robust and controllable text to speech](https://proceedings.neurips.cc/paper/2019/hash/f63f65b503e22cb970527f23c9ad7db1-Abstract.html)
* [8] [Effective use of variational embedding capacity in expressive end-to-end speech synthesis](https://arxiv.org/abs/1906.03402)
* [9] [Human-in-the-loop Speaker Adaptation for DNN-based Multi-speaker TTS](https://arxiv.org/abs/2206.10256)

# FastSpeech 2 - PyTorch Implementation

The base code for the project is clone from: https://github.com/ming024/FastSpeech2

This is a PyTorch implementation of Microsoft's text-to-speech system [**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**](https://arxiv.org/abs/2006.04558v1). 
This project is based on [xcmyz's implementation](https://github.com/xcmyz/FastSpeech) of FastSpeech. Feel free to use/modify the code.

There are several versions of FastSpeech 2.
This implementation is more similar to [version 1](https://arxiv.org/abs/2006.04558v1), which uses F0 values as the pitch features.
On the other hand, pitch spectrograms extracted by continuous wavelet transform are used as the pitch features in the [later versions](https://arxiv.org/abs/2006.04558).

![](./img/model.png)

# Updates
- 2021/7/8: Release the checkpoint and audio samples of a multi-speaker English TTS model trained on LibriTTS
- 2021/2/26: Support English and Mandarin TTS
- 2021/2/26: Support multi-speaker TTS (AISHELL-3 and LibriTTS)
- 2021/2/26: Support MelGAN and HiFi-GAN vocoder

# Audio Samples
Audio samples generated by this implementation can be found [here](https://ming024.github.io/FastSpeech2/). 

# Quickstart

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

## Inference

You have to download the [pretrained models](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing) and put them in ``output/ckpt/LJSpeech/``,  ``output/ckpt/AISHELL3``, or ``output/ckpt/LibriTTS/``.

For English single-speaker TTS, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

For Mandarin multi-speaker TTS, try
```
python3 synthesize.py --text "大家好" --speaker_id SPEAKER_ID --restore_step 600000 --mode single -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

For English multi-speaker TTS, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT"  --speaker_id SPEAKER_ID --restore_step 800000 --mode single -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
```

The generated utterances will be put in ``output/result/``.

Here is an example of synthesized mel-spectrogram of the sentence "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition", with the English single-speaker TTS model.  
![](./img/synthesized_melspectrogram.png)

## Batch Inference
Batch inference is also supported, try

```
python3 synthesize.py --source preprocessed_data/LJSpeech/val.txt --restore_step 900000 --mode batch -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```
to synthesize all utterances in ``preprocessed_data/LJSpeech/val.txt``

## Controllability
The pitch/volume/speaking rate of the synthesized utterances can be controlled by specifying the desired pitch/energy/duration ratios.
For example, one can increase the speaking rate by 20 % and decrease the volume by 20 % by

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --duration_control 0.8 --energy_control 0.8
```

# Training

## Datasets

The supported datasets are

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- [AISHELL-3](http://www.aishelltech.com/aishell_3): a Mandarin TTS dataset with 218 male and female speakers, roughly 85 hours in total.
- [LibriTTS](https://research.google/tools/datasets/libri-tts/): a multi-speaker English dataset containing 585 hours of speech by 2456 speakers.

We take LJSpeech as an example hereafter.

## Preprocessing
 
First, run 
```
python3 prepare_align.py config/LJSpeech/preprocess.yaml
```
for some preparations.

As described in the paper, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
Alignments of the supported datasets are provided [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing).
You have to unzip the files in ``preprocessed_data/LJSpeech/TextGrid/``.

After that, run the preprocessing script by
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

Alternately, you can align the corpus by yourself. 
Download the official MFA package and run
```
./montreal-forced-aligner/bin/mfa_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/LJSpeech
```
or
```
./montreal-forced-aligner/bin/mfa_train_and_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt preprocessed_data/LJSpeech
```
or 
With multiprocessing for single speaker data
```
mfa train /scratch/space1/tc046/lordzuko/work/data/raw_data/BC2013/ lexicon/librispeech-lexicon.txt /scratch/space1/tc046/lordzuko/work/data/preprocessed_data/BC2013/bc_align.model --output_directory /scratch/space1/tc046/lordzuko/work/data/preprocessed_data/BC2013/TextGrid/ --single_speaker --num_jobs 40  --use_mp --temporary_directory /scratch/space1/tc046/lordzuko/work/data/tmp/
```

to align the corpus and then run the preprocessing script.
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

## Training

Train your model with
```
python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

The model takes less than 10k steps (less than 1 hour on my GTX1080Ti GPU) of training to generate audio samples with acceptable quality, which is much more efficient than the autoregressive models such as Tacotron2.

# TensorBoard

Use
```
tensorboard --logdir output/log/LJSpeech
```

to serve TensorBoard on your localhost.
The loss curves, synthesized mel-spectrograms, and audios are shown.

![](./img/tensorboard_loss.png)
![](./img/tensorboard_spec.png)
![](./img/tensorboard_audio.png)

# Implementation Issues

- Following [xcmyz's implementation](https://github.com/xcmyz/FastSpeech), I use an additional Tacotron-2-styled Post-Net after the decoder, which is not used in the original FastSpeech 2.
- Gradient clipping is used in the training.
- In my experience, using phoneme-level pitch and energy prediction instead of frame-level prediction results in much better prosody, and normalizing the pitch and energy features also helps. Please refer to ``config/README.md`` for more details.

Please inform me if you find any mistakes in this repo, or any useful tips to train the FastSpeech 2 model.

# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [TensorSpeech's FastSpeech 2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [rishikksh20's FastSpeech 2 implementation](https://github.com/rishikksh20/FastSpeech2)

# Citation
```
@INPROCEEDINGS{chien2021investigating,
  author={Chien, Chung-Ming and Lin, Jheng-Hao and Huang, Chien-yu and Hsu, Po-chun and Lee, Hung-yi},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Investigating on Incorporating Pretrained and Learnable Speaker Representations for Multi-Speaker Multi-Style Text-to-Speech}, 
  year={2021},
  volume={},
  number={},
  pages={8588-8592},
  doi={10.1109/ICASSP39728.2021.9413880}}
```
