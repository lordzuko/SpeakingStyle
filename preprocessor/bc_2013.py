import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
import joblib
from joblib import Parallel, delayed

def process_wav(**kwargs):
    out_dir = kwargs["out_dir"]
    speaker = kwargs["speaker"]
    wav_path = kwargs["wav_path"]
    base_name = wav_path["base_name"]
    in_wav_path = wav_path["wav_path"]
    sampling_rate=kwargs["sampling_rate"]
    max_wav_value=kwargs["max_wav_value"]
    out_wav_path = os.path.join(out_dir, speaker, "{}.wav".format(base_name))
    if os.path.exists(in_wav_path):
        wav, _ = librosa.load(in_wav_path, sampling_rate)
        wav = wav / max(abs(wav)) * max_wav_value
        wavfile.write(out_wav_path, sampling_rate, wav.astype(np.int16))

def process_text(**kwargs):
    out_dir = kwargs["out_dir"]
    speaker = kwargs["speaker"]
    label = kwargs["label"]
    base_name = label["base_name"]
    text = label["text"]
    cleaners = kwargs["cleaners"]
    label_path = os.path.join(out_dir, speaker, "{}.lab".format(base_name))
    text = _clean_text(text, cleaners)
    with open(label_path, "w") as f1:
        f1.write(text)
    
def prepare_align(config):
        in_dir = config["path"]["corpus_path"]
        out_dir = config["path"]["raw_path"]
        sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
        cleaners = config["preprocessing"]["text"]["text_cleaners"]
        speaker = "CB"

        
        labels = []
        wav_paths = []
        
        with open(os.path.join(in_dir, "trainset-transcript.csv"), encoding="utf-8") as f:
            for line in tqdm(f):
                parts = line.strip().split("||")
                base_name = parts[0]
                text = parts[1].split("|")[0]
                in_wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
                labels.append({"base_name": base_name, "text": text})
            
                wav_paths.append({"base_name": base_name, "wav_path": in_wav_path})
            
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            
            with joblib.parallel_backend(backend="dask"):
                parallel = Parallel(verbose=100)
                print(parallel([delayed(process_text)(out_dir=out_dir,
                                                      speaker=speaker, 
                                                      label=label, 
                                                      cleaners=cleaners) for label in labels]))
                
                print(parallel([delayed(process_wav)(out_dir=out_dir,
                                                     speaker=speaker, 
                                                     wav_path=wav_path,
                                                     sampling_rate=sampling_rate,
                                                     max_wav_value=max_wav_value) for wav_path in wav_paths]))