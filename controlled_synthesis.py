import os
import re
from tqdm import tqdm
import argparse
from string import punctuation
import json
from scipy.io import wavfile
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from .utils.model import get_model, get_vocoder, vocoder_infer
from .utils.tools import to_device, expand
from .dataset import TextDataset
from .text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(text, lexicon, g2p, preprocess_config):
    """
    Preprocess the given text. Applies G2P
    We are tracking the word to phone using a mapping of
    word-idx : the idx tell the number of phones corresponding to word,
                when a contiguous sum is maintained, we can get word to phone mapping
                [sent] -> [[phone seq]]
                [synthesis is cool] -> array([[131, 109, 119, 134,  73, 131,  73, 131, 108, 146, 116, 141, 117]])
                words - ['synthesis', 'is', 'cool']
                idx - [8, 2, 3]
                "synthesis" - 131, 109, 119, 134,  73, 131,  73, 131
                "is" - 108, 146
                "cool" - 116, 141, 117
    """
    # sil_phones = ["sil", "sp", "spn"]
    avoid = [" ", "" , "."]
    text = text.rstrip(punctuation)
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    idx = []
    for w in words:
        len_before = len(phones)
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            # phones += list(filter(lambda p: p != " ", g2p(w)))
            phones += list(filter(lambda p: p not in avoid, g2p(w)))
        if w not in avoid:
            c_new_phones = len(phones) - len_before
            idx.append(c_new_phones)
            
    phones = "{" + "}{".join(phones) + "}"
    phones = phones.replace("}{", " ")
    words = [w for w in words if w not in avoid]
    # print("Raw Text Sequence: {}".format(text))
    # print("Phoneme Sequence: {}".format(phones))
    # print("Words: {}".format(words))
    # print("Idx: {}".format(idx))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence), words, idx

def setup_stats(preprocess_config):
    """
    Get stats to limit range of f0, duration, energy during prediction
    """
    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

def synth_sample(predictions, vocoder, model_config, preprocess_config):
    """
    Convert predicted mel spectrogram to wav.
    Does not save to file. To be used with downstream application without
    incurring unnecessary io.
    """
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = predictions[5][0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][0, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
    else:
        pitch = predictions[2][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = predictions[3][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = predictions[3][0, :mel_len].detach().cpu().numpy()

    if vocoder is not None:
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_prediction = None

    return wav_prediction

def synthesize(model, configs, vocoder, batchs, control_values):
    """
    Synthesize a speech sample given preprocessed text
    batchs: object with preprocessed text
    """
    preprocess_config, model_config, _ = configs
    pitch_control, energy_control, duration_control = control_values
    print("Controlled-synthesis-duration:", duration_control)
    batch = to_device(batchs[0], device)
    with torch.no_grad():
        # Forward
        output = model(
            *(batch[2:]),
            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control
        )

        wav_pred = synth_sample(output, vocoder, model_config, preprocess_config)
        # sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        # wavfile.write(os.path.join("./", "{}.wav".format(basename)), sampling_rate, wav_pred)
    
    return output, wav_pred

def preprocess_single(text, lexicon, g2p, args, preprocess_config, fine_control={}):
    """
    Prepare data for synthesis.
    fine_control: when this is not None, the synthesised speech will be controlled.
    """
    ids = raw_texts = [text]
    speakers = np.array([args.speaker_id])
    # if preprocess_config["preprocessing"]["text"]["language"] == "en":
    out = preprocess_english(text,lexicon, g2p, preprocess_config)
    texts, words, idxs = np.array([out[0]]), out[1], out[2]

    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    if fine_control:
        control_values = fine_control["p"], fine_control["e"], fine_control["d"]
    else:
        control_values = args.pitch_control, args.energy_control, args.duration_control
    return control_values, batchs

def synthesize_single(text, model, configs, vocoder, args, fine_control={}):
    g2p = G2p()
    preprocess_config, _, _ = configs
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    control_values, batchs = preprocess_single(text, lexicon, g2p, args, preprocess_config, fine_control)
    return synthesize(model, configs, vocoder, batchs, control_values)



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    print("Loading Model...")
    model = get_model(args, configs, device, train=False)
    print("Model Loaded")
    # Load vocoder
    print("Loading Vocoder...")
    vocoder = get_vocoder(model_config, device)
    print("Vocoder Loaded")
    
    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control        
    print("Synthesizing ...")
    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
