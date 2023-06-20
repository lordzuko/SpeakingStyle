import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, bc_2013
from dask.distributed import Client

def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    if "BC2013" in config["dataset"]:
        bc_2013.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if "BC2013" in config["dataset"]:
        client = Client()
    main(config)
