import argparse

import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-preprocess_config", type=str, help="path to preprocess.yaml")
    parser.add_argument("-model_config", type=str, help="path to model.yaml")
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(preprocess_config, model_config)
    preprocessor.build_from_path()
