import os
import argparse
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the project root to PYTHONPATH
from utils.utils import YamlLoader

from preprocessing.preprocessing import data_preprocessing
from training.train_module import train
from training.eval_module import evaluate

def get_argument():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', required=False, default='./data/')
    parser.add_argument('--config_path', required=False, default='configs.yaml', help="Pass the path of configs file.")

    parser.add_argument('--mode', required=False, choices=['train', 'eval', 'predict'])
    parser.add_argument('--save', required=False, action='store_true', help="save the model weights.")

    return parser.parse_args()




def main():

    args = get_argument()

    config_loader = YamlLoader(args.config_path)

    trainig_configs = config_loader.get_training_config()

    df_input = pd.read_csv(os.path.join(args.data_path, trainig_configs.data_config.input_data_name))

    df_preprocessed = data_preprocessing(df_input)
    
    # df_preprocessed.to_csv('./predictor/data/processed_data.csv')

    if args.mode == 'train':
        train(df_preprocessed, config=trainig_configs, mode=args.mode)
    elif args.mode == 'eval':
        evaluate(df_preprocessed, config=trainig_configs, mode=args.mode, save=args.save)
    
    print('end')

if __name__ == "__main__":
    main()