import os
import argparse
import pandas as pd
import torch
from utils.utils import YamlLoader

from data_modules.preprocessing import data_preprocessing
from training.train_module import train
from training.eval_module import evaluate
from training.data_loader import DataModule
from data_modules.data_for_prediction import create_data_for_prediction
from predict.predict_futures import predict


def get_argument():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', required=False, default='./data/')
    parser.add_argument('--config_path', required=False, default='configs_mlp.yaml', help="Pass the path of configs file.")

    parser.add_argument('--mode', required=False, choices=['train', 'eval', 'predict'])
    parser.add_argument('--save', required=False, action='store_true', help="save the model weights.")

    return parser.parse_args()




def main():

    args = get_argument()

    config_loader = YamlLoader(args.config_path)

    trainig_configs = config_loader.get_training_config()

    df_input = pd.read_csv(os.path.join(args.data_path, trainig_configs.data_config.input_data_name))

    df_preprocessed = data_preprocessing(df_input, config=trainig_configs)
    

    # df_preprocessed.to_csv('./data/processed_data.csv', index=False)


    if args.mode == 'train':
        train(df_preprocessed, config=trainig_configs, mode=args.mode)

    elif args.mode == 'eval':
        evaluate(df_preprocessed, config=trainig_configs, mode=args.mode, save=args.save)

    elif args.mode == 'predict':
        data_module = DataModule(df=df_preprocessed, config=trainig_configs, mode=args.mode)
        _, testing_data = data_module.custom_train_test_split(df_preprocessed, config=trainig_configs)
        data_loader = data_module.input_data_loader_for_prediction(num_workers=0)
        device = torch.device("cuda" if (torch.cuda.is_available() and trainig_configs.accelerator == "gpu") else "cpu")

        predict(
            testing_data, 
            config=trainig_configs, 
            data_loader=data_loader, 
            device=device, 
            model_path='./save_models/Pulp_Future_Price_nn_2024-12-30-17-16/model.pth'
        )
    
    print('end')

if __name__ == "__main__":
    main() 