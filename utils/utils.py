import os
import torch
from datetime import datetime
import random
import numpy as np
import yaml
import pandas as pd
import math
from utils.enums import ConfigType
from config.train_configs import TrainingConfig



class YamlLoader:
    def __init__(self, file_path):
        self.config_data = self.load_config_from_file(file_path)

    def load_config_from_file(self, file_path: str):
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        return config_data # normalize_keys(config_data)
    
    def get_training_config(self):
        return TrainingConfig(**self.config_data[ConfigType.Training.value])
    


def set_seed(seed: int):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch CUDA
    torch.cuda.manual_seed_all(seed)  # PyTorch Multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable optimizations for reproducibility

def save_model(model, root_path):
    
    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    folder_name = f"Pulp_Future_Price_nn_{time_stamp}"
    
    save_path = os.path.join(root_path, folder_name)
    
    os.makedirs(save_path, exist_ok=True)
    
    model_file = os.path.join(save_path, "model.pth")
    
    torch.save(model.state_dict(),model_file)
    
    print(f"Model saved at: {model_file}")
    
    
def load_model(model, path):
    
    model.load_state_dict(torch.load(path))
    
    print(f"Model loaded from: {path}")
    
    return model
