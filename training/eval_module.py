import pandas as pd
import torch
from torch import nn
from configs.train_configs import TrainingConfig
from training.data_loader import DataModule
from utils.utils import set_seed, save_model
from models.models import models
from utils.training_utils import train_one_epoch, evaluate_model

def evaluate(df, config: TrainingConfig, mode, save=False):
    set_seed(config.seed)
    
    # full train
    data_module = DataModule(df, config, mode)
    device = torch.device("cuda" if (torch.cuda.is_available() and config.accelerator == "gpu") else "cpu")

    full_train_data_loader, test_loader = data_module.get_train_test_data_loader(num_workers=config.wokers)

    # test_index = data_module.df_test.index
    
    input_size = data_module.full_train_dataset.features.shape[1]
    
    model, optimizer, scheular = models(input_size, config, device)
        
    criterion = nn.L1Loss()
    
    print('Full-training start!')
    
    best_test_loss = float('inf')
    no_improvement_epochs = 0
    
    for epoch in range(config.epochs):
        avg_train_loss = train_one_epoch(model, full_train_data_loader, optimizer, criterion, device)
            
        avg_test_loss = evaluate_model(model, test_loader, criterion, device)
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        
        scheular.step(avg_test_loss)
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            
        if no_improvement_epochs > config.early_stopping:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    if save:
        save_model(model, root_path=config.model_save_path)
