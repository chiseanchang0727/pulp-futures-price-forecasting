import torch
from torch import nn
from config.train_configs import TrainingConfig
from training.data_loader import DataModule
from utils.utils import set_seed
from utils.model_utils import create_mlp_model
from utils.training_utils import train_one_epoch, evalute_model

def train(df, config: TrainingConfig, mode):
    
    set_seed(config.seed)

    data_module = DataModule(df, config, mode)

    device = torch.device("cuda" if (torch.cuda.is_available() and config.accelerator == "gpu") else "cpu")

    total_train_loss = 0.0
    total_val_loss = 0.0

    for fold in range(config.n_fold):

        # train_loader = data_module.train_loader(fold)
        # valid_loader = data_module.valid_loader(fold)

        train_loader, valid_loader = data_module.get_fold_loader(fold, num_workers=config.wokers)

        input_size = data_module.train_dataset.features.shape[1]

        model, optimizer, scheular = create_mlp_model(input_size, config, device)

        criterion = nn.MSELoss()  

        best_val_loss = float('inf')
        no_improvement_epochs = 0

        print('Training start!')

        for epoch in range(config.epochs):

            avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            
            avg_val_loss = evalute_model(model, valid_loader, criterion, device)

            scheular.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs > config.early_stopping:
                print(f'Early stopping triggered at epoch {epoch+1} for fold {fold}')
                break

            if epoch % 2 == 0:
                print(f"Fold: {fold} | Epoch: {epoch+1}/{config.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        total_train_loss += avg_train_loss
        total_val_loss += avg_val_loss

        print(f"Last learning rate: {scheular.get_last_lr()}")
        print(f"Fold-{fold} training completed!")
    
    avg_train_loss = total_train_loss / config.n_fold
    avg_val_loss = total_val_loss / config.n_fold

    print(f"Average training loss: {avg_train_loss:.4f}")
    print(f"Average validation loss: {avg_val_loss:.4f}")