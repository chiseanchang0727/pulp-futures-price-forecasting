from torch import optim
from configs.train_configs import TrainingConfig

def initialize_mlp_components(input_size, config: TrainingConfig, device):
    from models.mlp import MLP
    
    model = MLP(
            input_size=input_size,
            hidden_dims=config.model_nn.n_hidden,
            dropouts=config.model_nn.dropout
        ).to(device)

   
    optimizer = optim.Adam(model.parameters(), lr=config.model_nn.lr, weight_decay=config.model_nn.weight_decay)

    scheular = None
    if config.scheduler.type == 'ReduceLROnPlateau':
        scheular = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=config.scheduler.mode, 
            factor=config.scheduler.factor, 
            patience=config.scheduler.patience, 
            min_lr=config.scheduler.min_lr
        )

    return model, optimizer, scheular



def initialize_lstm_components(input_size, config: TrainingConfig, device):
    from models.lstm import LSTMModel

    model = LSTMModel(

    )



def models(input_size, config: TrainingConfig, device):

    model, optimizer, scheular = initialize_mlp_components(input_size, config , device)


    return model, optimizer, scheular