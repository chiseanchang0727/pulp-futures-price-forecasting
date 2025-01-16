import pandas as pd
import torch
from configs.train_configs import TrainingConfig
from models.models import models
from utils.utils import set_seed

def predict(config: TrainingConfig, data_loader, device, model_path):
    """
    Function to predict results using the saved model.

    Args:
        df: DataFrame containing the test data.
        config: TrainingConfig object with configuration details.
        device: torch.device specifying the computation device.
        model_path: Path to the saved model.
    
    Returns:
        predictions: The predicted values as a NumPy array.
    """

    set_seed(config.seed)

    for X, y_true in data_loader:

        # Load the model
        input_size = X.shape[1]
        model, _, _ = models(input_size, config, device)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        predictions = []

        with torch.no_grad():

            inputs = X.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    df_y_true_predictions =  pd.DataFrame({
        "y_true": y_true,
        "preidctions": predictions
    })[:7] # only get the first 7 days

    return df_y_true_predictions
