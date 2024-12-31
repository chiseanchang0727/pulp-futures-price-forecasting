import torch
from config.train_configs import TrainingConfig
from utils.model_utils import create_mlp_model
from utils.utils import set_seed

def predict(df, config: TrainingConfig, data_loader, device, model_path):
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

    # Prepare the test dataset
    test_features = df.drop(columns=config.data_config.target).values  

    # Load the model
    input_size = test_features.shape[1]
    model, _, _ = create_mlp_model(input_size, config, device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    predictions = []


    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    return predictions
