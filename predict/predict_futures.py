import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.model_utils import create_mlp_model
from utils.utils import set_seed

def predict(df, config, device, model_path="/save_models/model.pth"):
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
    # Set the seed for reproducibility
    set_seed(config.seed)

    # Prepare the test dataset
    test_features = df.drop(columns=config.target_column).values  # Assuming `target_column` exists in config
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Load the model
    input_size = test_features.shape[1]
    model, _, _ = create_mlp_model(input_size, config, device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    predictions = []

    # Make predictions
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    return predictions
