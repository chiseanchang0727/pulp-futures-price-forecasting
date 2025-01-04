from typing import List
from pydantic import BaseModel, Field


class NNHyperparameters(BaseModel):
    """Hyperparameters for NN training."""
      
    lr: float = Field(
        default=1e-3, 
        description="Learning rate for the optimizer."
    )
    
    n_hidden: List[int] = Field(
        default=None, 
        description="Configuration for hidden layer sizes (e.g., [128, 64])."
    )
    
    weight_decay: float = Field(
        default=1e-4, 
        description="Weight decay (L2 regularization) for better convergence."
    )
    dropout: List[float] = Field(
        default=None,
        description="Configs for dropout layers."
    )


class LSTMHyperparameters(BaseModel):
    """Hyperparameters for an LSTM-based model."""

    lr: float = Field(
        default=1e-3,
        description="Learning rate for the optimizer."
    )

    hidden_size: int = Field(
        default=32,
        description="Dimension of the LSTM hidden state."
    )

    num_layers: int = Field(
        default=1,
        description="Number of stacked LSTM layers."
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout rate for LSTM (applied between stacked layers)."
    )

    weight_decay: float = Field(
        default=1e-4,
        description="Weight decay (L2 regularization) for better convergence."
    )
