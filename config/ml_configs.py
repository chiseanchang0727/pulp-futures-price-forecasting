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