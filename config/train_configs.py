from pydantic import BaseModel, Field
from config.data_configs import DataConfig
from config.ml_configs import NNHyperparameters
from config.scheduler_configs import SchedulerConfig
from typing import List, Optional

class TrainingConfig(BaseModel):
    data_config: DataConfig = Field(
        default_factory=DataConfig, 
        description="Data configuration."
    )
    model_nn: NNHyperparameters = Field(
        default_factory=NNHyperparameters,
        description="Neural network hyperparameter configuration."
    )

    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig
    )
    
    train_test_split_size: float = Field(
        default=0.8,
        description="Train test ratio."
    )
    
    accelerator: str = Field(
        default='cpu',
        description="Config for using GPU or CPU."
    )
    seed: int = Field(
        default=42,
        description="Random seed for consistent output."
    )
    epochs: int = Field(
        default=None,
        description="Number of epochs to train the model."
    )
    n_fold: int = Field(
        default=None,
        description="Number of folds for cross-validation."
    )
    batch_size: int = Field(
        default=None, 
        description="Size of the mini-batch for training."
    )

    early_stopping: int = Field(
        default=5,
        description="Early stop the training if no improvement."
    )

    model_save_path: str = Field(
        default=None,
        description='Directory for saving the model weights.'
    )

    wokers: int = Field(
        default=0
    )
