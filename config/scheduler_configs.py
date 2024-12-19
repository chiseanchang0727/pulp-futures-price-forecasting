from pydantic import BaseModel, Field


class SchedulerConfig(BaseModel):

    type: str = Field(
        default=None,
        description="Scheduler type."
    )

    mode: str = Field(
        default='min'
    )

    factor: float = Field(
        default=None,
        description="Learning reate * factor"
    )

    patience: int = Field(
        default=None,
        description="Epochs for tolerating no improvement."
    )

    min_lr: float = Field(
        default=None,
        description="Minimal learning rate."
    )