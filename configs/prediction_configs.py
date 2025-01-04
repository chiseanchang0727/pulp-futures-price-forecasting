from typing import List
from pydantic import BaseModel, Field


class PredictionConfig(BaseModel):
    timewindow: int = Field(
        default=0,
        description='Time window for prediction. Unit is day.'
    )