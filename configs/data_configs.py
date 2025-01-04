from pydantic import BaseModel, Field
from typing import Optional, List


class DataConfig(BaseModel):
    """Configs for training data"""

    input_data_name: str = Field(
        default=None,
        description="Input data file name."
    )

    target: str = Field(
        default=None,
        description="The target of the data."
    )

    drop_cols: Optional[List[str]] = Field(
        default=None,
        description="the cols for dropping before training."
    )

    features: Optional[List[str]] = Field(
        default=None,
        description="The cols for training."
    )

    use_standardization: bool = Field(
        default=True,
        description="Use standardization if True, normalizationif False."
    )