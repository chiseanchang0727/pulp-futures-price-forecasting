from enum import Enum


class ConfigType(Enum):

    LLM = 'llm'
    Training = 'Training'
    DataUpload = 'DataUpload'

    # def __repr__(self) -> str:
    #     return f'"{self.value}"'