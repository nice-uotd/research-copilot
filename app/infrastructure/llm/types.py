from enum import Enum

class ModelProvider(str, Enum):

    OPENAI = "openai"
    AZURE = "azure"
    CUSTOM = "custom"
