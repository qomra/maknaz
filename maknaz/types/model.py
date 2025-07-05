from typing import Optional
from pydantic import BaseModel

class BaseACModel(BaseModel):
    name: str
    dataset: Optional[str]

class ACFinetunedModel(BaseACModel):
    base_model: str
    local_path:  str