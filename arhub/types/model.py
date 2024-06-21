from typing import List,Optional,Union
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator


class BaseACModel(BaseModel):
    name: str
    dataset: Optional[str]

class ACFinetunedModel(BaseACModel):
    base_model: str
    local_path:  str