from typing import Any, Dict
from pydantic import BaseModel, ValidationError

class Parameter(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Dict[str, str]

class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: Parameter
