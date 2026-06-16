from typing import Dict
from pydantic import BaseModel

class Parameter(BaseModel):
    type: str

class FunctionSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Dict[str, str]

class PromptSchema(BaseModel):
    prompt: str

class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: Parameter
