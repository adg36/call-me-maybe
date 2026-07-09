"""Pydantic models representing the project's input and output schemas.

These models are used to validate function definitions, prompts, and
generated function calls loaded from or written to JSON files.
"""

from typing import Any, Dict, Literal
from pydantic import BaseModel


class Parameter(BaseModel):
    """Represent a function parameter definition.

    Stores the expected type of a function parameter as defined in the
    function schema.
    """
    type: Literal["number", "integer", "string"]


class FunctionSchema(BaseModel):
    """Represent the schema of a callable function.

    Stores the function name, description, parameter definitions, and
    return type information loaded from the function definition file.
    """
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Dict[str, str]


class PromptSchema(BaseModel):
    """Represent a single natural language prompt."""
    prompt: str


class FunctionCall(BaseModel):
    """Represent a generated function call.

    Stores the original prompt together with the selected function name
    and the generated parameter values.
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]
