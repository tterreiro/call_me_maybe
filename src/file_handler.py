from pydantic import BaseModel, field_validator, ValidationError
from pathlib import Path
import json
from typing import Literal, Union


class Parameter(BaseModel):
    """Schema representing the parameter type"""
    type: Literal["string", "integer", "number", "boolean"]


class Function(BaseModel):
    """Schema representing a function and its structure"""
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter

    @field_validator("name", "description")
    @classmethod
    def not_empty(cls, value: str) -> str:
        """Check if name and description are empty."""
        if not value.strip():
            raise ValueError("Function name/description cannot be empty.")
        return value

    @field_validator("parameters")
    @classmethod
    def no_param_name(
        cls,
        value: dict[str, Parameter]
            ) -> dict[str, Parameter]:
        """Check if param name is empty."""
        for name in value:
            if not name.strip():
                raise ValueError("Parameter name cannot be empty.")
        return value


class Prompt(BaseModel):
    """Shcema representing user prompt"""
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        """Check if prompt is empty."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        return prompt


class CallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Union[str, int, float, bool]]
