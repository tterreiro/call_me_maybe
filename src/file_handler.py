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
    returns: dict[str, str]

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
    """Result of a generated function call"""
    prompt: str
    name: str
    parameters: dict[str, Union[str, int, float, bool]]


def load_json(file_path: Path) -> list[dict[str, str]]:
    """Loads json data from a file"""
    with open(file_path, 'r', encoding="utf-8") as f:
        data: list[dict[str, str]] = json.load(f)
        return data


class FileLoader:
    """Loads input data and writes generated function call results"""
    def __init__(self) -> None:
        self.func_definitions: list[Function] = []
        self.prompts: list[Prompt] = []

    def load_func_def(self, funcs_path: Path) -> None:
        """Loads and validates function definitions form a JSON file"""
        func_lst = load_json(funcs_path)
        if not func_lst:
            raise ValidationError("Function definitions file is empty")
        for i, func in enumerate(func_lst):
            try:
                self.func_definitions.append(Function.model_validate(func))
            except ValidationError as e:
                print(f"Skipping Function #{i}: {e.errors()[0]['msg']}\n")
            none_func = Function(
                name='fn_none_found',
                description="Function selected when no option is available",
                parameters={},
                returns={"type": "none"}
            )
            self.func_definitions.append(Function.model_validate(none_func))

    def load_prompt(self, input_path: Path) -> None:
        """Loads and validates user prompts from a JSON file"""
        prompt_lst = load_json(input_path)
        if not prompt_lst:
            raise ValidationError("Function calling tests file is empty.")
        for i, prompt in enumerate(prompt_lst):
            try:
                self.prompts.append(Prompt.model_validate(prompt))
            except ValidationError as e:
                print(f"Skipping prompt #{i}: {e.errors()[0]['msg']}\n")

    def write_call_results(
        self,
        call_res: list[CallResult],
        output_path: Path
    ) -> None:
        """write generated function call results to a JSON file"""
        data = []
        for res in call_res:
            data.append(res.model_dump())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=2)
