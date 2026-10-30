from pydantic import BaseModel, field_validator, ValidationError
from pathlib import Path
import json
from typing import Literal, Union


class EmptyDataError(Exception):
    """
    Exception raised when an input file contains no data or empty content.
    """
    pass


class Parameter(BaseModel):
    """
    Schema representing a parameter type specification for a tool/function.

    Attributes:
        type: The data type expected for the parameter (string, integer,
            number, or boolean).
    """
    type: Literal["string", "integer", "number", "boolean"]


class Function(BaseModel):
    """
    Schema representing a callable function and its structural specification.

    Attributes:
        name: Name of the function.
        description: Text description explaining what the function does.
        parameters: Mapping of parameter names to their Parameter schema.
        returns: Dictionary specifying return type information.
    """
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: dict[str, str]

    @field_validator("name", "description")
    @classmethod
    def not_empty(cls, value: str) -> str:
        """Validate that string fields are not empty or whitespace-only.

        Args:
            value: The string value to validate.
        Returns:
            The validated non-empty string.
        Raises:
            ValueError: If the string is empty or contains only whitespace.
        """
        if not value.strip():
            raise ValueError("Function name/description cannot be empty.")
        return value

    @field_validator("parameters")
    @classmethod
    def no_param_name(
        cls,
        value: dict[str, Parameter]
            ) -> dict[str, Parameter]:
        """Validate that parameter names are non-empty strings.

        Args:
            value: Mapping of parameter names to parameter schemas.
        Returns:
            The validated parameter mapping.
        Raises:
            ValueError: If any parameter name is empty or whitespace-only.
        """
        for name in value:
            if not name.strip():
                raise ValueError("Parameter name cannot be empty.")
        return value


class Prompt(BaseModel):
    """Schema representing a user input prompt.

    Attributes:
        prompt: Raw user query or instruction text.
    """
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        """Validate that the prompt text is non-empty.

        Args:
            prompt: User prompt string.
        Returns:
            The validated prompt string.
        Raises:
            ValueError: If the prompt string is empty or whitespace-only.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        return prompt


class CallResult(BaseModel):
    """
    Schema representing the formatted outcome of a generated function call.

    Attributes:
        prompt: Original user prompt that triggered the call.
        name: Name of the invoked function.
        parameters: Map of parameter names to their values.
    """
    prompt: str
    name: str
    parameters: dict[str, Union[str, int, float, bool]]


def load_json(file_path: Path) -> list[dict[str, str]]:
    """Load and parse JSON content from a file path.

    Args:
        file_path: Path to the JSON target file.
    Returns:
        List of dictionaries parsed from the JSON file.
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        json.JSONDecodeError: If the file content is invalid JSON.
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        data: list[dict[str, str]] = json.load(f)
        return data


class FileLoader:
    """
    Manages input data loading, validation
    and writing function call outputs.

    Attributes:
        func_definitions: List of successfully loaded function definitions.
        prompts: List of successfully loaded user prompts.
    """
    def __init__(self) -> None:
        """
        Initialize empty containers for function definitions and prompts.
        """
        self.func_definitions: list[Function] = []
        self.prompts: list[Prompt] = []

    def load_func_def(self, funcs_path: Path) -> None:
        """Load and validate function specifications from a JSON file.

        Args:
            funcs_path: Path to the JSON file containing function metadata.
        Raises:
            EmptyDataError: If the JSON file contains no function records.
        """
        func_lst = load_json(funcs_path)
        if not func_lst:
            raise EmptyDataError("Function definitions file is empty")
        for i, func in enumerate(func_lst):
            try:
                self.func_definitions.append(Function.model_validate(func))
            except ValidationError as e:
                print(f"Skipping Function #{i}: {e.errors()[0]['msg']}\n")
        none_func = Function(
            name='fn_none_found',
            description="Function selected when no option is available",
            parameters={},
            returns={
                "type": "none"
            }
            )
        self.func_definitions.append(Function.model_validate(none_func))

    def load_prompt(self, input_path: Path) -> None:
        """Load and validate user input prompts from a JSON file.

        Args:
            input_path: Path to the JSON file containing prompt strings.
        Raises:
            EmptyDataError: If the JSON file contains no prompt records.
        """
        prompt_lst = load_json(input_path)
        if not prompt_lst:
            raise EmptyDataError("Function calling tests file is empty.")
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
        """
        Serialize generated function call results to a JSON destination file.

        Args:
            call_res: List of CallResult model objects to write.
            output_path: Path where output JSON file should be saved.
        """
        data = []
        for res in call_res:
            data.append(res.model_dump())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=2)
