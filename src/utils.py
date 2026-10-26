from .file_handler import Function
import json
from typing import Union, cast


class PromptBuilder:
    """Helper class to construct system prompts guiding LLM tool choice."""
    def make_prompt(self, functions: list[Function], user_prompt: str) -> str:
        """
        Construct a formatted system prompt with
        function schemas and examples.

        Args:
            functions: List of Function schemas
                       available to include in prompt.
            user_prompt: Raw query string submitted by user.
        Returns:
            Formatted full prompt string suitable for model encoding.
        """
        prompt = [
            'You are a function calling tool.\n',
            'Produce ONLY valid JSON.',
            'Select the correct function and parameters from the list:',
        ]
        for func in functions:
            prompt.append(f"Function: {func.name}")
            prompt.append("Parameters:")
            for p_name, p_type in func.parameters.items():
                prompt.append(f"{p_name} ({p_type.type})")
        prompt.extend([
            "",
            "Example:",
            "What is the result of 4 + 8?",
            "{",
            '  "name": "fn_add_numbers",',
            '  "parameters": {',
            '    "a": 4,',
            '    "b": 8',
            "  }",
            "}",
            "",
            f"\n{user_prompt}"
        ])
        return "\n".join(prompt)


def to_type(p_type: str, value: str) -> Union[str, int, float, bool]:
    """
    Convert a raw generated string into a target primitive Python data type.

    Args:
        p_type: Desired output type
                ('string', 'number', 'integer', or 'boolean').
        value: String value to parse and convert.

    Returns:
        Converted value as primitive Python type (str, int, float, or bool).
    Raises:
        ValueError: If value cannot be parsed or cast to target p_type.
    """
    try:
        if p_type.lower() == "string":
            try:
                return cast(str, json.loads(f'"{value}"'))
            except json.JSONDecodeError:
                pass
        if p_type.lower() == "number":
            return float(value)
        if p_type.lower() == "integer":
            return int(value)
        if p_type.lower() == "boolean":
            if value.lower() == 'true':
                return True
            if value.lower() == 'false':
                return False
            raise ValueError(f"Invalid boolean: {value!r}")
        return str(value)
    except ValueError as e:
        raise ValueError(f"Invalid {p_type}: {value!r}") from e


class GenerationLogger:
    """
    Streams generated prompt text and
    token output directly to standard output.
    """
    def token(self, text: str) -> None:
        """
        Stream generated text token immediately to stdout.

        Args:
            text: Token string content to print.
        """
        print(text, end="", flush=True)

    def prompt(self, text: str) -> None:
        """
        Print the current input prompt heading to stdout.

        Args:
            text: Raw prompt string.
        """
        print(f"\nPrompt: {text}")
