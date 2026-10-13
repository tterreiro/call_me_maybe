from .file_handler import Function
import json
from typing import Union, cast


class PromptBuilder:
    """Makes the prompt to constrain the function call generation"""
    def make_prompt(self, functions: list[Function], user_prompt: str) -> str:
        """Makes a prompt containing available functuins and user request"""
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
    """Converts a string to the requested parameter type"""
    try:
        if p_type.lower == "string":
            try:
                return cast(str, json.loads(f'"{value}"'))
            except json.JSONDecodeError:
                pass
        if p_type.lower == "number":
            return float(value)
        if p_type.lower == "integer":
            return int(value)
        if p_type.lower == "boolean":
            if value.lower == 'true':
                return True
            if value.lower == 'false':
                return False
            raise ValueError(f"Invalid boolean: {value!r}")
        return str(value)
    except ValueError as e:
        raise ValueError(f"Invalid {p_type}: {value!r}") from e


class GenerationLogger:
    """Logs prompts and generated tokens during decoding"""
    def token(self, text: str) -> None:
        """logs generated token"""
        print(text, end="", flush=True)

    def prompt(self, text: str) -> None:
        """logs user prompt"""
        print(f"\nPrompt: {text}")
