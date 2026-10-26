#!/usr/bin/env python3
from pathlib import Path
import argparse
from llm_sdk import Small_LLM_Model
from src import (
    FileLoader, FunctionCallGenerator, CallResult, EmptyDataError
    )
import json


def main() -> None:
    """
    Parse command line flags, execute generation pipeline, and save outputs.

    Raises:
        EmptyDataError: Handled internally if an input file is empty.
        json.JSONDecodeError: Handled internally if input JSON is malformed.
        OSError: Handled internally on system I/O errors.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        "-functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        "-input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        "-output",
        default="data/output/function_calling_results.json"
    )
    args = parser.parse_args()

    file_loader = FileLoader()
    try:
        file_loader.load_func_def(Path(args.functions_definition))
        file_loader.load_prompt(Path(args.input))
    except EmptyDataError as e:
        print(f"Error: file not found = {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON = {e}")
        return
    except (PermissionError, OSError) as e:
        print(f"Error: file system = {e}")
        return

    llm = Small_LLM_Model()
    generator = FunctionCallGenerator(file_loader.func_definitions, llm)
    results: list[CallResult] = []
    for prompt in file_loader.prompts:
        try:
            res: CallResult = generator.run(prompt.prompt)
            results.append(res)
            print()
        except ValueError as e:
            print(f"\nError: {e}")
            continue
    try:
        file_loader.write_call_results(results, Path(args.output))
    except OSError as e:
        print(f"Error: could not write to {args.output} - {e}")
        return


if __name__ == "__main__":
    main()
