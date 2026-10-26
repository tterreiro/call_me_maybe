*This project has been created as part of the 42 curriculum by hde_andr.*

# Call Me Maybe

## Description

Call Me Maybe is an intelligent function-calling system built on top of Large Language Models (LLMs). Small-scale open-source language models frequently fail to produce structured, valid JSON outputs when relying solely on traditional prompt engineering or standard free-form generation. 

The primary objective of this project is to eliminate formatting errors and schema violations by implementing constrained decoding. By intercepting and guiding the model's sampling loop token-by-token during inference, this system guarantees that the generated text strictly adheres to specified function signatures and valid JSON primitives.

## Instructions

### Prerequisites & Dependencies

This project uses `uv` for package management and environment isolation.

1. **Install dependencies and setup environment:**
   ```bash
   make install
   ```

### Execution

To run the function-calling pipeline using the default input test cases and function definitions:
```bash
make run
```

Alternatively, you can manually execute the module and pass custom input/output paths:
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/results.json
```

### Development Commands

* **Linting & Type Checking:**
  ```bash
  make lint
  ```
  Runs static analysis using flake8 and strict type validation with mypy.

* **Debugging:**
  ```bash
  make debug
  ```
  Launches the execution using Python's interactive debugger (pdb).

* **Clean Cache:**
  ```bash
  make clean
  ```
  Removes temporary build artifacts, .venv, and __pycache__ directories.

## Algorithm Explanation

The core mechanism of this project is Constrained Decoding via Logit Masking, which operates directly within the LLM generation loop:

1. Prompt Construction: The natural language user query and available function schemas are formatted into a structured prompt context.

2. Logit Extraction: At each generation step, the base LLM outputs a raw vector of logits corresponding to every token in its vocabulary.

3. Prefix Matching via Trie: Function names, argument keys, and structural JSON tokens (e.g., quotes, colons, braces) are compiled into a Trie data structure. Tokens that do not match a valid next step in the Trie are identified.

4. Type-Based Character Masking: When generating parameter values, active type constraints (e.g., restricting integer values to numerical digits or boolean values to true/false) filter out out-of-bounds vocabulary tokens.

5. Masking Process: Invalid token logits are set to negative infinity (-inf). This forces the sampling mechanism (greedy decoding or softmax) to draw exclusively from valid, syntactically correct tokens.


## Design Decisions

* Single-Step Constrained Generation: Instead of relying on multi-turn retries when invalid JSON is produced, constraints are evaluated dynamically per token to eliminate retries altogether and optimize performance.

* Trie Structure for Schema Enforcement: Using a Trie allows prefix lookup in $\mathcal{O}(K)$ time (where $K$ is the length of the string), allowing rapid masking of large vocabularies without slowing down inference.

* Modular Pipeline Architecture: Clean separation of concerns between input validation (file_handler.py), core decoding logic (pipeline.py), and CLI entry points (__main__.py).

## Performance Analysis

* Format Reliability: 100% JSON compliance. Invalid JSON syntax or missing required keys are mathematically impossible under the enforced token mask.
* Accuracy: Achieves high accuracy on standard tool-use test suites using compact LLM backends (such as Qwen/Qwen3-0.6).
* Inference Speed: Introduces minimal latency overhead over standard generation, as logit masking operations are vectorized over token tensors using standard PyTorch operations.

## Challenges Faced

1. **Subword Tokenizer Alignment:** 
   * Problem: Tokenizers often pair structural symbols (like " or ,) together with preceding/following whitespace into single subword tokens, breaking exact string-matching logic.
   * Solution: Built dynamic vocabulary tokenizers that evaluate multi-character subwords against partial string matches in the state machine before applying masks.

2. **Primitive Type Conversion:** 
   * Problem: Decoded string outputs needed to be reliably cast back into native Python types (int, float, bool) without runtime exceptions.
   * Solution: Enforced strict value validation masks during token generation so that final string extractions are guaranteed to be parseable by standard type constructors.

## Testing Strategy

Validation was conducted across several layers:

* Schema Conformance Tests: Evaluated model outputs against diverse JSON schemas containing string, numerical, and array parameters.
* Static Analysis: Strict mypy and flake8 checks to enforce static typing and code style across the codebase.
* Edge Case Datasets: Tested prompts requiring regex extraction, negative numbers, nested formatting, and edge-case function names.

## Example Usage

### Input Definition (`functions_definition.json`)
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    }
  }
]
```

### Input Query (`function_calling_tests.json`)
```json
[
  {
    "prompt": "What is the sum of 265 and 345?"
  }
]
```

### Output Generation (`results.json`)
```json
[
  {
    "name": "fn_add_numbers",
    "arguments": {
      "a": 265,
      "b": 345
    }
  }
]
```


## Resources
Github repositories
Peers
Qwen3-0.6B Model Card

### AI Assistance
Ai was used to help detect mistakes and to understand the subject