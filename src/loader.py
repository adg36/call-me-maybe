"""Utilities for loading and validating project input files.

This module provides helper functions to load the function definitions,
prompt definitions, and tokenizer vocabulary from disk. JSON files are
validated against the project's Pydantic schemas before being returned
to the caller.
"""

import json
import sys
from typing import Any, List
from pydantic import TypeAdapter
from exceptions import LoadingError
from models import FunctionSchema, PromptSchema


def load_and_validate_functions(fn_filepath: str) -> List[FunctionSchema]:
    """Load and validate function definitions from a JSON file.

    Read the function definition file, validate its contents against the
    FunctionSchema model, and return the validated objects.

    Raises:
        LoadingError: If the file cannot be opened.
        SystemExit: If the JSON content fails schema validation.
    """
    fn_adapter = TypeAdapter(List[FunctionSchema])

    try:
        with open(fn_filepath, "r") as fn_file:
            json_functions = fn_file.read()
    except PermissionError as e:
        raise LoadingError(
                "Functions file does not have reading permissions."
        ) from e
    except FileNotFoundError as e:
        raise LoadingError(
                "Functions file does not exist."
        ) from e

    try:
        validated_functions = fn_adapter.validate_json(json_functions)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    return validated_functions


def load_and_validate_prompts(pr_filepath: str) -> list[PromptSchema]:
    """Load and validate prompts from a JSON file.

    Read the prompt file, validate its contents against the PromptSchema
    model, and return the validated objects.

    Raises:
        LoadingError: If the file cannot be opened.
        SystemExit: If the JSON content fails schema validation.
    """
    pr_adapter = TypeAdapter(List[PromptSchema])

    try:
        with open(pr_filepath, "r") as pr_file:
            json_prompts = pr_file.read()
    except PermissionError as e:
        raise LoadingError(
                "Prompts file does not have reading permissions."
        ) from e
    except FileNotFoundError as e:
        raise LoadingError(
                "Prompts file does not exist."
        ) from e

    try:
        validated_prompts = pr_adapter.validate_json(json_prompts)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    return validated_prompts


def load_vocabulary(vocab_path: str) -> Any:
    """Load the tokenizer vocabulary from a JSON file.

    Read the vocabulary file, deserialize its JSON contents, and return
    the resulting object.

    Raises:
        LoadingError: If the file cannot be opened.
    """
    try:
        with open(vocab_path, "r") as f:
            raw_vocab = f.read()
    except PermissionError as e:
        raise LoadingError(
                "Vocabulary file does not have reading permissions."
        ) from e
    except FileNotFoundError as e:
        raise LoadingError(
                "Vocabulary file does not exist."
        ) from e
    try:
        vocab = json.loads(raw_vocab)
    except json.JSONDecodeError as e:
        raise LoadingError("Vocabulary file contains invalid JSON.") from e
    return vocab
