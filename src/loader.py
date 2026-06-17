import json
import sys
from typing import Dict, List
from pydantic import TypeAdapter
from exceptions import LoadingError
from models import FunctionSchema, PromptSchema

def load_and_validate_functions(fn_filepath: str) -> List[FunctionSchema]:

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
    
def load_vocabulary(vocab_path: str) -> List[Dict]:
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
    vocab = json.loads(raw_vocab)
    return vocab
