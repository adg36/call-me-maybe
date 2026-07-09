"""Entry point for the constrained function-calling pipeline.

This module parses command-line arguments, loads and validates the input
schemas and prompts, initializes the language model, and runs the
constrained decoding pipeline to generate structured function calls.
"""

import argparse
import os
import sys
from exceptions import LoadingError
from generate import Pipeline
from loader import load_and_validate_functions, load_and_validate_prompts
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json")
    parser.add_argument(
            "--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")
    args = parser.parse_args()

    if not args.functions_definition:
        print("Invalid functions_definition path.")
        sys.exit(1)
    if not args.input:
        print("Invalid input path.")
        sys.exit(1)
    if not args.output:
        print("Invalid output path.")
        sys.exit(1)

    fn_filepath = args.functions_definition
    pr_filepath = args.input
    output_filepath = args.output
    
    if os.path.isdir(output_filepath):
        print("Error: Output path must be a file, not a directory.")
        sys.exit(1)

    try:
        validated_functions = load_and_validate_functions(fn_filepath)
        validated_prompts = load_and_validate_prompts(pr_filepath) 

        model = Small_LLM_Model(device="cpu")
        pipeline = Pipeline(
                model,
                validated_prompts,
                validated_functions,
                output_filepath
        )
        pipeline.generate_function_call()
    except (RuntimeError, LoadingError) as e:
        print(f"Error: {e}")
        sys.exit(1)
