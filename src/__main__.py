"""Entry point for the constrained function-calling pipeline.

This module parses command-line arguments, loads and validates the input
schemas and prompts, initializes the language model, and runs the
constrained decoding pipeline to generate structured function calls.
"""

import argparse
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

    fn_filepath = args.functions_definition
    pr_filepath = args.input
    output_filepath = args.output

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
