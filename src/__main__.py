import argparse
import json
import numpy as np
from pydantic import ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model
from models import FunctionSchema, PromptSchema


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json")
    parser.add_argument(
            "--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    args = parser.parse_args()

    fn_filepath = args.functions_definition
    pr_filepath = args.input

    # testing json loading and function structure
    with open(fn_filepath, "r") as fn_file:
        functions = json.load(fn_file)
    functions_json = json.dumps(functions, indent=2)

    for function in functions:
        try:
            fn_data = FunctionSchema(**function)
        except ValidationError as e:
            print(e)
            raise ValidationError

    with open(pr_filepath, "r") as pr_file:
        prompts = json.load(pr_file)

    model = Small_LLM_Model()

    for prompt in prompts:
        try:
            pr_data = PromptSchema(**prompt)
        except ValidationError as e:
            print(e)
            raise ValidationError

        # testing the model
        my_prompt = f"""
        You are a function calling system.
        You must choose exactly ONE function from the
        list below and return ONLY a valid JSON object.

        Available functions:
        {functions_json}

        Rules:
        - Output must be valid JSON only.
        - Do NOT write explanations.
        - Do NOT output multiple objects.
        - Do NOT include any extra keys like 'result'.
        - You must select exactly one function name.

        Output format:
        {{
            "name": "<function_name>",
            "parameters": {{}}
        }}

        User request:
        {pr_data.prompt}
        """

        tokens = model.encode(my_prompt)[0].tolist()

        output = []

        for _ in range(200):
            logits = model.get_logits_from_input_ids(tokens)
            next_token = np.argmax(logits)

            tokens.append(int(next_token))
            output.append(next_token)
        text = model.decode(output)
        print("Answer: ", text)

    # testing the vocabulary
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocab = f.read()
    parsed_vocab = json.loads(vocab)
    curly_brace_left = parsed_vocab["{"]
    """
