import json
import numpy as np
from pydantic import ValidationError
from llm_sdk.llm_sdk import Small_LLM_Model
from models import Parameter, FunctionDefinition


if __name__ == "__main__":

    # WRITE CLI SKELETON THAT PRINTS - SEE ARGPARSE:
    # - which input file was chosen
    # - which output file was chosen
    # - which function file was chosen



    # testing the model
    """
    prompt = "
    You are a function calling system.
    Available functions:
    - fn_add_numbers(a, b)
    - fn_greet(name)

    User: What is the sum of 2 and 3?

    Answer:
    "

    model = Small_LLM_Model()
    tokens = model.encode(prompt)[0].tolist()

    output = []

    for _ in range(20):
        logits = model.get_logits_from_input_ids(tokens)
        next_token = np.argmax(logits)

        tokens.append(int(next_token))
        output.append(next_token)

    # testing the vocabulary

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocab = f.read()
    parsed_vocab = json.loads(vocab)
    curly_brace_left = parsed_vocab["{"]
    """
    # testing json loading and function structure

    filepath = "data/input/functions_definition.json"
    with open(filepath, "r") as file:
        functions = json.load(file)

    for function in functions:
        try:
            data = FunctionDefinition(**function)
            print(data)
        except ValidationError as e:
            print(e)
