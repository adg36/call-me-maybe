import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model

if __name__ == "__main__":

    prompt = """
    You are a function calling system.
    Available functions:
    - fn_add_numbrs(a, b)
    - fn_greet(name)

    User: What is the sum of 2 and 3?

    Answer:
    """

    model = Small_LLM_Model()
    tokens = model.encode(prompt)[0].tolist()

    output = []

    for _ in range(50):
        logits = model.get_logits_from_input_ids(tokens)
        next_token = np.argmax(logits)

        tokens.append(int(next_token))
        output.append(next_token)

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocab = f.read()
    print("Vocabulary: ", vocab[:150])
