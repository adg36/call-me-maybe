from typing import List
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from models import FunctionSchema, PromptSchema
from loader import load_vocabulary
from state_machine import State


def allowed_strings(state, functions):
    if state == State.START:
        return ["{\n"]
    if state == State.EXPECT_NAME_KEY:
        return ['  "name"']
    if state == State.EXPECT_COLON:
        return [": "]
    if state == State.EXPECT_FUNCTION_NAME:
        return [f"{function.name}" for function in functions]
    if state == State.DONE:
        return ["\n}"]

def generate_function_call(
        model: Small_LLM_Model,
        prompts: List[PromptSchema],
        functions: List[FunctionSchema]
        ) -> None:

    # 1. PROMPT
    for prompt in prompts:
        my_prompt = f"""
        You are a function calling system.
        Choose exactly ONE function from the list
        below and return ONLY a valid JSON object.
        Do not write anything else.

        Available functions:
        {functions}

        Output format:
        {{
          "name": "<function_name>",
          "parameters": {{ ... }}
        }}

        User request:
        {prompt.prompt}
        """

        vocab_path = model.get_path_to_vocab_file()
        vocab = load_vocabulary(vocab_path)
        eos_token_id = 151645
        # curly_left_id = vocab["{"]

        # 2. TOKENIZATION
        tokens = model.encode(my_prompt)[0].tolist()

        output = []

        # 3. LOGITS
        logits = model.get_logits_from_input_ids(tokens)
        # masked_logits = np.full(len(logits), -np.inf)
        # masked_logits[curly_left_id] = logits[curly_left_id]

        # 4. TOKEN SELECTION - CONSTRAINED DECODING HAPPENS HERE!!
        # next_token = np.argmax(masked_logits)
        # tokens.append(int(next_token))
        # output.append(next_token)

        state = State.START
        while True:
            if state == State.START:
                allowed = allowed_strings(state, functions)
                allowed_tokens = model.encode(allowed)[0].tolist()
                print(f"{state}\n{allowed_tokens}")
                decoded = model.decode(allowed_tokens)
                print(decoded)
                state = State.EXPECT_NAME_KEY
            elif state == State.EXPECT_NAME_KEY:
                allowed = allowed_strings(state, functions)
                allowed_tokens = model.encode(allowed)[0].tolist()
                print(f"{state}\n{allowed_tokens}")
                for token_list in allowed_tokens:
                    decoded = [model.decode(token) for token in token_list]
                    print("Decoded token: ", decoded)
                state = State.EXPECT_COLON
            elif state == State.EXPECT_COLON:
                allowed = allowed_strings(state, functions)
                allowed_tokens = model.encode(allowed)[0].tolist()
                print(f"{state}\n{allowed_tokens}")
                for token_list in allowed_tokens:
                    decoded = [model.decode(token) for token in token_list]
                    print("Decoded token: ", decoded)
                state = State.EXPECT_FUNCTION_NAME
            elif state == State.EXPECT_FUNCTION_NAME:
                allowed = allowed_strings(state, functions)
                allowed_tokens = [model.encode(word)[0].tolist() for word in allowed]
                print(f"{state}\n{allowed_tokens}")
                for token_list in allowed_tokens:
                    decoded = [model.decode(token) for token in token_list]
                    print("Decoded token: ", decoded)
                state = State.DONE
            elif state == State.DONE:
                allowed = allowed_strings(state, functions)
                allowed_tokens = model.encode(allowed)[0].tolist()
                print(f"{state}\n{allowed_tokens}")
                for token_list in allowed_tokens:
                    decoded = [model.decode(token) for token in token_list]
                    print("Decoded token: ", decoded)
                break
       # for _ in range(30):
       #     logits = model.get_logits_from_input_ids(tokens)
            # compute mask here  --> allowed tokens
            # apply mask to logits, e.g. logits[~allowed] = -np.inf
       #     next_token = np.argmax(logits)
       #     if next_token == eos_token_id:
       #         break
       #     tokens.append(int(next_token))
       #     print(model.decode([next_token]))
       #     output.append(next_token)
       # text = model.decode(output)
