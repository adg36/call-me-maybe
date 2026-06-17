from typing import List
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from models import FunctionSchema, PromptSchema

def generate_function_call(
        model: Small_LLM_Model,
        prompts: List[PromptSchema],
        functions: List[FunctionSchema]
        ) -> None:

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

        tokens = model.encode(my_prompt)[0].tolist()

        output = []

        for _ in range(30):
            logits = model.get_logits_from_input_ids(tokens)
            next_token = np.argmax(logits)

            tokens.append(int(next_token))
            output.append(next_token)
        text = model.decode(output)
        print(text)
