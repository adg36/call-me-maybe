from typing import List
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from models import FunctionSchema, PromptSchema
from state_machine import State


class Pipeline:
    def __init__(self,
                 model: Small_LLM_Model,
                 prompts: List[PromptSchema],
                 functions: List[FunctionSchema]
                 ) -> None:
        self.model = model
        self.prompts = prompts
        self.functions = functions

    def allowed_strings(self, state) -> List[str]:
        if state == State.START:
            return ["{\n"]
        if state == State.EXPECT_NAME_KEY:
            return ['  "name"']
        if state == State.EXPECT_COLON:
            return [": "]
        if state == State.EXPECT_FUNCTION_NAME:
            return [f"{function.name}" for function in self.functions]
        if state == State.DONE:
            return ["\n}"]

    def sample_one_token(self, tokens, state) -> List[str]:
        output = []
        allowed_token_ids = []

        logits = self.model.get_logits_from_input_ids(tokens)
        masked_logits = np.full(len(logits), -np.inf)
        allowed_strings = self.allowed_strings(state)
        for s in allowed_strings:
            ids = self.model.encode(s)[0].tolist()
            allowed_token_ids.extend(ids)
        for token in allowed_token_ids:
            masked_logits[token] = logits[token]
        # apply mask to logits, e.g. logits[~allowed] = -np.inf
        next_token = np.argmax(masked_logits)
        tokens.append(int(next_token))
        output.append(next_token)
        text = self.model.decode(output)
        return text

    def sample_tokens(self, tokens, state, max_tokens=30) -> List[str]:
        output = []
        allowed_token_ids = []
        eos_token_id = 151645

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(tokens)
            masked_logits = np.full(len(logits), -np.inf)
            allowed_strings = self.allowed_strings(state)
            for s in allowed_strings:
                ids = self.model.encode(s)[0].tolist()
                allowed_token_ids.extend(ids)
            for token in allowed_token_ids:
                masked_logits[token] = logits[token]
            # apply mask to logits, e.g. logits[~allowed] = -np.inf
            next_token = np.argmax(masked_logits)
            if next_token == eos_token_id:
                break
            tokens.append(int(next_token))
            output.append(next_token)
        text = self.model.decode(output)
        return text

    def generate_function_call(self) -> None:

        for prompt in self.prompts:
            # 1. PROMPT
            my_prompt = f"""
            You are a function calling system.
            Choose exactly ONE function from the list
            below and return ONLY a valid JSON object.
            Do not write anything else.

            Available functions:
            {self.functions}

            Output format:
            {{
              "name": "<function_name>",
              "parameters": {{ ... }}
            }}

            User request:
            {prompt.prompt}
            """

            # vocab_path = model.get_path_to_vocab_file()
            # vocab = load_vocabulary(vocab_path)
            # curly_left_id = vocab["{"]

            # 2. TOKENIZATION
            state = State.START
            tokens = self.model.encode(my_prompt)[0].tolist()

            while True:
                text = self.sample_one_token(tokens, state)
                print(text)
                if state == State.START:
                    state = State.EXPECT_NAME_KEY
                    print(f"State is now {state.name}")
                elif state == State.EXPECT_NAME_KEY:
                    state = State.EXPECT_COLON
                    print(f"State is now {state.name}")
                elif state == State.EXPECT_COLON:
                    state = State.EXPECT_FUNCTION_NAME
                    print(f"State is now {state.name}")
                elif state == State.EXPECT_FUNCTION_NAME:
                    state = State.DONE
                    print(f"State is now {state.name}")
                elif state == State.DONE:
                    break
        print("End of loop")
