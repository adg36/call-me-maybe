from typing import List
import numpy as np
from llm_sdk import Small_LLM_Model
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

    def allowed_strings(self, state, prompt, function_name) -> List[str]:
        if state == State.START:
            return ["{"]
        if state == State.EXPECT_PROMPT_KEY:
            return ["\"prompt\": "]
        if state == State.EXPECT_PROMPT:
            return [f"\"{prompt.prompt}\""]
        if state == State.EXPECT_NAME_KEY:
            return [",\"name\": "]
        if state == State.EXPECT_FUNCTION_NAME:
            return [f"\"{function.name}\"" for function in self.functions]
        if state == State.EXPECT_PARAMETERS_KEY:
            return [",\"parameters\": "]
        if state == State.EXPECT_PARAMETERS:
            params = self.get_parameters(function_name)
            param_keys = list(params.keys())
            num_params = len(param_keys)
            res = ""
            for i in range(num_params):
                res += f"\"{param_keys[i]}\":"
                if i != num_params - 1:
                    res += ", "
            # res = ",".join(param_keys)
            return [f"{{{res}}}"]
        if state == State.DONE:
            return ['}']

    def allowed_tokens(self, allowed_strings, remaining) -> List[int]:
        allowed_token_ids = []

        if not remaining:
            remaining = list(allowed_strings)
            for s in allowed_strings:
                ids = self.model.encode(s)[0].tolist()
                for token_id in ids:
                    if s.startswith(self.model.decode(token_id)):
                        allowed_token_ids.append(token_id)
        else:
            for suffix in remaining:
                ids = self.model.encode(suffix)[0].tolist()
                for token_id in ids:
                    if suffix.startswith(self.model.decode(token_id)):
                        allowed_token_ids.append(token_id)

        return allowed_token_ids

    def sample_one_token(self, allowed_token_ids, tokens) -> str:
        logits = self.model.get_logits_from_input_ids(tokens)
        masked_logits = np.full(len(logits), -np.inf)
        for token_id in allowed_token_ids:
            masked_logits[token_id] = logits[token_id]
        next_token = np.argmax(masked_logits)
        tokens.append(int(next_token))
        text = self.model.decode(next_token)
        return text

    def get_parameters(self, function_name) -> tuple[str]:
        for function in self.functions:
            if function.name == function_name:
                return function.parameters
        return None

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

            # 2. TOKENIZATION
            tokens = self.model.encode(my_prompt)[0].tolist()
            state = State.START
            answer = []
            remaining = None
            current_string = ""
            function_name = None

            while True:
                # print(f"State: {state.name}")
                allowed_strings = self.allowed_strings(
                        state, prompt, function_name)
                # print(f"Allowed strings: {allowed_strings}")
                candidates = [
                        s for s in allowed_strings
                        if s.startswith(current_string)
                ]
                allowed_token_ids = self.allowed_tokens(
                        allowed_strings, remaining)
                # print(f"Allowed token ids: {allowed_token_ids}")
                text = self.sample_one_token(allowed_token_ids, tokens)
                # print("Text: ", text)
                current_string += text
                # print("Current string: ", current_string)
                answer.append(text)
                if len(candidates) == 1 and current_string == candidates[0]:
                    if state == State.START:
                        state = State.EXPECT_PROMPT_KEY
                        current_string = ""
                        remaining = None
                        text = ""
                    elif state == State.EXPECT_PROMPT_KEY:
                        state = State.EXPECT_PROMPT
                        current_string = ""
                        remaining = None
                        text = ""
                    elif state == State.EXPECT_PROMPT:
                        state = State.EXPECT_NAME_KEY
                        current_string = ""
                        remaining = None
                        text = ""
                    elif state == State.EXPECT_NAME_KEY:
                        state = State.EXPECT_FUNCTION_NAME
                        current_string = ""
                        remaining = None
                        text = ""
                    elif state == State.EXPECT_FUNCTION_NAME:
                        function_name = current_string[1:-1]
                        state = State.EXPECT_PARAMETERS_KEY
                        current_string = ""
                        remaining = None
                        text = ""
                    elif state == State.EXPECT_PARAMETERS_KEY:
                        state = State.EXPECT_PARAMETERS
                        current_string = ""
                        remaining = None
                        text = ""
                    elif state == State.EXPECT_PARAMETERS:
                        state = State.DONE
                        current_string = ""
                        remaining = None
                        text = ""
                    elif state == State.DONE:
                        break
                else:
                    remaining = self.check_remaining(
                            allowed_strings, current_string)
                    # print("Remaining: ", remaining)
            print("".join(answer))

    def check_remaining(self, allowed_strings, current_string) -> List[str]:
        return [
                s[len(current_string):]
                for s in allowed_strings
                if s.startswith(current_string)
        ]
