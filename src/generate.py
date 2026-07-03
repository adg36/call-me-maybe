from typing import Any, Dict, List
import numpy as np
from llm_sdk import Small_LLM_Model
from models import FunctionSchema, PromptSchema, Parameter
from state_machine import State, NEXT_STATE


class Pipeline:
    def __init__(self,
                 model: Small_LLM_Model,
                 prompts: List[PromptSchema],
                 functions: List[FunctionSchema]
                 ) -> None:
        self.model = model
        self.prompts = prompts
        self.functions = functions
        self.current_parameter = 0

    def allowed_strings(self,
                        state: State,
                        function_name: str | None
                        ) -> List[str | Any]:
        if state == State.START:
            return ["{"]
        if state == State.EXPECT_NAME_KEY:
            return ["\"name\": "]
        if state == State.EXPECT_FUNCTION_NAME:
            return [f"\"{function.name}\"" for function in self.functions]
        if state == State.EXPECT_PARAMETERS_KEY:
            return [",\"parameters\": {"]
        if state == State.EXPECT_PARAM_KEY:
            if function_name is not None:
                params = self.get_parameters(function_name)
                if params is not None:
                    param_keys = list(params.keys())
            return [f"\"{param_keys[self.current_parameter]}\":"]
        if state == State.EXPECT_NUMBER_START:
            return [
                "0", "1", "2", "3", "4", "5",
                "6", "7", "8", "9", "-"
            ]
        if state == State.EXPECT_NUMBER_CONT:
            return [
                "0", "1", "2", "3", "4", "5", "6",
                "7", "8", "9", ".", ",", "}"

            ]
        if state == State.EXPECT_STRING:
            return []
        if state == State.DONE:
            return ['}']
        if state == State.FINISHED:
            return
        return []

    def allowed_tokens(self, allowed_strings, remaining) -> List[int]:
        allowed_token_ids = []

        if not remaining:
            remaining = list(allowed_strings)
            for s in allowed_strings:
                ids = self.model.encode(s)[0].tolist()
                for token_id in ids:
                    if (s.startswith(self.model.decode(token_id))
                            and token_id not in allowed_token_ids):
                        allowed_token_ids.append(token_id)
        else:
            for suffix in remaining:
                ids = self.model.encode(suffix)[0].tolist()
                for token_id in ids:
                    if (suffix.startswith(self.model.decode(token_id))
                            and token_id not in allowed_token_ids):
                        allowed_token_ids.append(token_id)

        return allowed_token_ids

    def sample_one_token(self, allowed_token_ids, tokens) -> str:
        logits = self.model.get_logits_from_input_ids(tokens)
        if allowed_token_ids:
            masked_logits = np.full(len(logits), -np.inf)
            for token_id in allowed_token_ids:
                masked_logits[token_id] = logits[token_id]
            next_token = np.argmax(masked_logits)
        else:
            next_token = np.argmax(logits)
        tokens.append(int(next_token))
        text = self.model.decode(next_token)
        return text

    def get_parameters(self, function_name) -> Dict[str, Parameter] | None:
        for function in self.functions:
            if function.name == function_name:
                return function.parameters
        return None

    def generate_function_call(self) -> None:
        output = []

        for prompt in self.prompts:
            # 1. PROMPT
            my_prompt = f"""
            Choose only ONE function from the list
            and return ONLY a valid JSON object.

            Functions:
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
            function_name: None | str = None
            self.current_parameter = 0

            while True:
                allowed_strings = self.allowed_strings(
                    state, function_name)
                candidates = [
                    s for s in allowed_strings
                    if s.startswith(current_string)
                ]
                allowed_token_ids = self.allowed_tokens(
                    allowed_strings, remaining)
                if len(allowed_token_ids) == 1:
                    tokens.append(int(allowed_token_ids[0]))
                    text = self.model.decode(allowed_token_ids)
                else:
                    text = self.sample_one_token(allowed_token_ids, tokens)
                current_string += text
                answer.append(text)
                if state == State.EXPECT_NUMBER_START:
                    state = State.EXPECT_NUMBER_CONT
                elif state == State.EXPECT_NUMBER_CONT:
                    if current_string[-1] == ",":
                        self.current_parameter += 1
                        state = State.EXPECT_PARAM_KEY
                        current_string = ""
                        remaining = None
                        text = ""
                    elif current_string[-1] == "}":
                        state = NEXT_STATE[state]
                        current_string = ""
                        remaining = None
                        text = ""
                elif state == State.EXPECT_STRING:
                    if current_string.endswith("\n"):
                        state = State.FINISHED
                        break
                elif len(candidates) == 1 and current_string == candidates[0]:
                    if state == State.EXPECT_FUNCTION_NAME:
                        function_name = current_string[1:-1]
                    elif state == State.EXPECT_PARAM_KEY:
                        if function_name is not None:
                            params = self.get_parameters(function_name)
                            if params:
                                for _, param in params.items():
                                    if param.type == "number":
                                        state = State.EXPECT_NUMBER_START
                                    elif param.type == "string":
                                        state = State.EXPECT_STRING
                    elif state == State.DONE:
                        break
                    if state not in (
                            State.EXPECT_NUMBER_START, State.EXPECT_STRING
                    ):
                        state = NEXT_STATE[state]
                    current_string = ""
                    remaining = None
                    text = ""
                else:
                    remaining = self.check_remaining(
                        allowed_strings, current_string)
            print("".join(answer))
            output.append("".join(answer))
        print(output)

    def check_remaining(self, allowed_strings, current_string) -> List[str]:
        return [
            s[len(current_string):]
            for s in allowed_strings
            if s.startswith(current_string)
        ]
