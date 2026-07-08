from typing import Any, Dict, List
import json
import os
import sys
import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from models import FunctionSchema, PromptSchema, Parameter
from state_machine import State, NEXT_STATE


class Pipeline:
    """
    Generate grammar-constrained function calls from natural language prompts.

    The pipeline coordinates prompt encoding, constrained decoding,
    finite-state machine transitions, and JSON output generation using
    the provided language model and function schemas.
    """
    def __init__(
        self,
        model: Small_LLM_Model,
        prompts: List[PromptSchema],
        functions: List[FunctionSchema],
        output_filepath: str
    ) -> None:
        self.model = model
        self.prompts = prompts
        self.functions = functions
        self.output_filepath = output_filepath
        self.parameter_count = 0
        self.current_parameter = 0

    def allowed_strings(
        self,
        state: State,
        function_name: str | None,
        current_string: str
    ) -> List[str | Any]:
        """
        Return the valid string continuations for the current parser state.

        The finite-state machine determines which JSON fragments are valid
        at each stage of constrained decoding. Depending on the current state,
        this method returns the set of strings that may legally follow the
        generated output.
        These strings are later converted into valid tokenizer tokens before
        the language model selects the next token.

        Args:
            state: Current state of the finite-state machine.
            function_name: Name of the selected function, if already generated.
            current_string: Part of the current string generated so far.

        Returns:
            A list of valid string continuations for the current parser state.
            An empty list indicates that no fixed string constraint applies
            and the model may continue generating freely within the current
            value.
        """
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
                self.parameter_count = len(params)
                if params is not None:
                    param_keys = list(params.keys())
                    return [f"\"{param_keys[self.current_parameter]}\":"]
            return []
        if state == State.EXPECT_NUMBER_START:
            return [
                "0", "1", "2", "3", "4", "5",
                "6", "7", "8", "9", "-", "."
            ]
        if state == State.EXPECT_NUMBER_CONT:
            if self.current_parameter < (self.parameter_count - 1):
                return [
                    "0", "1", "2", "3", "4", "5", "6",
                    "7", "8", "9", ","
                ]
            if self.current_parameter == (self.parameter_count - 1):
                return [
                    "0", "1", "2", "3", "4", "5", "6",
                    "7", "8", "9", "}"
                ]
        if state == State.EXPECT_STRING:
            if current_string == "":
                return ['"']
            return []
        if state == State.DONE:
            return ["}"]
        if state == State.FINISHED:
            return []
        return []

    def allowed_tokens(
        self,
        allowed_strings: List[str | Any],
        remaining: List[str]
    ) -> List[int]:
        """
        Return the tokenizer IDs that are valid at the current decoding step.

        The grammar is defined over strings, while the language model generates
        tokens. This method bridges the two by determining which vocabulary
        tokens remain compatible with at least one valid grammar continuation.

        When a grammar element has already been partially generated, only the
        remaining suffixes are considered. Duplicate token IDs are removed
        before returning the final set of allowed tokens.

        Args:
            allowed_strings: Complete grammar strings that are valid in the
                current parser state.
            remaining: Unmatched suffixes of partially generated strings.

        Returns:
            A list of token IDs that may be selected without violating the
            grammar.
        """

        allowed_token_ids = []
        if not remaining:
            remaining = list(allowed_strings)
            for s in allowed_strings:
                ids = self.model.encode(s)[0].tolist()
                for token_id in ids:
                    if (s.startswith(self.model.decode(token_id))
                            and token_id not in allowed_token_ids):
                        allowed_token_ids.append(token_id)
            return allowed_token_ids
        else:
            for suffix in remaining:
                ids = self.model.encode(suffix)[0].tolist()
                for token_id in ids:
                    if (suffix.startswith(self.model.decode(token_id))
                            and token_id not in allowed_token_ids):
                        allowed_token_ids.append(token_id)
            return allowed_token_ids

    def sample_one_token(
        self,
        allowed_token_ids: List[int],
        tokens: List[int]
    ) -> Any:
        """Generate the next token while enforcing grammar constraints.

        The language model produces logits for the next token based on the
        current input sequence. If grammar constraints are active, all
        invalid token logits are masked before selecting the
        highest-probability valid token. Otherwise, the highest-probability
        token from the full vocabulary is selected.

        The chosen token is appended to the current token sequence and
        returned as decoded text.

        Args:
            allowed_token_ids: Token IDs permitted by the current grammar
                state.
            tokens: Token sequence generated so far.

        Returns:
            The decoded text corresponding to the selected token.
        """
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

    def get_parameters(
        self,
        function_name: str
    ) -> Dict[str, Parameter]:
        """Return the parameter schema for a given function.

        Searches the available function definitions and retrieves the parameter
        schema associated with the specified function.

        Args:
            function_name: Name of the function whose parameters should be
                returned.

        Returns:
            A dictionary mapping parameter names to their corresponding
            parameter definitions. Returns an empty dictionary if the function
            is not found.
        """
        for function in self.functions:
            if function.name == function_name:
                return function.parameters
        return {}

    def generate_function_call(self) -> None:
        """Generate constrained function calls for all input prompts.

        Each prompt is processed independently through the constrained decoding
        pipeline. Successfully generated function calls are collected and
        written to the configured output file once all prompts have been
        processed.

        If generation fails for any prompt, an error message is displayed
        and the program terminates.

        Raises:
            SystemExit: If a prompt cannot be decoded to a valid function call.
        """
        output = []

        for i, prompt in enumerate(self.prompts):

            print(f"PROMPT {i+1}/{len(self.prompts)}: {prompt.prompt}")

            try:
                result = self.generate_one_prompt(prompt)
                output.append(result)
            except RuntimeError as e:
                print(f"Error: {e}")
                sys.exit(1)
            print(f"ANSWER: {result}")
        self.write_output(output)

    def generate_one_prompt(self, prompt: PromptSchema) -> Dict[str, Any]:
        """Generate a constrained function call for a single prompt.

        Constructs an instruction prompt for the language model and performs
        grammar-constrained decoding one token at a time. At each decoding
        step, the FSM determines the valid grammar continuations,
        which are converted into allowed tokenizer tokens before selecting the
        next model prediction.

        The generated JSON object is validated, converted into the expected
        output format, and returned.

        Args:
            prompt: Natural language prompt describing the requested function
                call.

        Returns:
            A dictionary containing the original prompt, the selected function
            name, and its generated parameters.

        Raises:
            RuntimeError: If generation exceeds safety limits or produces an
                invalid function call.
        """
        generated_tokens = 0

        # 1. PROMPT
        my_prompt = f"""
        Choose the function whose description
        best matches the user request.
        Return only a JSON object.

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
        remaining: List[str] = []
        current_string = ""
        function_name: None | str = None
        self.current_parameter = 0

        while True:
            allowed_strings = self.allowed_strings(
                state, function_name, current_string)
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
            generated_tokens += 1
            if generated_tokens > 200:
                raise RuntimeError("Generation exceeded maximum token limit.")
            current_string += text
            if state in (State.EXPECT_NUMBER_START, State.EXPECT_NUMBER_CONT):
                if len(current_string) > 20:
                    raise RuntimeError("Failed to generate a valid function "
                                       "call: numeric parameter exceeded "
                                       "maximum length.")
            answer.append(text)
            if state == State.EXPECT_NUMBER_START:
                if current_string.endswith("."):
                    state = State.EXPECT_NUMBER_CONT
            elif state == State.EXPECT_NUMBER_CONT:
                if current_string.endswith(","):
                    self.current_parameter += 1
                    state = State.EXPECT_PARAM_KEY
                    current_string = ""
                    remaining = []
                    text = ""
                elif current_string.endswith("}"):
                    state = NEXT_STATE[state]
                    current_string = ""
                    remaining = []
                    text = ""
            elif state == State.EXPECT_STRING:
                if (current_string.endswith("',")
                        or current_string.endswith('",')):
                    self.current_parameter += 1
                    state = State.EXPECT_PARAM_KEY
                    current_string = ""
                    remaining = []
                    text = ""
                if current_string.endswith("\n"):
                    state = State.FINISHED
                    break
            elif len(candidates) == 1 and current_string == candidates[0]:
                if state == State.EXPECT_FUNCTION_NAME:
                    function_name = current_string[1:-1]
                elif state == State.EXPECT_PARAM_KEY:
                    if function_name is not None:
                        params = self.get_parameters(function_name)
                        self.parameter_count = len(params)
                        if params:
                            current_param = list(
                                    params.values()
                                    )[self.current_parameter].type
                            if current_param == "number":
                                state = State.EXPECT_NUMBER_START
                            elif current_param == "integer":
                                state = State.EXPECT_NUMBER_CONT
                            elif current_param == "string":
                                state = State.EXPECT_STRING
                elif state == State.DONE:
                    break
                if state not in (
                        State.EXPECT_NUMBER_START,
                        State.EXPECT_NUMBER_CONT,
                        State.EXPECT_STRING
                ):
                    state = NEXT_STATE[state]
                current_string = ""
                remaining = []
                text = ""
            else:
                remaining = self.check_remaining(
                    allowed_strings, current_string)
        obj = json.loads("".join(answer))
        if "path" in obj["parameters"].keys():
            obj["parameters"]["path"] = obj["parameters"]["path"].lstrip()
        result = {
                "prompt": prompt.prompt,
                "name": obj["name"],
                "parameters": obj["parameters"]
        }
        return result

    def check_remaining(
        self,
        allowed_strings: List[str | Any],
        current_string: str
    ) -> List[str]:
        """Return the unmatched suffixes of all compatible grammar strings.

        Filters the allowed grammar strings to those matching the current
        generated prefix and returns only the remaining suffix of each match.

        Args:
            allowed_strings: Valid strings for the current decoder state.
            current_string: Prefix generated so far.

        Returns:
            A list of suffixes representing the valid continuations of the
            current prefix.
        """
        return [
            s[len(current_string):]
            for s in allowed_strings
            if s.startswith(current_string)
        ]

    def write_output(self, output: List[Any]) -> None:
        """Write the generated function calls to the output JSON file.

        Creates the output directory if it does not already exist and writes
        the generated function calls as formatted JSON.

        Args:
            output: List of generated function call objects to serialize.
        """
        os.makedirs(os.path.dirname(self.output_filepath), exist_ok=True)

        with open(self.output_filepath, "w") as f:
            json.dump(output, f, indent=2)
