"""Finite-state machine used during constrained decoding.

The decoder progresses through these states while generating a valid
JSON function call. State transitions enforce the expected output
grammar throughout token generation.
"""

from enum import Enum, auto


class State(Enum):
    """Represent the decoder states during constrained generation."""

    START = auto()
    EXPECT_NAME_KEY = auto()
    EXPECT_FUNCTION_NAME = auto()
    EXPECT_PARAMETERS_KEY = auto()
    EXPECT_PARAM_KEY = auto()
    EXPECT_NUMBER_START = auto()
    EXPECT_NUMBER_CONT = auto()
    EXPECT_STRING = auto()
    DONE = auto()
    FINISHED = auto()

#: Mapping of states with deterministic transitions.
#:
#: States that require additional runtime decisions (such as parameter
#: types or parameter counts) are handled separately by the decoder.


NEXT_STATE = {
    State.START: State.EXPECT_NAME_KEY,
    State.EXPECT_NAME_KEY: State.EXPECT_FUNCTION_NAME,
    State.EXPECT_FUNCTION_NAME: State.EXPECT_PARAMETERS_KEY,
    State.EXPECT_PARAMETERS_KEY: State.EXPECT_PARAM_KEY,
    State.EXPECT_NUMBER_START: State.EXPECT_NUMBER_CONT,
    State.EXPECT_NUMBER_CONT: State.DONE,
    State.EXPECT_STRING: State.DONE
}
