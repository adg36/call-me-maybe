from enum import Enum

class State(Enum):
    START = 1
    EXPECT_PROMPT_KEY = 2
    EXPECT_PROMPT = 3
    EXPECT_NAME_KEY = 4
    EXPECT_FUNCTION_NAME = 5
    EXPECT_PARAMETERS_KEY = 6
    EXPECT_PARAMETERS = 7
    DONE = 8
