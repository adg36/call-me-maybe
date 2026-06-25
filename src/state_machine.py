from enum import Enum

class State(Enum):
    START = 1
    LEADING_SPACES = 2
    EXPECT_NAME_KEY = 3
    EXPECT_COLON = 4
    EXPECT_FUNCTION_NAME = 5
    DONE = 6
