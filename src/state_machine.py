from enum import Enum

class State(Enum):
    START = 1
    EXPECT_NAME_KEY = 2
    EXPECT_COLON = 3
    EXPECT_FUNCTION_NAME = 4
    DONE = 5
