# All reward functions should return a list of values ranging from 0.0 to 1.0, reward weighting is handled in the trainer
from .diff import (
    sr_diff_format_reward_func,
    sr_diff_similarity_reward_func,
    unified_diff_similarity_reward_func,
    unified_diff_file_match_reward_func,
    unified_diff_similarity_reward_func_test,
)
from .categorical import categorical_correctness_reward_func
from .reasoning import (
    partial_reasoning_format_reward_func,
    strict_reasoning_format_reward_func,
)
from .terminal import terminal_debugging_habits_reward_func
