import math
import unittest

from src.rewards.terminal import terminal_debugging_habits_reward_func


class TestTerminalDebuggingHabitsReward(unittest.TestCase):
    def assertApproxEqual(self, a: float, b: float, tol: float = 1e-6):  # type: ignore[override]
        self.assertTrue(math.isclose(a, b, rel_tol=0, abs_tol=tol), msg=f"{a} != {b}")

    def test_ls_early_success_rewarded(self):
        transcript = (
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "ls -la"}}\n'
            "</tool_call>\n"
            "user\n<tool_response>\n"
            "command succeeded\n"
            "</tool_response>\n"
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "echo done"}}\n'
            "</tool_call>\n"
            "user\n<tool_response>\n"
            "done\n"
            "</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 0.3)

    def test_ls_not_early_not_rewarded(self):
        transcript = (
            # First two are non-ls
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "echo x"}}\n'
            "</tool_call>\nuser\n<tool_response>\nx\n</tool_response>\n"
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "pwd"}}\n'
            "</tool_call>\nuser\n<tool_response>\n/\n</tool_response>\n"
            # Third is ls
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "ls -la"}}\n'
            "</tool_call>\nuser\n<tool_response>\ncommand succeeded\n</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 0.0)

    def test_search_early_success_rewarded(self):
        transcript = (
            # 1: echo
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "echo hi"}}\n'
            "</tool_call>\nuser\n<tool_response>\nhi\n</tool_response>\n"
            # 2: rg -n (success)
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "rg -n \'pattern\' src/file.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\n1:match\n</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 0.4)

    def test_search_early_failure_not_rewarded(self):
        transcript = (
            # 1: grep (failure)
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "grep -n pattern file.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 0.0)

    def test_slicing_any_rewarded(self):
        transcript = (
            # two neutral successful commands, then sed
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "echo hi"}}\n'
            "</tool_call>\nuser\n<tool_response>\nhi\n</tool_response>\n"
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "pwd"}}\n'
            "</tool_call>\nuser\n<tool_response>\n/\n</tool_response>\n"
            # sed appears later and succeeds
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "sed -n \'10,20p\' file.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\nlines\n</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 0.3)

    def test_cat_not_rewarded(self):
        transcript = (
            # cat succeeds but should not contribute
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "cat bigfile.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\n...\n</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 0.0)

    def test_malformed_tool_call_ignored(self):
        transcript = (
            # two neutral successful commands first
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "echo ok"}}\n'
            "</tool_call>\nuser\n<tool_response>\nok\n</tool_response>\n"
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "pwd"}}\n'
            "</tool_call>\nuser\n<tool_response>\n/\n</tool_response>\n"
            # malformed JSON (ignored)
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": rg -n \'oops\'}}\n'
            "</tool_call>\nuser\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
            # valid sed after malformed one
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "sed -n \'1,5p\' file.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\nlines\n</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 0.3)

    def test_outlier_combined_pipes_still_rewarded(self):
        transcript = (
            # two neutral successful commands first
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "echo ok"}}\n'
            "</tool_call>\nuser\n<tool_response>\nok\n</tool_response>\n"
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "pwd"}}\n'
            "</tool_call>\nuser\n<tool_response>\n/\n</tool_response>\n"
            # Complex pipeline with head and sed should still count
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "head -n 100 file.py | sed -n \'10,20p\'"}}\n'
            "</tool_call>\nuser\n<tool_response>\nlines\n</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 0.3)

    def test_early_mix_multiple_components(self):
        transcript = (
            # 1: ls -> success (0.3)
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "ls"}}\n'
            "</tool_call>\nuser\n<tool_response>\ncommand succeeded\n</tool_response>\n"
            # 2: rg -> success (0.4)
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "rg -l \'x\'"}}\n'
            "</tool_call>\nuser\n<tool_response>\nfile.py\n</tool_response>\n"
            # 3: sed later (0.3)
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "sed -n \'1,2p\' file.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\n1:a\n2:b\n</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 1.0)

    # Removed fallback test; reward only operates on completion transcript

    def test_failure_responses_not_rewarded(self):
        transcript = (
            # a neutral successful command first
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "echo hi"}}\n'
            "</tool_call>\nuser\n<tool_response>\nhi\n</tool_response>\n"
            # rg present in first two but fails; ensure no early-search credit
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "rg -n \'x\' file.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
            # Later a successful sed still yields 0.3 (third command)
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "sed -n \'1,2p\' file.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\n1:a\n2:b\n</tool_response>\n"
        )
        reward = terminal_debugging_habits_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(reward, 0.3)


if __name__ == "__main__":
    unittest.main()


