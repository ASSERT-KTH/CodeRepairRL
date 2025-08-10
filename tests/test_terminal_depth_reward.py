import math
import unittest

from src.rewards.terminal import terminal_exploration_depth_reward_func


class TestTerminalExplorationDepthReward(unittest.TestCase):
    def assertApproxEqual(self, a: float, b: float, tol: float = 1e-6):  # type: ignore[override]
        self.assertTrue(math.isclose(a, b, rel_tol=0, abs_tol=tol), msg=f"{a} != {b}")

    def test_chain_bonus_awarded(self):
        # search returns a path; later a non-precise sed on that path → +0.25
        transcript = (
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "rg -l \'WSIReader\'"}}\n'
            "</tool_call>\nuser\n<tool_response>\n"
            "src/a.py\n"
            "</tool_response>\n"
            # later any sed on that path (not precise) to avoid the +0.20 precise bonus
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "sed src/a.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\n...\n</tool_response>\n"
        )
        r = terminal_exploration_depth_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(r, 0.25)

    def test_success_ratio_bonus_linear(self):
        # First 6 calls: 3 successes, 3 failures → 0.20 * 0.5 = 0.10
        transcript = (
            # 1 success
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "echo ok"}}\n' "</tool_call>\n"
            "user\n<tool_response>\nok\n</tool_response>\n"
            # 2 failure
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "rg -n \'x\' file.py"}}\n' "</tool_call>\n"
            "user\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
            # 3 success
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "pwd"}}\n' "</tool_call>\n"
            "user\n<tool_response>\n/\n</tool_response>\n"
            # 4 failure
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "grep -n y missing.py"}}\n' "</tool_call>\n"
            "user\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
            # 5 success
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "echo ok2"}}\n' "</tool_call>\n"
            "user\n<tool_response>\nok2\n</tool_response>\n"
            # 6 failure
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "find . -name nofile"}}\n' "</tool_call>\n"
            "user\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
        )
        r = terminal_exploration_depth_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(r, 0.10)

    def test_precise_read_bonus(self):
        # Presence of precise slice anywhere → +0.20
        transcript = (
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "sed -n \'10,20p\' src/a.py"}}\n'
            "</tool_call>\nuser\n<tool_response>\nlines\n</tool_response>\n"
        )
        r = terminal_exploration_depth_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(r, 0.20)

    def test_scoped_search_bonus_flag(self):
        # Early search has -n flag → +0.15
        transcript = (
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "rg -n \'pattern\'"}}\n'
            "</tool_call>\nuser\n<tool_response>\nmatch\n</tool_response>\n"
        )
        r = terminal_exploration_depth_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(r, 0.15)

    def test_scoped_search_bonus_path(self):
        # Early search scoped by directory path → +0.15
        transcript = (
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "rg \'foo\' src/"}}\n'
            "</tool_call>\nuser\n<tool_response>\nfile\n</tool_response>\n"
        )
        r = terminal_exploration_depth_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(r, 0.15)

    def test_duplicate_command_penalty(self):
        # Duplicate identical commands in first 10 → -0.10
        transcript = (
            # same echo twice
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "echo a"}}\n' "</tool_call>\n"
            "user\n<tool_response>\na\n</tool_response>\n"
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "echo a"}}\n' "</tool_call>\n"
            "user\n<tool_response>\na\n</tool_response>\n"
        )
        r = terminal_exploration_depth_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(r, max(0.0, 0.0 - 0.10))

    def test_consecutive_failure_penalty(self):
        # Same command fails twice in a row → -0.10
        transcript = (
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "rg -n \'x\' file.py"}}\n' "</tool_call>\n"
            "user\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "rg -n \'x\' file.py"}}\n' "</tool_call>\n"
            "user\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
        )
        r = terminal_exploration_depth_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(r, max(0.0, 0.0 - 0.10))

    def test_truncated_output_penalty_without_precise(self):
        # Decompose: (A) scoped search gives +0.15; (B) truncated-only gives 0.0; (A+B) yields +0.05
        transcript_A = (
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "rg -l \'x\'"}}\n'
            "</tool_call>\nuser\n<tool_response>\nfile.py\n</tool_response>\n"
        )
        transcript_B = (
            "assistant\n<tool_call>\n"
            '{"name": "shell", "arguments": {"cmd": "echo big"}}\n'
            "</tool_call>\nuser\n<tool_response>\noutput truncated\n</tool_response>\n"
        )
        r_A = terminal_exploration_depth_reward_func(completion=[transcript_A])[0]
        r_B = terminal_exploration_depth_reward_func(completion=[transcript_B])[0]
        r_AB = terminal_exploration_depth_reward_func(completion=[transcript_A + transcript_B])[0]
        self.assertApproxEqual(r_A, 0.15)
        self.assertApproxEqual(r_B, 0.0)
        self.assertApproxEqual(r_AB, 0.05)  # 0.15 - 0.10 = 0.05

    def test_penalty_only_truncated_yields_zero(self):
        # Penalty-only case should clamp to 0.0 (no positive bonuses present)
        transcript = (
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "echo big"}}\n' "</tool_call>\n"
            "user\n<tool_response>\noutput truncated\n</tool_response>\n"
        )
        r = terminal_exploration_depth_reward_func(completion=[transcript])[0]
        self.assertApproxEqual(r, 0.0)

    def test_penalty_cap(self):
        # Multiple penalties should cap at 0.20
        transcript = (
            # duplicate cmd
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "echo a"}}\n' "</tool_call>\n"
            "user\n<tool_response>\na\n</tool_response>\n"
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "echo a"}}\n' "</tool_call>\n"
            "user\n<tool_response>\na\n</tool_response>\n"
            # consecutive failures of same
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "grep -n y missing.py"}}\n' "</tool_call>\n"
            "user\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "grep -n y missing.py"}}\n' "</tool_call>\n"
            "user\n<tool_response>\ncommand failed with exit code 1\n</tool_response>\n"
            # truncated output
            "assistant\n<tool_call>\n" '{"name": "shell", "arguments": {"cmd": "rg -l \'x\'"}}\n' "</tool_call>\n"
            "user\n<tool_response>\nfile.py\n<nano:feedback>output truncated</nano:feedback>\n</tool_response>\n"
        )
        r = terminal_exploration_depth_reward_func(completion=[transcript])[0]
        # penalties would sum to 0.30, capped at 0.20; no bonuses present
        self.assertApproxEqual(r, 0.0)


if __name__ == "__main__":
    unittest.main()


