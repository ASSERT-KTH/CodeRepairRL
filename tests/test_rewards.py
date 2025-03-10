import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.rewards import (
    extract_xml_answer,
    correctness_reward_func,
    strict_format_reward_func,
    count_xml,
    xmlcount_reward_func
)


class TestClassificationRewards(unittest.TestCase):
    """Test cases for non-diff related reward functions in src/utils/rewards.py."""

    def test_extract_xml_answer(self):
        """Test extracting answers from XML tags."""
        # Valid XML answer
        text_with_answer = "Some text <answer>CWE-79</answer> more text"
        self.assertEqual(extract_xml_answer(text_with_answer), "CWE-79")
        
        # No XML answer
        text_without_answer = "Some text without answer tags"
        self.assertEqual(extract_xml_answer(text_without_answer), "N/A")
        
        # Multiple answers (should extract the first one)
        text_with_multiple = "<answer>First</answer> text <answer>Second</answer>"
        self.assertEqual(extract_xml_answer(text_with_multiple), "First")
        
        # Nested tags (should extract the outer one)
        nested_tags = "Text <answer>Outer <inner>Inner</inner> tag</answer>"
        self.assertEqual(extract_xml_answer(nested_tags), "Outer <inner>Inner</inner> tag")

    def test_correctness_reward_func(self):
        """Test the correctness reward function."""
        # Setup test data
        prompts = ["prompt1", "prompt2"]
        completions = [
            [{"content": "<answer>CWE-79</answer>"}],
            [{"content": "<answer>CWE-89: SQL Injection</answer>"}]
        ]
        answers = ["CWE-79", "CWE-89"]
        
        # Test exact matches
        rewards = correctness_reward_func(prompts, completions, answers)
        self.assertEqual(rewards, [2.0, 0.5])
        
        # Test partial matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-89</answer>"}]
        ]
        rewards = correctness_reward_func(prompts, completions, answers)
        self.assertEqual(rewards, [0.0, 2.0])
        
        # Test no matches
        completions = [
            [{"content": "<answer>CWE-80</answer>"}],
            [{"content": "<answer>CWE-90</answer>"}]
        ]
        rewards = correctness_reward_func(prompts, completions, answers)
        self.assertEqual(rewards, [0.0, 0.0])

    def test_strict_format_reward_func(self):
        """Test the strict format reward function."""
        # Valid format
        valid_completion = [
            {"content": "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"}
        ]
        rewards = strict_format_reward_func([valid_completion])
        self.assertEqual(rewards, [0.5])
        
        # Invalid format - missing think tag
        invalid_completion1 = [
            {"content": "Thinking process\n<answer>\nCWE-79\n</answer>"}
        ]
        rewards = strict_format_reward_func([invalid_completion1])
        self.assertEqual(rewards, [0.0])
        
        # Invalid format - missing answer tag
        invalid_completion2 = [
            {"content": "<think>\nThinking process\n</think>\nCWE-79"}
        ]
        rewards = strict_format_reward_func([invalid_completion2])
        self.assertEqual(rewards, [0.0])
        
        # Multiple completions
        completions = [
            valid_completion,
            invalid_completion1,
            invalid_completion2
        ]
        rewards = strict_format_reward_func(completions)
        self.assertEqual(rewards, [0.5, 0.0, 0.0])

    def test_count_xml(self):
        """Test the count_xml function."""
        # Perfect format
        perfect_format = (
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertEqual(count_xml(perfect_format), 0.5)
        
        # Missing think opening tag
        missing_think_open = (
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertAlmostEqual(count_xml(missing_think_open), 0.375, places=2)
        
        # Missing think closing tag
        missing_think_close = (
            "<think>\n"
            "Thinking process\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertAlmostEqual(count_xml(missing_think_close), 0.375, places=2)
        
        # Missing answer opening tag
        missing_answer_open = (
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertAlmostEqual(count_xml(missing_answer_open), 0.375, places=2)
        
        # Missing answer closing tag
        missing_answer_close = (
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79"
        )
        self.assertAlmostEqual(count_xml(missing_answer_close), 0.375, places=2)
        
        # No tags
        no_tags = "Just some text without any tags"
        self.assertEqual(count_xml(no_tags), 0.0)
        
        # Text before think tag (slight penalty)
        text_before_think = (
            "Some text before\n"
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>"
        )
        self.assertLess(count_xml(text_before_think), 0.5)
        
        # Text after answer tag (slight penalty)
        text_after_answer = (
            "<think>\n"
            "Thinking process\n"
            "</think>\n"
            "<answer>\n"
            "CWE-79\n"
            "</answer>\n"
            "Some text after"
        )
        self.assertLess(count_xml(text_after_answer), 0.5)

    def test_xmlcount_reward_func(self):
        """Test the xmlcount_reward_func function."""
        completions = [
            [{"content": "<think>\nThinking process\n</think>\n<answer>\nCWE-79\n</answer>"}],
            [{"content": "Just some text without any tags"}],
            [{"content": "<think>\nThinking process\n</think>\nCWE-79"}]
        ]
        
        rewards = xmlcount_reward_func(completions)
        self.assertEqual(rewards[0], 0.5)
        self.assertEqual(rewards[1], 0.0)
        self.assertLess(rewards[2], 0.5)


if __name__ == "__main__":
    unittest.main()