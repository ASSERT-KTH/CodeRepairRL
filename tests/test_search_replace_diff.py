import sys
import unittest
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diff import SearchReplaceDiff


class TestSearchReplaceDiff(unittest.TestCase):
    """Test cases for the refactored SearchReplaceDiff in src/utils/diff.py."""

    def test_basic_from_string(self):
        """Test basic parsing of a search/replace block."""
        diff_text = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        self.assertEqual(len(diff.blocks), 1)
        self.assertEqual(diff.blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diff.blocks[0][1], "def hello():\n    print('hello world')")

    def test_from_string_with_malformed_markers(self):
        """Test parsing with slightly malformed markers."""
        diff_text = (
            "<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>> REPLACE"
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        self.assertEqual(len(diff.blocks), 1)
        self.assertEqual(diff.blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diff.blocks[0][1], "def hello():\n    print('hello world')")

    def test_from_string_with_multiple_blocks(self):
        """Test parsing multiple blocks."""
        diff_text = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        self.assertEqual(len(diff.blocks), 2)
        self.assertEqual(diff.blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diff.blocks[0][1], "def hello():\n    print('hello world')")
        self.assertEqual(diff.blocks[1][0], "def goodbye():\n    print('goodbye')")
        self.assertEqual(diff.blocks[1][1], "def goodbye():\n    print('goodbye world')")

    def test_from_string_empty_diff(self):
        """Test parsing an empty diff."""
        diff = SearchReplaceDiff.from_string("")
        self.assertEqual(len(diff.blocks), 0)

    def test_from_string_invalid_diff(self):
        """Test parsing an invalid diff."""
        diff = SearchReplaceDiff.from_string("This is not a valid diff")
        self.assertEqual(len(diff.blocks), 0)

    def test_from_string_with_invalid_formats(self):
        """Test parsing various invalid formats."""
        # Missing search marker
        missing_search = (
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        # Missing divider
        missing_divider = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            ">>>>>>> REPLACE"
        )
        
        # Missing replace marker
        missing_replace = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')"
        )
        
        # Wrong order of markers
        wrong_order = (
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            ">>>>>>> REPLACE"
        )
        
        # All should result in empty diffs
        self.assertEqual(len(SearchReplaceDiff.from_string(missing_search).blocks), 0)
        self.assertEqual(len(SearchReplaceDiff.from_string(missing_divider).blocks), 0)
        self.assertEqual(len(SearchReplaceDiff.from_string(missing_replace).blocks), 0)
        self.assertEqual(len(SearchReplaceDiff.from_string(wrong_order).blocks), 0)

    def test_extract_from_llm_response(self):
        """Test extracting diffs from an LLM response."""
        llm_response = (
            "Here's the fix:\n"
            "\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "\n"
            "And another change:\n"
            "\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE\n"
            "```"
        )
        
        diffs = SearchReplaceDiff.extract_from_llm_response(llm_response)
        self.assertEqual(len(diffs), 2)
        
        # Check first diff
        self.assertEqual(len(diffs[0].blocks), 1)
        self.assertEqual(diffs[0].blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diffs[0].blocks[0][1], "def hello():\n    print('hello world')")
        
        # Check second diff
        self.assertEqual(len(diffs[1].blocks), 1)
        self.assertEqual(diffs[1].blocks[0][0], "def goodbye():\n    print('goodbye')")
        self.assertEqual(diffs[1].blocks[0][1], "def goodbye():\n    print('goodbye world')")

    def test_extract_from_complex_llm_response(self):
        """Test extracting diffs from a more complex LLM response with think/answer tags."""
        llm_response = (
            "<think>\n"
            "The main issue is that the function doesn't handle negative inputs correctly.\n"
            "Also, there's a potential division by zero error.\n"
            "</think>\n"
            "<answer>\n"
            "Here are the fixes:\n"
            "\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "def calculate(x, y):\n"
            "    return x / y\n"
            "=======\n"
            "def calculate(x, y):\n"
            "    if y == 0:\n"
            "        raise ZeroDivisionError(\"Cannot divide by zero\")\n"
            "    return x / y\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "\n"
            "And another issue:\n"
            "\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "    return value + 10\n"
            "=======\n"
            "    if value < 0:\n"
            "        value = 0\n"
            "    return value + 10\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "</answer>"
        )
        
        diffs = SearchReplaceDiff.extract_from_llm_response(llm_response)
        self.assertEqual(len(diffs), 2)
        
        # Check first diff
        self.assertEqual(len(diffs[0].blocks), 1)
        self.assertEqual(diffs[0].blocks[0][0], "def calculate(x, y):\n    return x / y")
        self.assertEqual(diffs[0].blocks[0][1], "def calculate(x, y):\n    if y == 0:\n        raise ZeroDivisionError(\"Cannot divide by zero\")\n    return x / y")
        
        # Check second diff
        self.assertEqual(len(diffs[1].blocks), 1)
        self.assertEqual(diffs[1].blocks[0][0], "    return value + 10")
        self.assertEqual(diffs[1].blocks[0][1], "    if value < 0:\n        value = 0\n    return value + 10")

    def test_extract_from_llm_response_without_code_fences(self):
        """Test extracting diffs from an LLM response without code fences."""
        llm_response = (
            "Here's the fix:\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
        )
        
        diffs = SearchReplaceDiff.extract_from_llm_response(llm_response)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(len(diffs[0].blocks), 1)
        self.assertEqual(diffs[0].blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diffs[0].blocks[0][1], "def hello():\n    print('hello world')")

    def test_from_codes(self):
        """Test generating a diff from before/after code."""
        before_code = "def hello():\n    print('hello')"
        after_code = "def hello():\n    print('hello world')"
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        self.assertEqual(len(diff.blocks), 1)
        self.assertEqual(diff.blocks[0][0], before_code)
        self.assertEqual(diff.blocks[0][1], after_code)

    def test_from_codes_identical(self):
        """Test generating a diff from identical before/after code."""
        code = "def hello():\n    print('hello')"
        
        diff = SearchReplaceDiff.from_codes(code, code)
        self.assertEqual(len(diff.blocks), 0)

    def test_apply_diff(self):
        """Test applying a diff to code."""
        code = "def hello():\n    print('hello')\n\ndef goodbye():\n    print('goodbye')"
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        expected = "def hello():\n    print('hello world')\n\ndef goodbye():\n    print('goodbye world')"
        self.assertEqual(result, expected)

    def test_apply_diff_empty_diff(self):
        """Test applying an empty diff."""
        code = "def hello():\n    print('hello')"
        diff = SearchReplaceDiff([])
        
        result = diff.apply_diff(code)
        self.assertEqual(result, code)

    def test_apply_diff_to_empty_code(self):
        """Test applying a diff to empty code."""
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff("")
        self.assertEqual(result, "def hello():\n    print('hello world')")

    def test_validate_quality(self):
        """Test quality validation."""
        # Perfect diff
        perfect_diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        self.assertEqual(perfect_diff.validate_quality(), 1.0)
        
        # Invalid diff
        invalid_diff = SearchReplaceDiff([("", "")])
        self.assertEqual(invalid_diff.validate_quality(), 0.0)
        
        # Empty diff
        empty_diff = SearchReplaceDiff([])
        self.assertEqual(empty_diff.validate_quality(), 0.0)
        
        # Identical search and replace
        identical_diff = SearchReplaceDiff([("def hello():", "def hello():")])
        self.assertLess(identical_diff.validate_quality(), 1.0)

    def test_is_valid_format(self):
        """Test format validation."""
        # Perfect diff
        perfect_diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        self.assertTrue(perfect_diff.is_valid_format())
        
        # Invalid diff
        invalid_diff = SearchReplaceDiff([("", "")])
        self.assertFalse(invalid_diff.is_valid_format())
        
        # Empty diff
        empty_diff = SearchReplaceDiff([])
        self.assertFalse(empty_diff.is_valid_format())
        
        # Non-strict validation
        self.assertTrue(perfect_diff.is_valid_format(strict=False))
        self.assertFalse(invalid_diff.is_valid_format(strict=False))

    def test_to_string(self):
        """Test converting a diff to a string."""
        before_code = "def hello():\n    print('hello')"
        after_code = "def hello():\n    print('hello world')"
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        expected = (
            "<<<<<<< SEARCH\n"
            f"{before_code}\n"
            "=======\n"
            f"{after_code}\n"
            ">>>>>>> REPLACE"
        )
        self.assertEqual(diff.to_string(), expected)
        
        # Multiple blocks
        diff = SearchReplaceDiff([
            (before_code, after_code),
            ("def goodbye():", "def goodbye(name):")
        ])
        expected = (
            "<<<<<<< SEARCH\n"
            f"{before_code}\n"
            "=======\n"
            f"{after_code}\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "=======\n"
            "def goodbye(name):\n"
            ">>>>>>> REPLACE"
        )
        self.assertEqual(diff.to_string(), expected)
        
        # Empty diff
        diff = SearchReplaceDiff([])
        self.assertEqual(diff.to_string(), "")

    def test_from_codes_with_context(self):
        """Test generating a diff from before/after code with context."""
        before_code = """def calculate(x, y):
    # Add two numbers
    result = x + y
    return result

def multiply(x, y):
    # Multiply two numbers
    return x * y
"""
        after_code = """def calculate(x, y):
    # Add two numbers and multiply by 2
    result = (x + y) * 2
    return result

def multiply(x, y):
    # Multiply two numbers
    return x * y
"""
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        
        # Should only have one block for the changed function
        self.assertEqual(len(diff.blocks), 1)
        
        # The block should include the changed function with context
        self.assertIn("def calculate", diff.blocks[0][0])
        self.assertIn("# Add two numbers", diff.blocks[0][0])
        self.assertIn("result = x + y", diff.blocks[0][0])
        
        self.assertIn("def calculate", diff.blocks[0][1])
        self.assertIn("# Add two numbers and multiply by 2", diff.blocks[0][1])
        self.assertIn("result = (x + y) * 2", diff.blocks[0][1])
        
        # The unchanged multiply function should not be included
        self.assertNotIn("def multiply", diff.blocks[0][0])
        self.assertNotIn("def multiply", diff.blocks[0][1])

    def test_from_codes_multiple_changes(self):
        """Test generating a diff with multiple separate changes."""
        before_code = """def function1():
    print("Original function 1")

def function2():
    print("Original function 2")

def function3():
    print("Original function 3")
"""
        after_code = """def function1():
    print("Modified function 1")

def function2():
    print("Original function 2")

def function3():
    print("Modified function 3")
"""
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        
        # Should have two blocks, one for each changed function
        self.assertEqual(len(diff.blocks), 2)
        
        # First block should be for function1
        self.assertIn("function1", diff.blocks[0][0])
        self.assertIn("Original function 1", diff.blocks[0][0])
        self.assertIn("function1", diff.blocks[0][1])
        self.assertIn("Modified function 1", diff.blocks[0][1])
        
        # Second block should be for function3
        self.assertIn("function3", diff.blocks[1][0])
        self.assertIn("Original function 3", diff.blocks[1][0])
        self.assertIn("function3", diff.blocks[1][1])
        self.assertIn("Modified function 3", diff.blocks[1][1])
        
        # The unchanged function2 should not be in any block
        for block in diff.blocks:
            self.assertNotIn("Original function 2", block[0])
            self.assertNotIn("Original function 2", block[1])

    def test_from_codes_with_additions(self):
        """Test generating a diff with added lines."""
        before_code = """def process_data(data):
    # Process the input data
    result = data * 2
    return result
"""
        after_code = """def process_data(data):
    # Process the input data
    if data < 0:
        data = 0
    result = data * 2
    return result
"""
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        
        # Should have one block
        self.assertEqual(len(diff.blocks), 1)
        
        # The block should include context and the added lines
        self.assertIn("def process_data", diff.blocks[0][0])
        self.assertIn("# Process the input data", diff.blocks[0][0])
        self.assertNotIn("if data < 0:", diff.blocks[0][0])
        
        self.assertIn("def process_data", diff.blocks[0][1])
        self.assertIn("# Process the input data", diff.blocks[0][1])
        self.assertIn("if data < 0:", diff.blocks[0][1])
        self.assertIn("    data = 0", diff.blocks[0][1])

    def test_from_codes_with_deletions(self):
        """Test generating a diff with deleted lines."""
        before_code = """def process_data(data):
    # Process the input data
    if data < 0:
        data = 0
    result = data * 2
    return result
"""
        after_code = """def process_data(data):
    # Process the input data
    result = data * 2
    return result
"""
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        
        # Should have one block
        self.assertEqual(len(diff.blocks), 1)
        
        # The block should include context and show the deleted lines
        self.assertIn("def process_data", diff.blocks[0][0])
        self.assertIn("# Process the input data", diff.blocks[0][0])
        self.assertIn("if data < 0:", diff.blocks[0][0])
        self.assertIn("    data = 0", diff.blocks[0][0])
        
        self.assertIn("def process_data", diff.blocks[0][1])
        self.assertIn("# Process the input data", diff.blocks[0][1])
        self.assertNotIn("if data < 0:", diff.blocks[0][1])
        self.assertNotIn("    data = 0", diff.blocks[0][1])

    def test_from_codes_with_whitespace_changes(self):
        """Test generating a diff with only whitespace changes."""
        before_code = """def example():
    x = 1
    y = 2
    return x + y"""
        after_code = """def example():
    x = 1
    
    y = 2
    return x + y"""
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        
        # Should have one block
        self.assertEqual(len(diff.blocks), 1)
        
        # The block should show the whitespace difference
        self.assertIn("def example", diff.blocks[0][0])
        self.assertIn("    x = 1", diff.blocks[0][0])
        self.assertIn("    y = 2", diff.blocks[0][0])
        
        self.assertIn("def example", diff.blocks[0][1])
        self.assertIn("    x = 1", diff.blocks[0][1])
        self.assertIn("    \n    y = 2", diff.blocks[0][1])

    def test_from_codes_with_whitespace_changes(self):
        """Test generating a diff from code with whitespace changes."""
        before_code = (
            "def calculate(x, y):\n"
            "    result = x + y\n"
            "    return result"
        )
        
        after_code = (
            "def calculate(x, y):\n"
            "    result = x + y\n"
            "    return result  "  # Added two spaces at the end
        )
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        
        # The diff should have one block
        self.assertEqual(len(diff.blocks), 1)
        
        # The search content should be the original line
        self.assertEqual(diff.blocks[0][0], "    return result")
        
        # The replace content should be the line with added whitespace
        self.assertEqual(diff.blocks[0][1], "    return result  ")
        
        # Applying the diff should produce the after_code
        result = diff.apply_diff(before_code)
        self.assertEqual(result, after_code)

    def test_parse_search_replace_block(self):
        """Test parsing a single search/replace block."""
        block = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        diff = SearchReplaceDiff.from_string(block)
        
        # The diff should have one block
        self.assertEqual(len(diff.blocks), 1)
        
        # The search content should be the original code
        self.assertEqual(diff.blocks[0][0], "def hello():\n    print('hello')")
        
        # The replace content should be the modified code
        self.assertEqual(diff.blocks[0][1], "def hello():\n    print('hello world')")

    def test_parse_search_replace_block_missing_divider(self):
        """Test parsing a search/replace block with a missing divider."""
        broken_block = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            # missing divider
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
            "def hello():\n"
        )

        diff = SearchReplaceDiff.from_string(broken_block)
        
        # The diff should have one block, even with the missing divider
        self.assertEqual(len(diff.blocks), 1)
        
        # The search and replace content should be extracted as best as possible
        self.assertTrue(len(diff.blocks[0][0]) > 0)
        self.assertTrue(len(diff.blocks[0][1]) > 0)

    def test_simple_replace(self):
        """Test a simple replacement."""
        code = "def hello():\n    print('hello')\n    return None"
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    print('hello')\n"
            "=======\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = "def hello():\n    print('hello world')\n    return None"
        self.assertEqual(result, expected)

    def test_multi_line_replace(self):
        """Test a multi-line replacement."""
        code = (
            "def calculate(x, y):\n"
            "    # Add two numbers\n"
            "    result = x + y\n"
            "    return result"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    # Add two numbers\n"
            "    result = x + y\n"
            "=======\n"
            "    # Multiply two numbers\n"
            "    result = x * y\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def calculate(x, y):\n"
            "    # Multiply two numbers\n"
            "    result = x * y\n"
            "    return result"
        )
        self.assertEqual(result, expected)

    def test_simple_deletion(self):
        """Test a simple deletion."""
        code = (
            "def calculate(x, y):\n"
            "    # Add two numbers\n"
            "    result = x + y\n"
            "    return result"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    # Add two numbers\n"
            "=======\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def calculate(x, y):\n"
            "    result = x + y\n"
            "    return result"
        )
        self.assertEqual(result, expected)

    def test_multi_line_deletion(self):
        """Test a multi-line deletion."""
        code = (
            "def calculate(x, y):\n"
            "    # Add two numbers\n"
            "    # This is a comment\n"
            "    result = x + y\n"
            "    return result"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    # Add two numbers\n"
            "    # This is a comment\n"
            "=======\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def calculate(x, y):\n"
            "    result = x + y\n"
            "    return result"
        )
        self.assertEqual(result, expected)

    def test_simple_insertion(self):
        """Test a simple insertion."""
        code = (
            "def calculate(x, y):\n"
            "    result = x + y\n"
            "    return result"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    result = x + y\n"
            "=======\n"
            "    # Add two numbers\n"
            "    result = x + y\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def calculate(x, y):\n"
            "    # Add two numbers\n"
            "    result = x + y\n"
            "    return result"
        )
        self.assertEqual(result, expected)

    def test_multi_line_insertion(self):
        """Test a multi-line insertion."""
        code = (
            "def calculate(x, y):\n"
            "    result = x + y\n"
            "    return result"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    result = x + y\n"
            "=======\n"
            "    # Add two numbers\n"
            "    # This is a comment\n"
            "    result = x + y\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def calculate(x, y):\n"
            "    # Add two numbers\n"
            "    # This is a comment\n"
            "    result = x + y\n"
            "    return result"
        )
        self.assertEqual(result, expected)

    def test_multiple_changes(self):
        """Test multiple changes in a single diff."""
        code = (
            "def calculate(x, y):\n"
            "    # Add two numbers\n"
            "    result = x + y\n"
            "    return result\n"
            "\n"
            "def greet(name):\n"
            "    print('Hello, ' + name)\n"
            "    return None"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    # Add two numbers\n"
            "    result = x + y\n"
            "=======\n"
            "    # Multiply two numbers\n"
            "    result = x * y\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "    print('Hello, ' + name)\n"
            "=======\n"
            "    print(f'Hello, {name}!')\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def calculate(x, y):\n"
            "    # Multiply two numbers\n"
            "    result = x * y\n"
            "    return result\n"
            "\n"
            "def greet(name):\n"
            "    print(f'Hello, {name}!')\n"
            "    return None"
        )
        self.assertEqual(result, expected)

    def test_indentation_preservation(self):
        """Test that indentation is preserved in replacements."""
        code = (
            "def outer():\n"
            "    def inner():\n"
            "        x = 10\n"
            "        return x\n"
            "    return inner()"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "        x = 10\n"
            "=======\n"
            "        x = 20\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def outer():\n"
            "    def inner():\n"
            "        x = 20\n"
            "        return x\n"
            "    return inner()"
        )
        self.assertEqual(result, expected)

    def test_empty_before_code(self):
        """Test handling of empty before_code."""
        before_code = ""
        after_code = "def hello():\n    print('hello')"
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        
        # The diff should have one block
        self.assertEqual(len(diff.blocks), 1)
        
        # The search content should be empty
        self.assertEqual(diff.blocks[0][0], "")
        
        # The replace content should be the entire after_code
        self.assertEqual(diff.blocks[0][1], after_code)
        
        # Applying the diff to empty code should produce the after_code
        result = diff.apply_diff(before_code)
        self.assertEqual(result, after_code)

    def test_empty_after_code(self):
        """Test handling of empty after_code."""
        before_code = "def hello():\n    print('hello')"
        after_code = ""
        
        diff = SearchReplaceDiff.from_codes(before_code, after_code)
        
        # The diff should have one block
        self.assertEqual(len(diff.blocks), 1)
        
        # The search content should be the entire before_code
        self.assertEqual(diff.blocks[0][0], before_code)
        
        # The replace content should be empty
        self.assertEqual(diff.blocks[0][1], "")
        
        # Applying the diff should produce empty code
        result = diff.apply_diff(before_code)
        self.assertEqual(result, after_code)

    def test_no_changes(self):
        """Test handling of identical before and after code."""
        code = "def hello():\n    print('hello')"
        
        diff = SearchReplaceDiff.from_codes(code, code)
        
        # The diff should have no blocks
        self.assertEqual(len(diff.blocks), 0)
        
        # Applying the diff should not change the code
        result = diff.apply_diff(code)
        self.assertEqual(result, code)

    def test_parse_malformed_markers(self):
        """Test parsing a diff with malformed markers."""
        diff_text = (
            "<<<<<< SEARCH\n"  # Missing one <
            "def hello():\n"
            "    print('hello')\n"
            "======\n"  # Missing one =
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>> REPLACE"  # Missing one >
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        
        # The diff should still be parsed
        self.assertEqual(len(diff.blocks), 1)
        
        # The search and replace content should be extracted
        self.assertEqual(diff.blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diff.blocks[0][1], "def hello():\n    print('hello world')")

    def test_parse_excessive_markers(self):
        """Test parsing a diff with excessive markers."""
        diff_text = (
            "<<<<<<<<<< SEARCH\n"  # Too many <
            "def hello():\n"
            "    print('hello')\n"
            "=========\n"  # Too many =
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>>>> REPLACE"  # Too many >
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        
        # The diff should still be parsed
        self.assertEqual(len(diff.blocks), 1)
        
        # The search and replace content should be extracted
        self.assertEqual(diff.blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diff.blocks[0][1], "def hello():\n    print('hello world')")

    def test_parse_whitespace_in_markers(self):
        """Test parsing a diff with whitespace in markers."""
        diff_text = (
            "<<<<<<< SEARCH \n"  # Extra space after SEARCH
            "def hello():\n"
            "    print('hello')\n"
            "======= \n"  # Extra space after =======
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE "  # Extra space after REPLACE
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        
        # The diff should still be parsed
        self.assertEqual(len(diff.blocks), 1)
        
        # The search and replace content should be extracted
        self.assertEqual(diff.blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diff.blocks[0][1], "def hello():\n    print('hello world')")

    def test_quality_validation_perfect(self):
        """Test quality validation on a perfect diff."""
        diff_text = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        quality = diff.validate_quality()
        
        # A perfect diff should have quality 1.0
        self.assertEqual(quality, 1.0)

    def test_quality_validation_good_enough(self):
        """Test quality validation on a good enough diff."""
        diff_text = (
            "<<<<<< SEARCH\n"  # Missing one <
            "def hello():\n"
            "    print('hello')\n"
            "======\n"  # Missing one =
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>> REPLACE"  # Missing one >
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        quality = diff.validate_quality()
        
        # A good enough diff should have quality >= 0.7
        self.assertGreaterEqual(quality, 0.7)

    def test_quality_validation_recoverable(self):
        """Test quality validation on a recoverable diff."""
        diff_text = (
            "SEARCH\n"  # Missing <<<<<<< 
            "def hello():\n"
            "    print('hello')\n"
            "DIVIDER\n"  # Using DIVIDER instead of =======
            "def hello():\n"
            "    print('hello world')\n"
            "REPLACE"  # Missing >>>>>>>
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        quality = diff.validate_quality()
        
        # A recoverable diff should have quality >= 0.4
        self.assertGreaterEqual(quality, 0.4)

    def test_quality_validation_poor(self):
        """Test quality validation on a poor diff."""
        diff_text = (
            "Here's a diff that changes 'hello' to 'hello world':\n"
            "def hello():\n"
            "    print('hello')\n"
            "should be changed to:\n"
            "def hello():\n"
            "    print('hello world')\n"
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        quality = diff.validate_quality()
        
        # A poor diff should have quality >= 0.1
        self.assertGreaterEqual(quality, 0.1)

    def test_safe_apply_perfect_diff(self):
        """Test safely applying a perfect diff."""
        code = "def hello():\n    print('hello')\n    return None"
        diff_text = (
            "<<<<<<< SEARCH\n"
            "    print('hello')\n"
            "=======\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        result, quality = diff.safe_apply_diff(code)
        
        # Check the quality - should be perfect
        self.assertEqual(quality, 1.0)
        
        # Check the result
        expected = "def hello():\n    print('hello world')\n    return None"
        self.assertEqual(result, expected)

    def test_comment_only_change(self):
        """Test a change that only affects comments."""
        code = (
            "def calculate(x, y):\n"
            "    # Add two numbers\n"
            "    result = x + y\n"
            "    return result"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    # Add two numbers\n"
            "=======\n"
            "    # Sum two numbers\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def calculate(x, y):\n"
            "    # Sum two numbers\n"
            "    result = x + y\n"
            "    return result"
        )
        self.assertEqual(result, expected)

    def test_apply_diff_with_surrounding_context(self):
        """Test applying a diff with surrounding context."""
        code = (
            "def calculate(x, y):\n"
            "    # This is a function to calculate the sum of two numbers\n"
            "    # It takes two parameters: x and y\n"
            "    result = x + y\n"
            "    # Return the result\n"
            "    return result"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    # This is a function to calculate the sum of two numbers\n"
            "    # It takes two parameters: x and y\n"
            "    result = x + y\n"
            "=======\n"
            "    # This is a function to calculate the product of two numbers\n"
            "    # It takes two parameters: x and y\n"
            "    result = x * y\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def calculate(x, y):\n"
            "    # This is a function to calculate the product of two numbers\n"
            "    # It takes two parameters: x and y\n"
            "    result = x * y\n"
            "    # Return the result\n"
            "    return result"
        )
        self.assertEqual(result, expected)

    def test_partial_match_with_specific_replacement(self):
        """Test a partial match with a specific replacement."""
        code = (
            "def calculate(x, y):\n"
            "    # Calculate the sum\n"
            "    result = x + y\n"
            "    return result\n"
            "\n"
            "def calculate_again(a, b):\n"
            "    # Calculate the sum again\n"
            "    result = a + b\n"
            "    return result"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    # Calculate the sum\n"
            "    result = x + y\n"
            "=======\n"
            "    # Calculate the product\n"
            "    result = x * y\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        # Only the first occurrence should be replaced
        expected = (
            "def calculate(x, y):\n"
            "    # Calculate the product\n"
            "    result = x * y\n"
            "    return result\n"
            "\n"
            "def calculate_again(a, b):\n"
            "    # Calculate the sum again\n"
            "    result = a + b\n"
            "    return result"
        )
        self.assertEqual(result, expected)

    def test_multiple_diffs_with_same_search(self):
        """Test applying multiple diffs with the same search content."""
        code = (
            "def calculate(x, y):\n"
            "    result = x + y\n"
            "    return result\n"
            "\n"
            "def calculate_again(a, b):\n"
            "    result = a + b\n"
            "    return result"
        )
        
        diff = SearchReplaceDiff.from_string(
            "<<<<<<< SEARCH\n"
            "    result = x + y\n"
            "=======\n"
            "    result = x * y\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "<<<<<<< SEARCH\n"
            "    result = a + b\n"
            "=======\n"
            "    result = a * b\n"
            ">>>>>>> REPLACE"
        )
        
        result = diff.apply_diff(code)
        
        expected = (
            "def calculate(x, y):\n"
            "    result = x * y\n"
            "    return result\n"
            "\n"
            "def calculate_again(a, b):\n"
            "    result = a * b\n"
            "    return result"
        )
        self.assertEqual(result, expected)

    def test_alternative_block_separators_triple_newline(self):
        """Test parsing a diff with alternative block separators (triple newline)."""
        diff_text = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE\n"
            "\n\n\n"  # Triple newline as block separator
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        
        # The diff should have two blocks
        self.assertEqual(len(diff.blocks), 2)
        
        # Check the content of both blocks
        self.assertEqual(diff.blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diff.blocks[0][1], "def hello():\n    print('hello world')")
        self.assertEqual(diff.blocks[1][0], "def goodbye():\n    print('goodbye')")
        self.assertEqual(diff.blocks[1][1], "def goodbye():\n    print('goodbye world')")

    def test_no_block_separators(self):
        """Test parsing a diff with no block separators."""
        diff_text = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    print('hello')\n"
            "=======\n"
            "def hello():\n"
            "    print('hello world')\n"
            ">>>>>>> REPLACE"
            # No separator
            "<<<<<<< SEARCH\n"
            "def goodbye():\n"
            "    print('goodbye')\n"
            "=======\n"
            "def goodbye():\n"
            "    print('goodbye world')\n"
            ">>>>>>> REPLACE"
        )
        
        diff = SearchReplaceDiff.from_string(diff_text)
        
        # The diff should have at least one block
        self.assertGreaterEqual(len(diff.blocks), 1)
        
        # The first block should be correctly parsed
        self.assertEqual(diff.blocks[0][0], "def hello():\n    print('hello')")
        self.assertEqual(diff.blocks[0][1], "def hello():\n    print('hello world')")


if __name__ == '__main__':
    unittest.main() 