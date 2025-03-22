"""
Task generation from The Stack dataset docstrings.

This module processes Python code from The Stack dataset to create coding tasks
based on docstrings. It extracts functions with good docstrings, masks their
implementations, and prepares them as tasks for LLM generation.
"""
import ast
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetInfo, Features, Value
from transformers import PreTrainedTokenizer


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Tracking statistics for quality filtering
quality_stats = {
    "total_files": 0,
    "skipped_sql": 0,
    "skipped_too_many_vars": 0,
    "skipped_too_simple": 0,
    "accepted_files": 0
}

@dataclass
class DocstringTask:
    """A simple task extracted from a docstring."""
    function_name: str
    docstring: str
    masked_code: str  # Code with entire function masked out
    implementation: str  # Original implementation
    start_line: int  # Start line of the function in the original file
    end_line: int  # End line of the function in the original file
    file_content: str  # The full original file content

def is_quality_docstring(docstring: str, node=None, code: str = None, min_length: int = 50) -> bool:
    """
    Determine if a docstring is of sufficient quality.
    
    Args:
        docstring: The docstring to evaluate
        node: The AST node of the function (optional)
        code: The function's code (optional)
        min_length: Minimum character length
        
    Returns:
        Boolean indicating if the docstring meets quality criteria
    """
    # Check minimum length
    if not docstring or len(docstring) < min_length:
        return False
    
    # Check if it has actual content (not just quotes or whitespace)
    cleaned = docstring.strip('"\' \t\n')
    if len(cleaned) < min_length * 0.8:  # 80% of min_length should be actual content
        return False
    
    # Check if it has some structure (paragraphs, sections, etc.)
    lines = [line.strip() for line in docstring.split('\n') if line.strip()]
    if len(lines) < 2:  # At least two non-empty lines
        return False
    
    # Filter out boilerplate docstrings
    boilerplate_indicators = [
        "# noqa", 
        ":rtype:", 
        "Gets the", 
        "Sets the"
    ]
    
    if any(indicator in docstring for indicator in boilerplate_indicators):
        return False
        
    # Check the node if provided - filter out property getters/setters
    if node and isinstance(node, ast.FunctionDef):
        # Check if it's a property getter/setter
        if any(decorator.id == 'property' for decorator in node.decorator_list 
               if isinstance(decorator, ast.Name)):
            return False
            
        # Check if it's a simple getter/setter (one return or assignment statement)
        if len(node.body) <= 2:  # Simple function with docstring and one line of code
            return False
    
    # Check for presence of descriptive language
    descriptive_patterns = [
        r'Args:', r'Arguments:', r'Parameters:', r'Params:',
        r'Returns:', r'Yields:', r'Raises:', r'Example', r'Examples:'
    ]
    has_descriptive_section = any(re.search(pattern, docstring, re.IGNORECASE) for pattern in descriptive_patterns)
    
    # Either has descriptive sections OR has at least 3 sentences
    sentences = re.split(r'[.!?]+', docstring)
    meaningful_sentences = [s for s in sentences if len(s.strip()) > 10]
    
    return has_descriptive_section or len(meaningful_sentences) >= 3

def is_quality_code(node: ast.AST, code: str, max_vars: int = 15) -> Tuple[bool, str]:
    """
    Analyze whether code meets quality standards.
    
    Args:
        node: AST node of the function
        code: Source code string
        max_vars: Maximum allowed number of variables
        
    Returns:
        Tuple of (is_quality, reason) where reason explains rejection if not quality
    """
    # Check for SQL strings
    sql_keywords = [
        "SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", 
        "HAVING", "UNION", "INSERT", "UPDATE", "DELETE", "CREATE TABLE"
    ]
    
    # Find all string literals in the function
    for n in ast.walk(node):
        if isinstance(n, ast.Str) or (isinstance(n, ast.Constant) and isinstance(n.value, str)):
            string_value = n.s if hasattr(n, 's') else n.value
            # Convert to uppercase for case-insensitive matching
            upper_str = string_value.upper()
            # Check if any SQL keyword is in the string
            if any(keyword in upper_str for keyword in sql_keywords):
                return False, "Contains SQL query"
    
    # Count variables (parameters, local vars, etc.)
    variables = set()
    
    # Add function parameters
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in node.args.args:
            if hasattr(arg, 'arg'):  # Python 3
                variables.add(arg.arg)
    
    # Add variables from assignments
    for n in ast.walk(node):
        # Variable assignment
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    variables.add(target.id)
                elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            variables.add(elt.id)
        
        # AnnAssign (annotated assignment) in Python 3
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            variables.add(n.target.id)
            
        # For loops add variables
        elif isinstance(n, ast.For) and isinstance(n.target, ast.Name):
            variables.add(n.target.id)
            
        # With statements can add variables
        elif isinstance(n, ast.With):
            for item in n.items:
                if hasattr(item, 'optional_vars') and isinstance(item.optional_vars, ast.Name):
                    if item.optional_vars is not None:
                        variables.add(item.optional_vars.id)
    
    if len(variables) > max_vars:
        return False, f"Too many variables: {len(variables)} > {max_vars}"
    
    # Check if the function is too simple
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Count non-docstring statements
        if len(node.body) < 4 and not any(isinstance(n, (ast.If, ast.For, ast.While)) for n in node.body):
            return False, "Too simple (fewer than 4 statements with no control flow)"
    
    return True, "Passed quality checks"

def extract_functions_with_docstrings(code: str, min_docstring_length: int = 50) -> List[Dict]:
    """
    Extract functions with good docstrings from Python code.
    
    Args:
        code: Python code to parse
        min_docstring_length: Minimum length for a docstring to be considered
        
    Returns:
        List of dictionaries containing function info
    """
    # Update statistics
    quality_stats["total_files"] += 1
    
    # Skip invalid code
    if not code or not isinstance(code, str):
        return []
        
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Skip files with syntax errors
        return []
    
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            
            # Skip functions without docstrings or with low quality docstrings
            if not is_quality_docstring(docstring, node=node, code=code, min_length=min_docstring_length):
                continue
            
            # Skip functions with low quality code
            is_quality, reason = is_quality_code(node, code)
            if not is_quality:
                # Update rejection statistics
                if "SQL" in reason:
                    quality_stats["skipped_sql"] += 1
                elif "variables" in reason:
                    quality_stats["skipped_too_many_vars"] += 1
                elif "simple" in reason:
                    quality_stats["skipped_too_simple"] += 1
                continue
            
            # Get the line range for the function
            # Check for decorators as they need to be included
            function_start_line = node.lineno
            decorator_start_line = min(decorator.lineno for decorator in node.decorator_list) if node.decorator_list else function_start_line
            start_line = decorator_start_line
            
            try:
                end_line = max(
                    child.end_lineno if hasattr(child, 'end_lineno') else node.lineno 
                    for child in ast.walk(node)
                )
            except ValueError:  # In case node has no children with end_lineno
                end_line = function_start_line
            
            # Get the function implementation as text
            code_lines = code.splitlines()
            function_impl = "\n".join(code_lines[start_line-1:end_line])
            
            # Create masked code (entire file with function removed)
            masked_lines = code_lines.copy()
            
            # Replace function with a placeholder comment
            comment = f"# MASKED: {node.name} function (lines {start_line}-{end_line})"
            masked_lines[start_line-1:end_line] = [comment]
            masked_code = "\n".join(masked_lines)
            
            # Create a function info dictionary
            functions.append({
                "name": node.name,
                "docstring": docstring,
                "implementation": function_impl,
                "masked_code": masked_code,
                "start_line": start_line,
                "end_line": end_line,
                "file_content": code
            })
    
    # Update acceptance statistic if we found any functions
    if functions:
        quality_stats["accepted_files"] += 1
    
    return functions

def create_docstring_tasks(max_samples: int = 100, min_docstring_length: int = 50) -> List[DocstringTask]:
    """
    Process samples from The Stack dataset to create docstring tasks.
    
    Args:
        max_samples: Maximum number of samples to process
        min_docstring_length: Minimum length for a docstring to be considered
        
    Returns:
        List of generated docstring tasks
    """
    logger.info(f"Loading up to {max_samples} samples from the smol-stack dataset...")
    
    # Load the small dataset
    dataset = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split=f"train[:{max_samples}]")
    
    tasks = []
    for sample in tqdm(dataset, desc="Processing samples"):
        # Extract functions with good docstrings
        function_infos = extract_functions_with_docstrings(
            sample["content"], 
            min_docstring_length=min_docstring_length
        )
        
        # Create tasks for each function with a good docstring
        for function_info in function_infos:
            task = DocstringTask(
                function_name=function_info["name"],
                docstring=function_info["docstring"],
                masked_code=function_info["masked_code"],
                implementation=function_info["implementation"],
                start_line=function_info["start_line"],
                end_line=function_info["end_line"],
                file_content=function_info["file_content"]
            )
            tasks.append(task)
    
    return tasks

def display_task_sample(task: DocstringTask):
    """Display a task in a readable format."""
    print("\n" + "="*80)
    print(f"Function: {task.function_name}")
    print(f"Location: Lines {task.start_line}-{task.end_line}")
    print("-"*80)
    print("Docstring:")
    print(task.docstring)
    print("-"*80)
    print("Masked Code (file with function removed):")
    print(task.masked_code)
    print("-"*80)
    print("Original Implementation:")
    print(task.implementation)
    print("="*80)

def display_task_stats(tasks: List[DocstringTask]):
    """Display statistics about the generated tasks."""
    if not tasks:
        print("No tasks generated!")
        return
        
    print(f"\nGenerated {len(tasks)} tasks")
    
    # Calculate statistics
    avg_docstring_length = sum(len(task.docstring) for task in tasks) / len(tasks)
    avg_implementation_length = sum(len(task.implementation) for task in tasks) / len(tasks)
    
    # Display statistics
    print(f"Average docstring length: {avg_docstring_length:.1f} characters")
    print(f"Average implementation length: {avg_implementation_length:.1f} characters")
    
    # Display most common function names
    function_names = {}
    for task in tasks:
        function_names[task.function_name] = function_names.get(task.function_name, 0) + 1
    
    top_functions = sorted(function_names.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nMost common function names:")
    for func, count in top_functions:
        print(f"  {func}: {count} occurrences")
    
    # Display quality filtering statistics
    print("\nQuality filtering statistics:")
    print(f"  Total files processed: {quality_stats['total_files']}")
    print(f"  Files with accepted functions: {quality_stats['accepted_files']} ({quality_stats['accepted_files']/quality_stats['total_files']*100:.1f}%)")
    print(f"  Skipped due to SQL: {quality_stats['skipped_sql']}")
    print(f"  Skipped due to too many variables: {quality_stats['skipped_too_many_vars']}")
    print(f"  Skipped due to being too simple: {quality_stats['skipped_too_simple']}")

def get_stack_repair_dataset(
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int = 512,
    max_samples: int = 100,
    system_prompt: Optional[str] = None,
    diff_type: str = "search_replace"
) -> Tuple[Dataset, int]:
    """
    Create a dataset for code repair tasks from The Stack dataset docstrings.
    
    Args:
        tokenizer: Tokenizer for tokenizing prompts
        max_prompt_length: Maximum prompt length for filtering (default: 512)
        max_samples: Maximum number of samples to process from the dataset
        system_prompt: Optional system prompt to use
        diff_type: Type of diff to use (search_replace or unified)
        
    Returns:
        Tuple of (processed dataset, maximum token length)
    """
    # stack.py thinks it is a module, and can therefore not import from src when run directly (creating the ds)
    from src.data.code_repair import create_repair_dataset
    
    # Load the cached dataset
    ds = load_dataset("ASSERT-KTH/stack-smol-docstrings")["train"].select(range(max_samples))
    
    logger.info(f"Successfully loaded {len(ds)} tasks from cached dataset")
    
    # Prepare data for repair dataset creation
    before_codes = []  # Masked code (with function removed)
    after_codes = []   # Original code (with function implementation)
    descriptions = []  # Function docstrings as descriptions
    
    for item in ds:
        # The masked code (with function replaced by comment) is the "before" code
        before_codes.append(item["masked_code"])
        
        # The original file is the "after" code, the diff now replacing the placeholder comment with the function implementation
        after_codes.append(item["file_content"])
        
        # Create a descriptive prompt that explains the task:
        descriptions.append(
            f"Task: Implement the function '{item['function_name']}' by replacing the comment line:\n"
            f"# MASKED: {item['function_name']} function (lines {item['start_line']}-{item['end_line']})\n\n"
            f"Your implementation should match this docstring:\n{item['docstring']}"
        )
    
    # Create the repair dataset using the generalized function
    return create_repair_dataset(
        before_codes=before_codes,
        after_codes=after_codes,
        descriptions=descriptions,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        system_prompt=system_prompt,
        diff_type=diff_type
    )


if __name__ == "__main__":
    import sys, os, argparse
    from datasets import Dataset
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.utils.llm import LLM  # we can not import from a module as a module
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process The Stack dataset to create docstring-based code tasks")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum number of samples to process")
    parser.add_argument("--min-docstring-length", type=int, default=50, help="Minimum docstring length")
    parser.add_argument("--push-to-hub", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--sample-tasks", type=int, default=3, help="Number of sample tasks to display")
    parser.add_argument("--llm-provider", type=str, default="anthropic", choices=["anthropic", "openrouter"], 
                        help="LLM API provider (anthropic or openrouter)")
    parser.add_argument("--llm-model", type=str, default="claude-3-5-haiku-latest", 
                        help="Model ID to use for instruction prompt generation")
    parser.add_argument("--llm-batch-size", type=int, default=10, help="Batch size for LLM API requests")
    args = parser.parse_args()
    
    # Set parameters
    MAX_SAMPLES = args.max_samples
    MIN_DOCSTRING_LENGTH = args.min_docstring_length
    
    # Process samples
    logger.info(f"Processing up to {MAX_SAMPLES} samples from the smol-stack dataset...")
    tasks = create_docstring_tasks(
        max_samples=MAX_SAMPLES,
        min_docstring_length=MIN_DOCSTRING_LENGTH
    )
    
    # Display statistics
    display_task_stats(tasks)
    
    # Display sample tasks if requested
    if tasks and args.sample_tasks > 0:
        print(f"\nSample tasks ({min(args.sample_tasks, len(tasks))} of {len(tasks)}):")
        for task in tasks[:args.sample_tasks]:
            display_task_sample(task)
    
    # Create HuggingFace Dataset
    if tasks:
        # Convert tasks to a list of dictionaries for the dataset
        task_dicts = []
        for task in tasks:
            task_dict = {
                "function_name": task.function_name,
                "docstring": task.docstring,
                "masked_code": task.masked_code,
                "implementation": task.implementation,
                "start_line": task.start_line,
                "end_line": task.end_line,
                "file_content": task.file_content,
                "instruction_prompt": ""  # Initialize with empty string
            }
            task_dicts.append(task_dict)
        
        # Create the dataset with proper DatasetInfo
        info = DatasetInfo(
            description="""This dataset contains Python functions extracted from [the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol), 
filtered for high-quality docstrings and implementations. Each sample includes 
the function's docstring, implementation, and a masked version of the code where the 
function is replaced with a comment.

The dataset is designed for code completion tasks where a model needs to restore a 
function that has been replaced with a comment. The model is provided with:
1. The full file context with the function replaced by a comment
2. The docstring of the function
3. The function name
4. An LLM-generated instruction prompt

The model's task is to generate code that replaces the comment with a proper implementation
of the function based on the docstring and surrounding context.

## Dataset Structure
Each sample contains:
- `function_name`: Name of the function
- `docstring`: The function's docstring
- `masked_code`: The full file with the function replaced by a comment
- `implementation`: The original function implementation
- `start_line`: The starting line number of the function in the original file
- `end_line`: The ending line number of the function in the original file
- `file_content`: The full original file content
- `instruction_prompt`: LLM-generated instruction prompt combining docstring and code context

## Quality Filtering
Functions are filtered based on:
- Docstring quality (length, structure, descriptiveness)
- Implementation quality (no SQL strings, reasonable number of variables, sufficient complexity)
        """,
            features=Features({
                "function_name": Value("string"),
                "docstring": Value("string"),
                "masked_code": Value("string"),
                "implementation": Value("string"),
                "start_line": Value("int32"),
                "end_line": Value("int32"),
                "file_content": Value("string"),
                "instruction_prompt": Value("string")
            })
        )

        # Create the dataset with the info attached
        dataset = Dataset.from_list(task_dicts, info=info)
        
        logger.info("Generating instruction prompts with LLM...")
        
        llm = LLM(
            provider=args.llm_provider,
            model=args.llm_model,
            batch_size=args.llm_batch_size
        )
        
        system_prompt = (
            "Create a clear, concise instruction prompt for implementing a function. "
            "Focus on combining information from both the docstring and surrounding code context."
        )
        
        # Generate prompts for each task
        prompts = [
            f"FUNCTION: {item['function_name']}\n\nDOCSTRING:\n{item['docstring']}\n\nCODE CONTEXT:\n{item['masked_code']}"
            for item in dataset
        ]
        
        # Generate instruction prompts
        instruction_prompts = llm.generate_completions(prompts, system_prompt)

        # Add instruction prompts to the dataset
        enhanced_data = []
        for i, item in enumerate(dataset):
            item_dict = dict(item)
            item_dict["instruction_prompt"] = instruction_prompts[i] if i < len(instruction_prompts) else ""
            enhanced_data.append(item_dict)
        
        # Create enhanced dataset
        dataset = Dataset.from_list(enhanced_data, info=info)
        logger.info(f"Successfully enhanced {len(dataset)} items with instruction prompts")
        
        # Push to hub if requested
        if args.push_to_hub:
            logger.info(f"Pushing dataset to HuggingFace Hub as ASSERT-KTH/stack-smol-docstrings-enhanced")
            dataset.push_to_hub("ASSERT-KTH/stack-smol-docstrings-enhanced")
            logger.info(f"Dataset pushed successfully")
            print(f"\nDataset URL: https://huggingface.co/datasets/ASSERT-KTH/stack-smol-docstrings-enhanced")



