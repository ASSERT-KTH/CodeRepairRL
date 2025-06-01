import time
import logging
from typing import Any, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests
from nano import Agent

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


@dataclass
class NanoConfig:
    model: Optional[str] = None
    api_base: str = "http://localhost:8000/v1"
    thinking: bool = False
    token_limit: int = 8192
    tool_limit: int = 20
    temperature: float = 0.7
    top_p: float = 0.8
    min_p: float = 0.01
    top_k: int = 20
    verbose: bool = False
    log: bool = False


def _process_one(data: dict[str, Any], config: NanoConfig) -> dict[str, Any]:
    assert "repo" in data and "base_commit" in data and "problem_statement" in data

    logger.info(f"[START] {data['repo']} @ {data['base_commit'][:7]}")

    agent = Agent(**dict(config))

    diff = ""
    temp_folder = None
    try:
        temp_folder = clone_repo_at_commit(handle_to_url(data["repo"]), data["base_commit"])
        diff = agent.run(task=data["problem_statement"], repo_root=temp_folder)
    except Exception as e:
        logger.error(f"Error in _process_one: {type(e).__name__}: {e}")
        diff = ""
    finally:
        if temp_folder: clean_repo_dir(temp_folder)

        token_usage = agent.token_usage
        tool_usage = agent.tool_usage
        diff_success = diff != ""
        logger.info(f"[FINISH] {data['repo']} @ {data['base_commit'][:7]} - Tokens: {token_usage}, Tools: {tool_usage}, Diff Success: {diff_success}")

    return dict(
        prompt=agent.messages[:2],
        completion=agent.messages[2:],
        tools=agent.tools,
        generated_diff=diff,
    )


def nano_rollout_func(data: list[dict[str, Any]], config: NanoConfig, timeout: int = 120) -> list[dict[str, Any]]:
    """Deploys parallel Nano agents talking to our trl vllm-serve-async endpoint to process the given data"""

    results = []
    ok, tout, err = 0, 0, 0

    logger.info(f"Starting {len(data)} agent rollouts")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=min(len(data), mp.cpu_count())) as executor:
        futures = [executor.submit(_process_one, datum, config) for datum in data]

        for fut in as_completed(futures):
            try:
                results.append(fut.result(timeout=timeout))
                ok += 1
            except TimeoutError:
                logger.warning(f"Rollout timed out after {timeout}s")
                results.append(dict(prompt=[], completion=[], tools=[], generated_diff=""))
                tout += 1
            except Exception as e:
                logger.error(f"Rollout error: {type(e).__name__}: {e}")
                results.append(dict(prompt=[], completion=[], tools=[], generated_diff=""))
                err += 1

    logger.info(f"Finished rollouts {len(data)} in {time.time() - start_time:.2f}s")
    logger.info(f"Success: {ok}, Timeout: {tout}, Error: {err}")
    return results


if __name__ == "__main__":
    import time
    from matplotlib import pyplot as plt

    import litellm
    litellm._turn_on_debug()

    from src.data.swe_gym import get_swe_gym_repo_repair_dataset

    # Test different batch sizes for parallel timing
    batch_sizes = [1, 2, 4, 8, 16]
    runs = 2
    data = get_swe_gym_repo_repair_dataset().shuffle(seed=42)

    avg_times = []

    for size in batch_sizes:
        print(f"Testing batch size {size}")
        subset = data.select(range(size))
        subset_dicts = [dict(x) for x in subset]
        times = []
        for i in range(runs):
            start_time = time.time()
            results = nano_rollout_func(subset_dicts, timeout=120)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")
        avg_time = sum(times) / runs
        avg_times.append(avg_time)
        print(f"Average time for batch size {size}: {avg_time:.2f}s\n")

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, avg_times, 'bo-', linewidth=2)
    plt.xlabel('Batch Size (Parallel Samples)')
    plt.ylabel('Average Time (seconds)')
    plt.title('Nano Agent Parallel Batch Timing')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('nano_agent_parallel_time.png', dpi=300, bbox_inches='tight')
    plt.close()