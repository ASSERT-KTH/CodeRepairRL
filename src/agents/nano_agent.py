import os
import time
import logging
from typing import Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

from nano import Agent

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


@dataclass
class NanoConfig:
    agent_kind: str = "nano"
    model: Optional[str] = None
    api_base: str = "http://localhost:8000/v1"
    thinking: bool = False
    token_limit: int = 8192
    tool_limit: int = 30
    time_limit: int = 60
    temperature: float = 0.7
    top_p: float = 0.95
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    verbose: bool = False
    log: bool = False
    backend: str = "local"
    env: Optional[Any] = None


def _process_one(data: dict[str, Any], config: NanoConfig) -> dict[str, Any]:
    assert "repo" in data and "base_commit" in data and "problem_statement" in data

    logger.info(f"[START] {data['repo']} @ {data['base_commit'][:7]}")
    start_time = time.time()

    agent_kwargs = asdict(config)
    agent_kwargs.pop("agent_kind", None)
    
    # Handle backend and environment setup
    backend = agent_kwargs.pop("backend", "local")
    env = agent_kwargs.pop("env", None)

    # Initialize agent with appropriate environment
    if backend == "apptainer" and env is None:
        # If using Apptainer backend but env not provided in config, we need to construct it
        # This logic might belong better in the caller, but we can handle it here or
        # expect the caller to pass the fully constructed environment in `config.env`.
        # Based on the reference, the caller (run_nano_eval) should likely construct the environment.
        # However, `_process_one` is called per instance, and the environment depends on the instance ID.
        
        from nano.env import ApptainerEnvironment
        instance_id = data.get("instance_id", "")
        if instance_id:
            #  image_name = f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"
             image_name = f"docker.io/swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"
             workdir = "/testbed"
             env = ApptainerEnvironment(image=f"docker://{image_name}", workdir=workdir)
             agent_kwargs["env"] = env
    elif env:
        agent_kwargs["env"] = env

    agent = Agent(**agent_kwargs)

    diff = ""
    temp_folder = None
    
    if backend == "local":
        try:
            repo_url = handle_to_url(data["repo"])
            temp_folder = clone_repo_at_commit(repo_url, data["base_commit"])
        except Exception as e:
            agent._reset()
            agent._append({"role": "user", "content": data["problem_statement"]})
            agent._append({"role": "assistant", "content": ""})
            logger.error(f"Error with git in _process_one: {type(e).__name__}: {e}")
            if temp_folder: clean_repo_dir(temp_folder)
            return dict(
                prompt=agent.messages[:2],
                completion=agent.messages[2:],
                tools=agent.tools,
                generated_diff="",
                token_usage=agent.token_usage,
                tool_usage=agent.tool_usage,
                **agent.tool_stats
            )
    else:
        # For container backends, the repo is already in the container image at workdir
        temp_folder = "/testbed" # default workdir for SWE-bench containers

    try:
        diff = agent.run(task=data["problem_statement"], repo_root=temp_folder)
    except Exception as e:
        logger.error(f"Error in _process_one: {type(e).__name__}: {e}")
        diff = ""
    finally:
        if backend == "local" and temp_folder: clean_repo_dir(temp_folder)

        token_usage = agent.token_usage
        tool_usage = agent.tool_usage
        diff_success = diff != ""
        logger.info(f"[FINISH] {data['repo']} @ {data['base_commit'][:7]} - Tokens: {token_usage}, Tools: {tool_usage}, Diff Success: {diff_success}, Time: {time.time() - start_time:.2f}s")

    # Ensure agent.messages has enough messages to avoid empty completion
    if len(agent.messages) < 3:
        logger.warning(f"Agent messages incomplete ({len(agent.messages)} messages), padding with fallback")
        # Pad with user message and empty assistant response
        if len(agent.messages) < 2:
            agent._append({"role": "user", "content": data["problem_statement"]})
        if len(agent.messages) < 3:
            agent._append({"role": "assistant", "content": ""})
    
    result = dict(
        prompt=agent.messages[:2],
        completion=agent.messages[2:],
        tools=agent.tools,
        generated_diff=diff,
        token_usage=agent.token_usage,
        tool_usage=agent.tool_usage,
        **agent.tool_stats
    )
    return result


def nano_rollout_func(data: list[dict[str, Any]], config: NanoConfig, **kwargs) -> list[dict[str, Any]]:
    """Deploys parallel Nano agents talking to our trl vllm-serve-async endpoint to process the given data"""

    logger.info(f"Starting {len(data)} agent rollouts")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=min(len(data), os.cpu_count())) as executor:
        results = list(executor.map(lambda datum: _process_one(datum, config), data))

    logger.info(f"Finished {len(data)} rollouts in {time.time() - start_time:.2f}s")
    return results


if __name__ == "__main__":
    import time

    from src.data.swe_gym import get_swe_gym_repo_repair_dataset

    # Test different batch sizes for parallel timing
    batch_sizes = [2]
    runs = 1
    data = get_swe_gym_repo_repair_dataset().shuffle(seed=42)

    config = NanoConfig(model="hosted_vllm/Qwen/Qwen3-8B")

    avg_times = []

    for size in batch_sizes:
        print(f"Testing batch size {size}")
        subset = data.select(range(size))
        subset_dicts = [dict(x) for x in subset]
        times = []
        for i in range(runs):
            start_time = time.time()
            results = nano_rollout_func(subset_dicts, config)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")
        avg_time = sum(times) / runs
        avg_times.append(avg_time)
        print(f"Average time for batch size {size}: {avg_time:.2f}s\n")
