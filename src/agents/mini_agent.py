import os
import time
import logging
import subprocess
from typing import Any, Optional
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.local import LocalEnvironment

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir
from src.agents.nano_agent import NanoConfig as AgentConfig

logger = logging.getLogger(__name__)


def git_diff(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "diff"],
        text=True, errors="ignore"
    )

def _process_one(data: dict[str, Any], config: AgentConfig) -> dict[str, Any]:
    assert "repo" in data and "base_commit" in data and "problem_statement" in data

    logger.info(f"[START] {data['repo']} @ {data['base_commit'][:7]}")
    t0 = time.time()

    # pre-build a fallback transcript in case git fails
    fallback_prompt = [{"role": "user", "content": data["problem_statement"]}]
    fallback_completion = [{"role": "assistant", "content": ""}]

    repo_dir: Optional[Path] = None
    try:
        repo_url = handle_to_url(data["repo"])
        repo_dir = Path(clone_repo_at_commit(repo_url, data["base_commit"]))
    except Exception as e:
        logger.error(f"Git setup error: {type(e).__name__}: {e}")
        if repo_dir:
            clean_repo_dir(str(repo_dir))
        return dict(
            prompt=fallback_prompt,
            completion=fallback_completion,
            tools=[],
            generated_diff=""
        )

    try:
        # mini’s OpenAI-compatible model via litellm
        model_kwargs = {
            "custom_llm_provider": "openai",
            "api_base": config.api_base,
            "api_key": "DUMMY",
            "temperature": config.temperature,
            "max_tokens": config.token_limit,
            "drop_params": True,          # ignore extras vLLM may not support
        }
        if config.top_p is not None:
            model_kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            model_kwargs["top_k"] = config.top_k

        # use mini's defaults
        model = LitellmModel(model_name=config.model, model_kwargs=model_kwargs)
        env = LocalEnvironment(cwd=str(repo_dir), timeout=config.time_limit)
        agent = DefaultAgent(model=model, env=env)
        status, final_msg = agent.run(task=data["problem_statement"])

        generated = git_diff(repo_dir)
        logger.info(
            f"[FINISH] {data['repo']} @ {data['base_commit'][:7]} "
            f"- Status: {status}, Diff: {bool(generated)}, Time: {time.time() - t0:.2f}s"
        )

        return dict(
            prompt=agent.messages[:2],        # system + user
            completion=agent.messages[2:],    # full linear history thereafter
            tools=[],                         # mini doesn't use OpenAI tools
            generated_diff=generated,
        )

    except Exception as e:
        logger.error(f"Mini run error: {type(e).__name__}: {e}")
        return dict(
            prompt=fallback_prompt,
            completion=fallback_completion,
            tools=[],
            generated_diff=""
        )
    finally:
        if repo_dir:
            clean_repo_dir(str(repo_dir))


def mini_rollout_func(data: list[dict[str, Any]], config: MiniConfig, **kwargs) -> list[dict[str, Any]]:
    """Deploys parallel Mini-SWE-Agent rollouts against vLLM."""
    logger.info(f"Starting {len(data)} agent rollouts")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=min(len(data), os.cpu_count() or 1)) as ex:
        results = list(ex.map(lambda d: _process_one(d, config), data))

    logger.info(f"Finished {len(data)} rollouts in {time.time() - t0:.2f}s")
    return results


if __name__ == "__main__":
    from src.data.swe_gym import get_swe_gym_repo_repair_dataset

    batch_sizes = [2]
    runs = 1
    data = get_swe_gym_repo_repair_dataset().shuffle(seed=42)

    config = MiniConfig(model="hosted_vllm/Qwen/Qwen3-8B")

    for size in batch_sizes:
        print(f"Testing batch size {size}")
        subset = data.select(range(size))
        subset_dicts = [dict(x) for x in subset]
        times = []
        for i in range(runs):
            t0 = time.time()
            _ = mini_rollout_func(subset_dicts, config)
            times.append(time.time() - t0)
            print(f"  Run {i+1}: {times[-1]:.2f}s")
        print(f"Average time for batch size {size}: {sum(times) / len(times):.2f}s\n")
