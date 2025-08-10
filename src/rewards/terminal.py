import json
import re
from typing import Any


def _parse_calls(text: str) -> list[tuple[str, bool, str]]:
    if not isinstance(text, str) or not text:
        return []
    calls: list[tuple[str, bool, str]] = []
    call_iter = list(re.finditer(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text))
    resp_iter = list(re.finditer(r"<tool_response>\s*([\s\S]*?)\s*</tool_response>", text))
    responses = [(m.start(), m.group(1) or "") for m in resp_iter]
    for call_m in call_iter:
        cmd = None
        try:
            parsed = json.loads(call_m.group(1))
            args = parsed.get("arguments", {}) if isinstance(parsed, dict) else {}
            cmd = args.get("cmd") if isinstance(args, dict) else None
        except Exception:
            cmd = None
        if not isinstance(cmd, str) or not cmd.strip():
            continue
        success = False
        resp_text = ""
        for (pos, content) in responses:
            if pos > call_m.end():
                cn = (content or "").strip().lower()
                if not ("command failed with exit code" in cn or "command timed out" in cn or "shell execution failed" in cn):
                    success = True
                resp_text = content or ""
                break
        calls.append((cmd.strip(), success, resp_text))
    return calls


def terminal_debugging_habits_reward_func(completion: list[str], **kwargs) -> list[float]:
    """Reward terminal debugging behaviors from completion transcript only.

    Rewards added directly:
    - +0.3 if ls appears within the first 2 successful shell commands
    - +0.4 if one of {rg, grep, find} appears within the first 10 successful shell commands
    - +0.3 if one of {sed, head, tail} appears anywhere among successful shell commands
    """
    items: list[str] = list(completion or [])
    re_ls = re.compile(r"\bls\b")
    re_search = re.compile(r"\b(rg|grep|find)\b")
    re_slice = re.compile(r"\b(sed|head|tail)\b")

    rewards: list[float] = []
    for text in items:
        triples = _parse_calls(text or "")
        calls = [(cmd, ok) for cmd, ok, _ in triples]
        # +0.3 if ls appears within the first 2 successful shell commands
        ls_early_reward = 0.3 if any(re_ls.search(cmd) and ok for cmd, ok in calls[:2]) else 0.0
        # +0.4 if one of {rg, grep, find} appears within the first 10 successful shell commands
        search_early_reward = 0.4 if any(re_search.search(cmd) and ok for cmd, ok in calls[:10]) else 0.0
        # +0.3 if one of {sed, head, tail} appears after at least two prior commands
        slice_any_reward = 0.3 if any(ok and re_slice.search(cmd) for cmd, ok in calls[2:]) else 0.0
        total = ls_early_reward + search_early_reward + slice_any_reward
        rewards.append(max(0.0, min(1.0, total)))
    return rewards


def terminal_exploration_depth_reward_func(completion: list[str], **kwargs) -> list[float]:
    """Richer terminal shaping to differentiate productive exploration when no diff is produced.

    Components (sum of bonuses <= 0.8, with up to 0.2 penalties; final clipped to [0,1]):
    - +0.25 chain: a search (rg/grep/find) whose response yields a path that is later used in a precise read (sed/head/tail)
    - +0.20 success ratio: fraction of successes among first 6 shell calls (linear)
    - +0.20 precise reads: presence of sed -n 'a,bp' or head/tail -n anywhere
    - +0.15 scoped search: presence of search flags (-n/--type) or directory scoping in early search
    - up to -0.20 penalty: duplicated identical commands and repeated failures; truncated outputs before any precise read
    """
    items: list[str] = list(completion or [])
    re_search = re.compile(r"\b(rg|grep|find)\b")
    re_slice = re.compile(r"\b(sed|head|tail)\b")
    re_precise_slice = re.compile(r"\b(sed\s+-n\s+'?\d+\s*,\s*\d+\s*p'?|head\s+-n\s+\d+|tail\s+-n\s+\d+)\b")
    # Match common scoping flags; -n is preceded by space or start, --type is word-bounded
    re_search_flags = re.compile(r"(?:^|\s)-(?:n)\b|\b--type=\w+\b")
    re_path_token = re.compile(r"[\w./-]+\.(py|ts|js|java|go|rs|md|txt|json|yaml|yml|toml|ini)\b|\b(?:src|lib|tests?|docs)/[\w./-]+\b")

    rewards: list[float] = []
    for text in items:
        triples = _parse_calls(text or "")
        cmds = [c for c, _, _ in triples]
        oks = [ok for _, ok, _ in triples]
        resps = [r for _, _, r in triples]

        # +0.25: search → response contains path → later precise read on that path
        bonus_chain = 0.0
        for idx, (cmd, ok, resp) in enumerate(triples[:10]):
            if ok and re_search.search(cmd):
                # Find path-like tokens in response
                raw_paths = re_path_token.findall(resp)
                # re groups may return tuples; normalize to strings
                norm_paths = set(p if isinstance(p, str) else ''.join(p) for p in raw_paths)
                if norm_paths:
                    later = triples[idx + 1 :]
                    if any(ok2 and re_slice.search(cmd2) and any(p in cmd2 for p in norm_paths) for cmd2, ok2, _ in later):
                        bonus_chain = 0.25
                break

        # +0.20: success ratio in first 6 (apply only when full 6 cmds are present)
        first6 = oks[:6]
        if len(first6) == 6:
            bonus_success = 0.20 * (sum(1 for x in first6 if x) / len(first6))
        else:
            bonus_success = 0.0

        # +0.20: precise reads anywhere (but do not double-count with chain; tests expect isolation)
        bonus_precise = 0.0
        if bonus_chain == 0.0:
            if any(ok and re_precise_slice.search(cmd) for cmd, ok in zip(cmds, oks)):
                bonus_precise = 0.20

        # +0.15: scoped search in early search (flags or actionable path in response)
        bonus_scoped = 0.0
        for (cmd, ok, resp) in triples[:10]:
            if ok and re_search.search(cmd):
                if re_search_flags.search(cmd) or "/" in cmd or re_path_token.search(resp):
                    bonus_scoped = 0.15
                    break

        # do not combine scoped bonus with chain bonus to avoid over-crediting simple sequences
        if bonus_chain > 0.0:
            bonus_scoped = 0.0

        # Penalties (capped at 0.20) — scale down each to 0.07/0.10 per tests
        penalty = 0.0
        # duplicate identical commands in first 10
        seen = set()
        for cmd in cmds[:10]:
            if cmd in seen:
                penalty += 0.07
                break
            seen.add(cmd)
        # consecutive failures of same command
        for i in range(1, min(10, len(triples))):
            if not oks[i] and not oks[i - 1] and cmds[i].strip() == cmds[i - 1].strip():
                penalty += 0.07
                break
        # truncated outputs before any precise read (case-insensitive, partial match)
        if bonus_precise == 0.0 and any("truncated" in (r.lower()) for r in resps[:10]):
            penalty += 0.10

        total = max(0.0, min(1.0, bonus_chain + bonus_success + bonus_precise + bonus_scoped - min(penalty, 0.20)))
        rewards.append(total)
    return rewards
