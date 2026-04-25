"""
inference.py — Single-agent, multi-agent, and comparison inference for PRRegressionAuditEnv.

Required env vars:
    API_BASE_URL  - LLM endpoint (default: Groq)
    MODEL_NAME    - Model name (default: llama-3.1-8b-instant)
    HF_TOKEN      - API key
    ENV_URL       - Environment URL (default: localhost:7860)

Usage:
    python inference.py --mode single    --episodes 45
    python inference.py --mode multi     --episodes 45 --verbose
    python inference.py --mode compare
"""

import argparse
import json
import os
import re
import statistics
from typing import Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

from agents.orchestrator import MultiAgentOrchestrator

load_dotenv()

# ---------------------------------------------------------------------------
# Environment / client setup
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HF_TOKEN environment variable not set. LLM calls will fail.")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
    max_retries=3,
    timeout=30.0,
)

# ---------------------------------------------------------------------------
# System prompt — "Senior Code Reviewer" persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior code reviewer performing an automated audit of Pull Requests.
Your job is to detect accidental regressions introduced by developers.
The PR title and description describe the INTENDED feature — the flaw is unrelated to it.
For Hard tasks you will see a Context Snippet showing existing code the PR interacts with.

Respond with ONLY valid, strict JSON matching this schema:
{
  "thought_process": ["step 1 reasoning", "step 2 reasoning"],
  "review_decision": "approve" | "request_changes",
  "blocker_type": "debug_output" | "hardcoded_secret" | "do_not_merge_comment" | "debug_test_bypass" | "syntax_error" | null,
  "defect_category": "security" | "logic" | "performance" | null,
  "faulty_line": <integer 1-indexed in Proposed Code> | null,
  "reviewer_team": "infosec" | "devops" | "core-frontend" | "core-sysdev" | "aiml" | null,
  "suggested_change": "<one concrete sentence under 200 chars>" | null
}

Field rules:
- blocker_type: only for Easy tasks. Null on clean PRs, one of the 5 types on flagged PRs.
- defect_category: for Medium + Hard. Null for Easy.
- faulty_line: 1-indexed line number in Proposed Code where the defect lives. Null for Easy.
- reviewer_team: for Hard only. Null for Easy + Medium.
- suggested_change: for Hard only. MUST be under 200 characters. Name the exact fix.

Output raw JSON only. No markdown. No backticks. No explanation outside the JSON."""


# ---------------------------------------------------------------------------
# parse_action
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> dict:
    text = re.sub(r"```json\s*", "", raw)
    text = re.sub(r"```\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "review_decision": "approve",
        "blocker_type": None,
        "defect_category": None,
        "faulty_line": None,
        "reviewer_team": None,
        "suggested_change": None,
        "thought_process": [],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(obs: dict) -> str:
    """Build the single-agent user prompt from a PR observation."""
    task_level = obs.get("task_level", "easy")
    parts = [
        f"PR Title: {obs.get('title', '')}",
        f"Description: {obs.get('description', '')}",
        f"Labels: {', '.join(obs.get('labels', []) or [])}",
    ]
    if obs.get("proposed_code"):
        parts.append(f"\nProposed Code (1-indexed lines):\n{obs['proposed_code']}")
    if obs.get("context_snippet"):
        parts.append(
            f"\nContext Snippet (existing code/config this PR interacts with):\n{obs['context_snippet']}"
        )
    parts.append(f"\nTask Level: {task_level}\nRespond with ONLY JSON.")
    return "\n".join(parts)


def _call_single_agent(obs: dict) -> dict:
    """Call the single-agent LLM and return the parsed action dict."""
    prompt = _build_prompt(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=600,
            temperature=0.0,
        )
        content = response.choices[0].message.content
    except Exception as exc:
        print(f"Model request failed ({exc}). Using fallback action.")
        content = "{}"
    return parse_action(content)


def _print_results_table(task_scores: dict) -> None:
    """Print the end-of-run results table."""
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    for level in ["easy", "medium", "hard"]:
        scores = task_scores[level]
        if scores:
            avg = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f"  {level:6s}: {avg:.3f} ± {std:.3f}  (n={len(scores)})")
        else:
            print(f"  {level:6s}: no data")


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_episode(
    mode: str = "single",
    orchestrator: Optional[MultiAgentOrchestrator] = None,
    verbose: bool = False,
) -> tuple:
    """Run one PR review episode in single-agent or multi-agent mode.

    Args:
        mode:         "single" uses the unified LLM call path.
                      "multi" delegates to the MultiAgentOrchestrator.
        orchestrator: Required when mode is "multi".
        verbose:      If True, print the orchestrator trace summary per episode.

    Returns:
        Tuple of (score, task_level, mode).
    """
    obs = requests.post(f"{ENV_URL}/reset", timeout=10).json()
    task_level = obs.get("task_level", "easy")
    print(f"[START] task={task_level} mode={mode}", flush=True)

    if mode == "single":
        action = _call_single_agent(obs)
    else:
        try:
            if orchestrator is None:
                raise ValueError("orchestrator must be provided for multi mode")
            action = orchestrator.run(obs)
            if verbose:
                summary = orchestrator.get_trace_summary()
                print(
                    f"  [TRACE] ep={summary['episode']} "
                    f"agents={summary['agents_run']} "
                    f"avg_conf={summary['avg_confidence']}"
                )
        except Exception as exc:
            print(f"  [WARN] Multi-agent failed ({exc}). Falling back to single-agent.")
            action = _call_single_agent(obs)

    action.pop("thought_process", None)
    result = requests.post(f"{ENV_URL}/step", json=action, timeout=10).json()
    score = float(result.get("reward") if result.get("reward") is not None else 0.001)

    breakdown = result.get("reward_breakdown") or {}
    breakdown_str = "  ".join(f"{k}={v:.2f}" for k, v in breakdown.items())
    print(f"[STEP] reward={score}  breakdown=[ {breakdown_str} ]", flush=True)
    print(f"[END] task={task_level} score={score} steps=1 mode={mode}", flush=True)
    return score, task_level, mode


def run_comparison_episode(orchestrator: MultiAgentOrchestrator) -> dict:
    """Compare single-agent vs multi-agent action dicts for one PR (no submission).

    Args:
        orchestrator: An initialised MultiAgentOrchestrator instance.

    Returns:
        Dict with single_action, multi_action, task_level, pr_title.
    """
    obs = requests.post(f"{ENV_URL}/reset", timeout=10).json()
    task_level: str = obs.get("task_level", "easy")
    pr_title: str = obs.get("title", "")

    single_action = _call_single_agent(obs)
    single_action.pop("thought_process", None)

    try:
        multi_action = orchestrator.run(obs)
        multi_action.pop("thought_process", None)
    except Exception as exc:
        print(f"  [WARN] Multi-agent failed in comparison ({exc}). Using fallback.")
        multi_action = {
            "review_decision": "approve",
            "blocker_type": None,
            "defect_category": None,
            "faulty_line": None,
            "reviewer_team": None,
            "suggested_change": None,
        }

    return {
        "single_action": single_action,
        "multi_action":  multi_action,
        "task_level":    task_level,
        "pr_title":      pr_title,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — parse CLI arguments and run the chosen inference mode."""
    parser = argparse.ArgumentParser(
        description="PRRegressionAuditEnv inference — single, multi, or compare mode."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "compare"],
        default="multi",
        help="Inference mode: single (one LLM call), multi (4-agent orchestrator), or compare.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=45,
        help="Number of episodes to run (ignored in compare mode, which runs 20).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode orchestrator trace summaries in multi mode.",
    )
    args = parser.parse_args()

    print(f"PRRegressionAuditEnv Inference")
    print(f"Environment: {ENV_URL}")
    print(f"Model: {MODEL_NAME}  |  Mode: {args.mode}")
    print("-" * 55)

    orchestrator = MultiAgentOrchestrator(
        client=client,
        model_name=MODEL_NAME,
        temperature=0.0,
    )

    # ------------------------------------------------------------------
    # COMPARE MODE — shows side-by-side single vs multi decisions
    # ------------------------------------------------------------------
    if args.mode == "compare":
        compare_episodes = 20
        print(f"Running {compare_episodes} comparison episodes (actions only, not submitted).\n")
        print(
            f"{'#':>3}  {'Level':8}  "
            f"{'Single decision':^16}  {'Single line':^11}  {'Single team':^14}  "
            f"{'Multi decision':^16}  {'Multi line':^11}  {'Multi team':^14}  "
            f"PR Title"
        )
        print("-" * 130)

        for ep in range(1, compare_episodes + 1):
            try:
                comp = run_comparison_episode(orchestrator)
                sa = comp["single_action"]
                ma = comp["multi_action"]
                level = comp["task_level"]
                title = comp["pr_title"][:38]

                print(
                    f"{ep:>3}  {level:8}  "
                    f"{str(sa.get('review_decision', '-')):^16}  "
                    f"{str(sa.get('faulty_line', '-')):^11}  "
                    f"{str(sa.get('reviewer_team', '-')):^14}  "
                    f"{str(ma.get('review_decision', '-')):^16}  "
                    f"{str(ma.get('faulty_line', '-')):^11}  "
                    f"{str(ma.get('reviewer_team', '-')):^14}  "
                    f"{title}"
                )
            except Exception as exc:
                print(f"{ep:>3}  ERROR: {exc}")

        print("\nCompare run complete. No scores submitted.")
        return

    # ------------------------------------------------------------------
    # SINGLE / MULTI MODE
    # ------------------------------------------------------------------
    task_scores: dict = {"easy": [], "medium": [], "hard": []}
    all_confidences: list = []

    for ep in range(1, args.episodes + 1):
        try:
            score, level, used_mode = run_episode(
                mode=args.mode,
                orchestrator=orchestrator,
                verbose=args.verbose,
            )
            task_scores[level].append(score)
            print(f"  Ep {ep:02d} [{level:6s}] score={score:.3f} mode={used_mode}")

            if args.mode == "multi" and orchestrator.agent_trace:
                ep_avg = sum(r.confidence for r in orchestrator.agent_trace) / len(orchestrator.agent_trace)
                all_confidences.append(ep_avg)

        except Exception as exc:
            print(f"  Ep {ep:02d} ERROR: {exc}")

    _print_results_table(task_scores)

    if args.mode == "multi" and all_confidences:
        avg_conf = statistics.mean(all_confidences)
        std_conf = statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0.0
        print("\nMULTI-AGENT CONFIDENCE STATS")
        print("-" * 50)
        print(f"  Avg episode confidence: {avg_conf:.3f} ± {std_conf:.3f}  (n={len(all_confidences)})")
        print(f"  Min: {min(all_confidences):.3f}   Max: {max(all_confidences):.3f}")


if __name__ == "__main__":
    main()
