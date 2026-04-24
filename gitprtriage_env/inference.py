"""
inference.py — Single-agent, multi-agent, and comparison inference for DevTriageEnv.

Required env vars:
  API_BASE_URL  - LLM endpoint
  MODEL_NAME    - Model name
  HF_TOKEN      - API key (works for both HF and Groq)
  ENV_URL       - Environment URL

Usage:
  python inference.py --mode single    --episodes 60
  python inference.py --mode multi     --episodes 60 --verbose
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
# Environment / client setup (preserved exactly from original)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")      # reused for Groq key too
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
    max_retries=3,          # Practical Fix: Limited 3 retries for rate limits
    timeout=30.0            # Practical Fix: 30s timeout for cold starts
)

# ---------------------------------------------------------------------------
# System prompt (preserved exactly from original)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior software engineer triaging GitHub issues.
Respond with ONLY valid, strict JSON matching this schema:
{
  "type": "object",
  "properties": {
    "thought_process": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Step-by-step reasoning. 1. Analyze the issue. 2. Identify the category. 3. Look for the exact line number of the bug in the code snippet. 4. Determine the internal team. 5. Draft the fix. ALWAYS DO THIS FIRST."
    },
    "classification": {
      "type": "string",
      "enum": ["bug", "feature", "duplicate"],
      "description": "Required. The category of the issue."
    },
    "bug_line": {
      "type": ["integer", "null"],
      "description": "The 1-indexed line number where the bug is located. Set to null if there is no code snippet or no obvious bug."
    },
    "team": {
      "type": ["string", "null"],
      "enum": ["webdev", "devops", "aiml", null],
      "description": "The team best suited to handle this. webdev=frontend/backend/api, devops=infra/ci/docker/k8s, aiml=ml/models. Set to null if unable to determine."
    },
    "suggested_fix": {
      "type": ["string", "null"],
      "description": "One concrete sentence suggesting HOW to fix the bug. Do not just say 'fix the bug'. Set to null if no clear fix."
    }
  },
  "required": ["thought_process", "classification", "bug_line", "team", "suggested_fix"]
}
No markdown formatting, no backticks, no markdown JSON blocks. Output exactly the raw JSON text. Make sure bug_line is exactly an integer or null."""

# ---------------------------------------------------------------------------
# parse_action (preserved exactly from original)
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
    return {"classification": "bug", "bug_line": None,
            "team": None, "suggested_fix": None, "thought_process": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(obs: dict) -> str:
    """Build the single-agent user prompt from an environment observation.

    Identical logic to the original ``run_episode``, extracted so it can be
    shared with the comparison path.

    Args:
        obs: The observation dict returned by ``POST /reset``.

    Returns:
        Formatted string ready for the LLM user turn.
    """
    task_level = obs.get("task_level", "easy")
    parts = [f"Title: {obs['title']}", f"Body: {obs['body']}"]
    if obs.get("code_snippet"):
        parts.append(f"\nCode (lines are 1-indexed):\n{obs['code_snippet']}")
    if obs.get("existing_labels"):
        parts.append(f"Labels: {', '.join(obs['existing_labels'])}")
    parts.append(f"\nTask Level: {task_level}\nRespond with ONLY JSON.")
    return "\n".join(parts)


def _call_single_agent(obs: dict) -> dict:
    """Call the single-agent LLM and return the parsed action dict.

    This is the identical logic from the original ``run_episode``, factored
    out so it can be used from both ``run_episode`` and comparison paths.

    Args:
        obs: The observation dict returned by ``POST /reset``.

    Returns:
        Parsed action dict (always has all required keys).
    """
    prompt = _build_prompt(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=600,
            temperature=0.1,
        )
        content = response.choices[0].message.content
    except Exception as exc:
        print(f"Model request failed ({exc}). Using fallback action.")
        content = "{}"
    return parse_action(content)


def _print_results_table(task_scores: dict[str, list[float]]) -> None:
    """Print the end-of-run results table identical to the original format.

    Args:
        task_scores: Dict mapping task level to list of float scores.
    """
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
) -> tuple[float, str, str]:
    """Run one triage episode in single-agent or multi-agent mode.

    Args:
        mode:         ``"single"`` uses the original LLM call path.
                      ``"multi"`` delegates to the orchestrator.
        orchestrator: Required when mode is ``"multi"``. Ignored for ``"single"``.
        verbose:      If True, print the orchestrator trace summary after each
                      multi-agent episode.

    Returns:
        A tuple of ``(score, task_level, mode)`` where score is a float in
        [0.0, 1.0], task_level is one of ``"easy"``, ``"medium"``, ``"hard"``,
        and mode mirrors the argument.

    Note:
        The multi-agent path falls back to single-agent automatically on any
        unhandled exception, printing a warning to stdout.
    """
    obs = requests.post(f"{ENV_URL}/reset", timeout=10).json()
    task_level = obs.get("task_level", "easy")
    print(f"[START] task={task_level} mode={mode}", flush=True)

    # --- Single-agent path (100% identical to original) ---
    if mode == "single":
        action = _call_single_agent(obs)

    # --- Multi-agent path ---
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

    action.pop("thought_process", None)  # Clean internal reasoning payload before sending to strict API
    result = requests.post(f"{ENV_URL}/step", json=action, timeout=10).json()
    score = float(result.get("reward") if result.get("reward") is not None else 0.001)

    print(f"[STEP] step=1 reward={score}", flush=True)
    print(f"[END] task={task_level} score={score} steps=1 mode={mode}", flush=True)
    return score, task_level, mode


def run_comparison_episode(orchestrator: MultiAgentOrchestrator) -> dict:
    """Collect and compare single-agent vs multi-agent action dicts for one issue.

    Calls ``POST /reset`` exactly once to get the observation, then builds both
    action dicts independently without submitting either to the environment.
    The caller is responsible for any submission or scoring.

    Args:
        orchestrator: An initialised ``MultiAgentOrchestrator`` instance.

    Returns:
        A dict with keys:
            - ``single_action`` — action dict from the single LLM agent
            - ``multi_action``  — action dict from the orchestrator
            - ``task_level``    — difficulty level of the fetched issue
            - ``issue_title``   — title string of the fetched issue
    """
    obs = requests.post(f"{ENV_URL}/reset", timeout=10).json()
    task_level: str = obs.get("task_level", "easy")
    issue_title: str = obs.get("title", "")

    # Single-agent — use standard call path, strip reasoning payload
    single_action = _call_single_agent(obs)
    single_action.pop("thought_process", None)

    # Multi-agent — use orchestrator, fall back gracefully
    try:
        multi_action = orchestrator.run(obs)
        multi_action.pop("thought_process", None)
    except Exception as exc:
        print(f"  [WARN] Multi-agent failed in comparison ({exc}). Using fallback.")
        multi_action = {
            "classification": "bug",
            "bug_line": None,
            "team": None,
            "suggested_fix": None,
        }

    return {
        "single_action": single_action,
        "multi_action":  multi_action,
        "task_level":    task_level,
        "issue_title":   issue_title,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — parse CLI arguments and run the chosen inference mode."""
    parser = argparse.ArgumentParser(
        description="DevTriageEnv inference — single, multi, or compare mode."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "compare"],
        default="multi",
        help="Inference mode: single (original LLM), multi (orchestrator), or compare.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=60,
        help="Number of episodes to run (ignored in compare mode, which always runs 20).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode orchestrator trace summaries in multi mode.",
    )
    args = parser.parse_args()

    print(f"Running inference against: {ENV_URL}")
    print(f"Model: {MODEL_NAME}  |  Mode: {args.mode}")
    print("-" * 50)

    # Build orchestrator once — used by both "multi" and "compare" modes.
    orchestrator = MultiAgentOrchestrator(
        client=client,
        model_name=MODEL_NAME,
        temperature=0.0,
    )

    # ------------------------------------------------------------------
    # COMPARE MODE
    # ------------------------------------------------------------------
    if args.mode == "compare":
        compare_episodes = 20
        print(f"Running {compare_episodes} comparison episodes (actions only, not submitted).\n")

        col_w = 18
        header = (
            f"{'#':>3}  {'Level':8}  "
            f"{'Single cls':^{col_w}}  {'Single line':^11}  {'Single team':^11}  "
            f"{'Multi cls':^{col_w}}  {'Multi line':^11}  {'Multi team':^11}  "
            f"Title"
        )
        print(header)
        print("-" * len(header))

        for ep in range(1, compare_episodes + 1):
            try:
                comp = run_comparison_episode(orchestrator)
                sa = comp["single_action"]
                ma = comp["multi_action"]
                level = comp["task_level"]
                title = comp["issue_title"][:40]

                print(
                    f"{ep:>3}  {level:8}  "
                    f"{str(sa.get('classification', '-')):^{col_w}}  "
                    f"{str(sa.get('bug_line', '-')):^11}  "
                    f"{str(sa.get('team', '-')):^11}  "
                    f"{str(ma.get('classification', '-')):^{col_w}}  "
                    f"{str(ma.get('bug_line', '-')):^11}  "
                    f"{str(ma.get('team', '-')):^11}  "
                    f"{title}"
                )
            except Exception as exc:
                print(f"{ep:>3}  ERROR: {exc}")

        print("\nCompare run complete. No scores submitted.")
        return

    # ------------------------------------------------------------------
    # SINGLE / MULTI MODE
    # ------------------------------------------------------------------
    task_scores: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}

    # Confidence tracking for multi mode summary
    all_confidences: list[float] = []

    for ep in range(1, args.episodes + 1):
        try:
            score, level, used_mode = run_episode(
                mode=args.mode,
                orchestrator=orchestrator,
                verbose=args.verbose,
            )
            task_scores[level].append(score)
            print(f"  Ep {ep:02d} [{level:6s}] score={score:.3f} mode={used_mode}")

            # Collect confidence data from multi-agent trace
            if args.mode == "multi" and orchestrator.agent_trace:
                ep_avg = sum(r.confidence for r in orchestrator.agent_trace) / len(orchestrator.agent_trace)
                all_confidences.append(ep_avg)

        except Exception as exc:
            print(f"  Ep {ep:02d} ERROR: {exc}")

    # --- Results table (identical format to original) ---
    _print_results_table(task_scores)

    # --- Additional multi-mode confidence summary ---
    if args.mode == "multi" and all_confidences:
        avg_conf = statistics.mean(all_confidences)
        std_conf = statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0.0
        print("\nMULTI-AGENT CONFIDENCE STATS")
        print("-" * 50)
        print(f"  Avg episode confidence: {avg_conf:.3f} ± {std_conf:.3f}  (n={len(all_confidences)})")
        print(f"  Min: {min(all_confidences):.3f}   Max: {max(all_confidences):.3f}")


if __name__ == "__main__":
    main()
