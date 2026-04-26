# /// script
# dependencies = [
#     "unsloth",
#     "transformers",
#     "huggingface_hub",
#     "requests",
# ]
# ///
"""
Post-training evaluation script.
Downloads SaiSanjayR/pr-triage-grpo-adapter and runs it against the ENV
to produce evaluation/post_training/trained_summary.json for comparison.
"""
import os, json, re, statistics, requests
from huggingface_hub import snapshot_download

# ── Config ──────────────────────────────────────────────────────────────────
ADAPTER_REPO  = "SaiSanjayR/pr-triage-grpo-adapter"
BASE_MODEL    = "unsloth/Qwen2.5-1.5B-Instruct"
ENV_URL       = os.environ.get("ENV_URL", "http://localhost:7860")
NUM_EPISODES  = 60   # 20 per difficulty
OUTPUT_DIR    = os.path.join("evaluation", "post_training")
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    from unsloth import FastLanguageModel
    print(f"Loading adapter from {ADAPTER_REPO} ...")
    local_dir = snapshot_download(ADAPTER_REPO)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, local_dir)
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def build_prompt(obs: dict) -> str:
    return (
        f"PR Title: {obs.get('title','')}"
        f"\nDescription: {obs.get('description','')}"
        f"\nProposed Code:\n{obs.get('proposed_code','')}"
        f"\nContext:\n{obs.get('context_snippet','')}"
        f"\nLabels: {', '.join(obs.get('labels', []))}"
        "\n\nRespond ONLY with a JSON object containing your review decision."
    )

SYSTEM_PROMPT = (
    "You are an expert code reviewer. Analyze the PR and respond with ONLY a JSON object. "
    'Example: {"review_decision": "request_changes", "blocker_type": "syntax_error", '
    '"defect_category": "logic", "faulty_line": 42, '
    '"reviewer_team": "core-sysdev", "suggested_change": "Fix the null check on line 42"}'
)

def run_inference(model, tokenizer, obs: dict) -> dict:
    prompt = build_prompt(obs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=256, use_cache=True)
    new_tokens = outputs[0][inputs.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    text = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {}

VALID_REVIEW_DECISIONS  = {"approve", "request_changes"}
VALID_BLOCKER_TYPES     = {"debug_output", "hardcoded_secret", "do_not_merge_comment", "debug_test_bypass", "syntax_error"}
VALID_DEFECT_CATEGORIES = {"security", "logic", "performance"}
VALID_REVIEWER_TEAMS    = {"infosec", "devops", "core-frontend", "core-sysdev", "aiml"}

def sanitize_action(raw: dict) -> dict:
    """Map model output to valid Pydantic enum values; drop anything not allowed."""
    decision = str(raw.get("review_decision", "")).lower()
    if decision not in VALID_REVIEW_DECISIONS:
        # Best-effort mapping for common hallucinations
        decision = "request_changes" if any(w in decision for w in ["reject", "change", "block", "deny"]) else "approve"

    def _pick(val, valid_set):
        v = str(val or "").lower()
        return v if v in valid_set else None

    faulty = raw.get("faulty_line")
    try:
        faulty = int(faulty) if faulty is not None else None
        if faulty is not None and faulty < 1:
            faulty = None
    except (ValueError, TypeError):
        faulty = None

    return {
        "review_decision":  decision,
        "blocker_type":     _pick(raw.get("blocker_type"),     VALID_BLOCKER_TYPES),
        "defect_category":  _pick(raw.get("defect_category"),  VALID_DEFECT_CATEGORIES),
        "faulty_line":      faulty,
        "reviewer_team":    _pick(raw.get("reviewer_team"),    VALID_REVIEWER_TEAMS),
        "suggested_change": str(raw.get("suggested_change") or "") or None,
    }

def main():
    # Health check
    try:
        requests.get(f"{ENV_URL}/health", timeout=5).raise_for_status()
        print(f"ENV healthy at {ENV_URL}")
    except Exception as e:
        print(f"ENV not reachable: {e}. Set ENV_URL env var.")
        return

    model, tokenizer = load_model()

    episodes_data = []
    print(f"Running {NUM_EPISODES} post-training episodes...")

    for i in range(1, NUM_EPISODES + 1):
        obs   = requests.post(f"{ENV_URL}/reset").json()
        level = obs.get("task_level", "unknown")

        action = run_inference(model, tokenizer, obs)
        action = sanitize_action(action)  # enforce valid enum values

        # DEBUG: print first 3 episodes so we can verify action format
        if i <= 3:
            print(f"[DEBUG ep{i}] level={level} sanitized_action={action}", flush=True)

        # Send action directly as body (not wrapped in {"action": ...})
        step_resp = requests.post(f"{ENV_URL}/step", json=action)
        result = step_resp.json() if step_resp.ok else {}
        if not step_resp.ok and i <= 3:
            print(f"[DEBUG ep{i}] HTTP {step_resp.status_code}: {step_resp.text[:200]}", flush=True)
        reward = result.get("reward", 0.0)

        if i <= 3:
            print(f"[DEBUG ep{i}] reward={reward} breakdown={result.get('reward_breakdown')}", flush=True)

        episodes_data.append({
            "episode_number": i,
            "pr_id": obs.get("pr_id", "unknown"),
            "task_level": level,
            "reward": reward,
            "reward_breakdown": result.get("reward_breakdown"),
            "raw_action": action,
        })

        if i % 10 == 0:
            print(f"  {i}/{NUM_EPISODES} done ...", flush=True)

    # Stats
    rewards_by_level = {"easy": [], "medium": [], "hard": []}
    all_rewards = []
    for ep in episodes_data:
        lvl = ep["task_level"].lower()
        if lvl in rewards_by_level:
            rewards_by_level[lvl].append(ep["reward"])
        all_rewards.append(ep["reward"])

    def stats(lst):
        if not lst:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        return {
            "mean":  round(statistics.mean(lst), 4),
            "std":   round(statistics.stdev(lst) if len(lst) > 1 else 0.0, 4),
            "count": len(lst),
        }

    summary = {
        "overall": stats(all_rewards),
        "by_difficulty": {
            "easy":   stats(rewards_by_level["easy"]),
            "medium": stats(rewards_by_level["medium"]),
            "hard":   stats(rewards_by_level["hard"]),
        }
    }

    # Print comparison
    baseline_path = os.path.join("evaluation", "pre_training", "baseline_summary.json")
    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)

    print("\n=== POST-TRAINING EVALUATION ===")
    print(f"Overall: {summary['overall']['mean']:.4f}  (baseline: {baseline.get('overall',{}).get('mean','?')})")
    for lvl in ["easy", "medium", "hard"]:
        t  = summary["by_difficulty"][lvl]["mean"]
        b  = baseline.get("by_difficulty", {}).get(lvl, {}).get("mean", "?")
        delta = f"+{t - float(b):.4f}" if isinstance(b, float) else "?"
        print(f"  {lvl.capitalize():8s}: {t:.4f}  (baseline: {b})  delta: {delta}")

    # Save
    results_path = os.path.join(OUTPUT_DIR, "trained_results.json")
    summary_path = os.path.join(OUTPUT_DIR, "trained_summary.json")
    with open(results_path, "w") as f:
        json.dump(episodes_data, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print("\nPASTE THIS INTO ResultsDashboard.jsx as POST_TRAINING:")
    ui_json = {
        lvl: {
            "avg": summary["by_difficulty"][lvl]["mean"],
            "std": summary["by_difficulty"][lvl]["std"],
            "n":   summary["by_difficulty"][lvl]["count"],
        }
        for lvl in ["easy", "medium", "hard"]
    }
    print(json.dumps(ui_json, indent=2))

if __name__ == "__main__":
    main()
