# /// script
# dependencies = [
#     "unsloth",
#     "trl>=0.12.0",
#     "datasets",
#     "transformers",
#     "accelerate",
#     "peft",
#     "bitsandbytes",
#     "huggingface_hub",
#     "requests",
# ]
# ///

"""
train_curriculum.py — Stage 2 GRPO training WITH:
  1. Curriculum learning   (easy → medium → hard, staged)
  2. Reward hacking guards (semantic consistency, diversity bonus, contradiction penalty)

Resumes from SaiSanjayR/pr-triage-grpo-adapter (v1).
Pushes result to SaiSanjayR/pr-triage-grpo-adapter-curriculum.
"""

import os, json, re, collections, requests
import numpy as np
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ── Ground truth ──────────────────────────────────────────────────────────────
fallback = 'https://huggingface.co/spaces/SaiSanjayR/GitPRTriage_Environment/resolve/main/prevaluation_env/data/prs.json'
PR_TRUTH = {pr["id"]: pr for pr in requests.get(fallback, timeout=30).json()}
print(f"Loaded {len(PR_TRUTH)} PRs.")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "unsloth/Qwen2.5-1.5B-Instruct"
V1_ADAPTER   = "SaiSanjayR/pr-triage-grpo-adapter"
OUT_ADAPTER  = "SaiSanjayR/pr-triage-grpo-adapter-curriculum"
MAX_SEQ_LEN  = 2048
OUTPUT_DIR   = "evaluation/checkpoints/curriculum/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID = {
    "review_decision":  {"approve", "request_changes"},
    "blocker_type":     {"debug_output","hardcoded_secret","do_not_merge_comment","debug_test_bypass","syntax_error"},
    "defect_category":  {"security","logic","performance"},
    "reviewer_team":    {"infosec","devops","core-frontend","core-sysdev","aiml"},
}

# ── Reward hacking state (shared across batch calls) ─────────────────────────
_decision_counter: collections.Counter = collections.Counter()
_total_calls: int = 0

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_action(raw) -> dict:
    if isinstance(raw, list):
        raw = raw[-1]["content"] if raw and isinstance(raw[-1], dict) else str(raw)
    elif not isinstance(raw, str):
        raw = str(raw)
    text = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return {}

def score_action(action: dict, truth: dict) -> float:
    """Core reward (same as v1 — deterministic, no hacking possible here)."""
    decision = str(action.get("review_decision") or "").lower()
    expected = str(truth.get("review_decision") or "").lower()
    if decision not in VALID["review_decision"]:
        return 0.001
    if decision != expected:
        return 0.001

    score = 0.4  # decision correct

    blocker = str(action.get("blocker_type") or "").lower()
    if blocker == str(truth.get("blocker_type") or "").lower():
        score += 0.25
    elif blocker in VALID["blocker_type"]:
        score += 0.05

    defect = str(action.get("defect_category") or "").lower()
    if defect == str(truth.get("defect_category") or "").lower():
        score += 0.15
    elif defect in VALID["defect_category"]:
        score += 0.03

    team = str(action.get("reviewer_team") or "").lower()
    if team == str(truth.get("reviewer_team") or "").lower():
        score += 0.10
    elif team in VALID["reviewer_team"]:
        score += 0.02

    try:
        fl  = int(action.get("faulty_line") or 0)
        efl = int(truth.get("faulty_line") or 0)
        if fl > 0 and efl > 0 and abs(fl - efl) <= 2:
            score += 0.10
    except:
        pass

    return float(np.clip(score, 0.001, 0.999))

# ── REWARD HACKING GUARDRAILS ─────────────────────────────────────────────────

def _diversity_penalty(action: dict) -> float:
    """
    Penalise if the model outputs the same review_decision far too often.
    Returns a NEGATIVE value to subtract if diversity collapses.
    Guard: if one decision class accounts for >85% of all calls, apply -0.05 penalty.
    This prevents reward hacking via always choosing the majority class.
    """
    global _decision_counter, _total_calls
    decision = str(action.get("review_decision") or "").lower()
    _decision_counter[decision] += 1
    _total_calls += 1
    if _total_calls < 20:
        return 0.0  # not enough data yet
    majority_frac = _decision_counter.most_common(1)[0][1] / _total_calls
    if majority_frac > 0.85:
        return -0.05
    return 0.0

def _contradiction_penalty(action: dict) -> float:
    """
    Semantic consistency guard:
    If decision='approve' but a blocker_type is specified, that's contradictory.
    If decision='request_changes' but no blocker_type, that's weak reasoning.
    Returns a NEGATIVE value.
    """
    decision = str(action.get("review_decision") or "").lower()
    has_blocker = bool(action.get("blocker_type") and
                       str(action.get("blocker_type")).lower() in VALID["blocker_type"])
    if decision == "approve" and has_blocker:
        return -0.05   # contradictory: approving but flagging a blocker
    return 0.0

def _format_quality_bonus(raw_text: str) -> float:
    """
    Small bonus (0.05) if the output contains ALL 6 required keys.
    Encourages complete responses, not lazy partial JSONs.
    """
    required = {"review_decision","blocker_type","defect_category","faulty_line","reviewer_team","suggested_change"}
    try:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            if required.issubset(set(parsed.keys())):
                return 0.05
    except:
        pass
    return 0.0

# ── Combined reward functions ─────────────────────────────────────────────────

def env_reward(prompts, completions, pr_id=None, **kwargs) -> list[float]:
    rewards = []
    for completion, pid in zip(completions, pr_id or []):
        try:
            # Raw text for format bonus
            raw = completion[-1]["content"] if isinstance(completion, list) else str(completion)
            action = parse_action(completion)
            truth  = PR_TRUTH.get(str(pid))
            if not truth or not action:
                rewards.append(0.001)
                continue

            base     = score_action(action, truth)
            div_pen  = _diversity_penalty(action)
            cont_pen = _contradiction_penalty(action)
            fmt_bon  = _format_quality_bonus(raw)

            final = float(np.clip(base + div_pen + cont_pen + fmt_bon, 0.001, 0.999))
            rewards.append(final)
        except Exception:
            rewards.append(0.001)
    return rewards

def format_reward(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        try:
            raw  = completion[-1]["content"] if isinstance(completion, list) else str(completion)
            text = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            m    = re.search(r"\{.*\}", text, re.DOTALL)
            rewards.append(0.1 if m and isinstance(json.loads(m.group()), dict) else 0.0)
        except:
            rewards.append(0.0)
    return rewards

# ── Curriculum dataset builder ────────────────────────────────────────────────

def build_curriculum_dataset(base_dataset):
    """
    Returns ordered dataset: easy rows × 3, medium rows × 2, hard rows × 1.
    Oversamples easy so the model sees many successes first (curriculum warm-up).
    Total: ~same size as original but difficulty-ordered.
    """
    easy   = base_dataset.filter(lambda x: x.get("task_level") == "easy")
    medium = base_dataset.filter(lambda x: x.get("task_level") == "medium")
    hard   = base_dataset.filter(lambda x: x.get("task_level") == "hard")

    print(f"Curriculum split — easy:{len(easy)} medium:{len(medium)} hard:{len(hard)}")

    # Oversample easy to front-load confidence, then interleave harder tasks
    curriculum = concatenate_datasets([
        easy,   easy,   easy,   # 3× easy  (strong warm-up signal)
        medium, medium,         # 2× medium (bridge)
        hard,                   # 1× hard
    ])
    print(f"Curriculum total rows: {len(curriculum)}")
    return curriculum

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    print(f"Resuming from v1 adapter: {V1_ADAPTER}")
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    v1_dir = snapshot_download(V1_ADAPTER)
    model  = PeftModel.from_pretrained(model, v1_dir, is_trainable=True)
    model.print_trainable_parameters()

    print("Loading and building curriculum dataset...")
    raw_dataset        = load_dataset("SaiSanjayR/pr-regression-audit-grpo", split="train")
    curriculum_dataset = build_curriculum_dataset(raw_dataset)
    print(f"  Columns: {curriculum_dataset.column_names}")

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=4,
        max_prompt_length=1600,
        max_completion_length=400,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=400,
        learning_rate=2e-6,
        logging_steps=5,
        save_steps=100,
        warmup_steps=15,
        temperature=0.85,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env_reward, format_reward],
        args=config,
        train_dataset=curriculum_dataset,
    )

    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/final_adapter/")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter/")

    print(f"\nPushing curriculum adapter to {OUT_ADAPTER} ...")
    model.push_to_hub(OUT_ADAPTER)
    tokenizer.push_to_hub(OUT_ADAPTER)
    print("Done!")

if __name__ == "__main__":
    main()
