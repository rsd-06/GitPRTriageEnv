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
#     "numpy",
# ]
# ///

"""
train_v2.py — Stage 2 GRPO (Curriculum) training.

  • Resumes from rsd-06/pr-regression-audit-grpo-adapter (v1 trained adapter)
  • Trains for 600 steps (vs 400 in v1) with curriculum ordering
  • Reward hacking guards: diversity penalty + contradiction penalty
  • Pushes output to rsd-06/pr-regression-audit-grpo-adapter-v2
  • Post-training stats written to evaluation/post_training/trained_summary.json
"""

import os, json, re, collections, requests
import numpy as np
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ── Ground truth ──────────────────────────────────────────────────────────────
GT_URL   = "https://huggingface.co/spaces/rsd-06/PRRegressionAuditEnv/resolve/main/prevaluation_env/data/prs.json"
PR_TRUTH = {pr["id"]: pr for pr in requests.get(GT_URL, timeout=30).json()}
print(f"Loaded {len(PR_TRUTH)} PRs from ground truth.")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "unsloth/Qwen2.5-1.5B-Instruct"
V1_ADAPTER   = "SaiSanjayR/pr-triage-grpo-adapter"          # SAI's trained v1 adapter
OUT_ADAPTER  = "rsd-06/pr-regression-audit-grpo-adapter-v2"
MAX_SEQ_LEN  = 2048
OUTPUT_DIR   = "evaluation/checkpoints/v2/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("evaluation/post_training", exist_ok=True)

VALID = {
    "review_decision":  {"approve", "request_changes"},
    "blocker_type":     {"debug_output","hardcoded_secret","do_not_merge_comment","debug_test_bypass","syntax_error"},
    "defect_category":  {"security","logic","performance"},
    "reviewer_team":    {"infosec","devops","core-frontend","core-sysdev","aiml"},
}

# ── Reward hacking state ──────────────────────────────────────────────────────
_decision_counter: collections.Counter = collections.Counter()
_total_calls: int = 0

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_action(raw) -> dict:
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict) and "content" in raw[0]:
            raw = raw[0]["content"]
        elif len(raw) > 0 and isinstance(raw[-1], dict):
            raw = raw[-1]["content"]
        else:
            raw = str(raw)
    if not isinstance(raw, str):
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
    """Deterministic ground-truth scorer (mirrors v1 logic)."""
    level = truth["task_level"]
    score = 0.0
    decision = str(action.get("review_decision") or "").lower()
    true_decision = str(truth.get("true_decision") or "").lower()

    if level == "easy":
        if decision == true_decision:
            score += 0.55
        blocker = str(action.get("blocker_type") or "").lower().strip()
        true_blocker = truth.get("true_blocker_type")
        if true_blocker is None:
            if not blocker:
                score += 0.45
        elif blocker == str(true_blocker).lower():
            score += 0.45

    elif level == "medium":
        if decision == "request_changes":
            score += 0.10
        cat = str(action.get("defect_category") or "").lower()
        true_cat = str(truth.get("true_defect_category") or "").lower()
        if cat == true_cat:
            score += 0.40
        fl = action.get("faulty_line")
        tfl = truth.get("true_faulty_line")
        if fl is not None and tfl is not None:
            try:
                diff = abs(int(fl) - int(tfl))
                if diff == 0: score += 0.35
                elif diff == 1: score += 0.15
            except: pass

    elif level == "hard":
        if decision == "request_changes":
            score += 0.05
        cat = str(action.get("defect_category") or "").lower()
        if cat == str(truth.get("true_defect_category") or "").lower():
            score += 0.20
        fl = action.get("faulty_line")
        tfl = truth.get("true_faulty_line")
        if fl is not None and tfl is not None:
            try:
                diff = abs(int(fl) - int(tfl))
                if diff == 0: score += 0.25
                elif diff == 1: score += 0.10
            except: pass
        team = str(action.get("reviewer_team") or "").lower()
        if team == str(truth.get("true_reviewer_team") or "").lower():
            score += 0.25
        suggestion = str(action.get("suggested_change") or "").lower()
        keywords = [k.lower() for k in truth.get("true_fix_keywords", [])]
        if suggestion and keywords:
            matched = sum(1 for k in keywords if k in suggestion)
            if matched >= 2: score += 0.25
            elif matched == 1: score += 0.15
            elif len(suggestion) > 10: score += 0.05

    return float(np.clip(score, 0.001, 0.999))

# ── Reward hacking guardrails ──────────────────────────────────────────────────
def _diversity_penalty(action: dict) -> float:
    """Penalise if the model always picks the same review_decision (>85% of calls)."""
    global _decision_counter, _total_calls
    decision = str(action.get("review_decision") or "").lower()
    _decision_counter[decision] += 1
    _total_calls += 1
    if _total_calls < 20:
        return 0.0
    majority_frac = _decision_counter.most_common(1)[0][1] / _total_calls
    if majority_frac > 0.85:
        return -0.05
    return 0.0

def _contradiction_penalty(action: dict) -> float:
    """Penalise approve + blocker_type (contradiction)."""
    decision = str(action.get("review_decision") or "").lower()
    has_blocker = bool(action.get("blocker_type") and
                       str(action.get("blocker_type")).lower() in VALID["blocker_type"])
    if decision == "approve" and has_blocker:
        return -0.05
    return 0.0

def _format_quality_bonus(raw_text: str) -> float:
    """Small bonus if output contains ALL 6 required keys."""
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

# ── Reward functions ──────────────────────────────────────────────────────────
def env_reward(completions, pr_id, **kwargs) -> list:
    rewards = []
    for comp, pid in zip(completions, pr_id):
        try:
            raw    = comp[0]["content"] if isinstance(comp, list) and isinstance(comp[0], dict) else str(comp)
            action = parse_action(comp)
            truth  = PR_TRUTH.get(str(pid))
            if not truth or not action:
                rewards.append(0.001)
                continue
            base     = score_action(action, truth)
            div_pen  = _diversity_penalty(action)
            cont_pen = _contradiction_penalty(action)
            fmt_bon  = _format_quality_bonus(raw)
            rewards.append(float(np.clip(base + div_pen + cont_pen + fmt_bon, 0.001, 0.999)))
        except Exception:
            rewards.append(0.001)
    return rewards

def format_reward(completions, **kwargs) -> list:
    rewards = []
    for comp in completions:
        try:
            raw  = comp[0]["content"] if isinstance(comp, list) and isinstance(comp[0], dict) else str(comp)
            text = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            m    = re.search(r"\{.*\}", text, re.DOTALL)
            rewards.append(0.2 if m and isinstance(json.loads(m.group()), dict) else 0.0)
        except:
            rewards.append(0.0)
    return rewards

# ── Curriculum dataset builder ────────────────────────────────────────────────
def build_curriculum_dataset(base_dataset):
    """easy×3 → medium×2 → hard×1  (curriculum warm-up ordering)."""
    easy   = base_dataset.filter(lambda x: x.get("task_level") == "easy")
    medium = base_dataset.filter(lambda x: x.get("task_level") == "medium")
    hard   = base_dataset.filter(lambda x: x.get("task_level") == "hard")
    print(f"Curriculum split — easy:{len(easy)} medium:{len(medium)} hard:{len(hard)}")
    curriculum = concatenate_datasets([easy, easy, easy, medium, medium, hard])
    print(f"Curriculum total rows: {len(curriculum)}")
    return curriculum

# ── Post-training evaluator ───────────────────────────────────────────────────
def evaluate_and_save(model, tokenizer, dataset):
    import json
    print("\n--- Post-Training Evaluation ---")
    stats = {"easy": [], "medium": [], "hard": []}
    FastLanguageModel.for_inference(model)

    for item in dataset:
        level = item.get("task_level", "easy")
        if len(stats[level]) >= 20:
            if all(len(v) >= 20 for v in stats.values()):
                break
            continue
        try:
            inputs = tokenizer.apply_chat_template(
                item["prompt"], tokenize=True,
                add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
            outputs   = model.generate(inputs, max_new_tokens=256, use_cache=True)
            new_tokens = outputs[0][inputs.shape[1]:]
            resp      = tokenizer.decode(new_tokens, skip_special_tokens=True)
            action    = parse_action(resp)
            truth     = PR_TRUTH.get(item["pr_id"])
            if truth and action:
                stats[level].append(score_action(action, truth))
        except Exception:
            pass

    # Compute and write summary
    by_difficulty = {}
    all_vals = []
    for level in ["easy", "medium", "hard"]:
        arr = stats[level]
        all_vals.extend(arr)
        by_difficulty[level] = {
            "mean": round(float(np.mean(arr)), 4) if arr else 0.0,
            "std":  round(float(np.std(arr)),  4) if arr else 0.0,
            "count": len(arr),
        }

    summary = {
        "overall": {
            "mean":  round(float(np.mean(all_vals)), 4) if all_vals else 0.0,
            "std":   round(float(np.std(all_vals)),  4) if all_vals else 0.0,
            "count": len(all_vals),
        },
        "by_difficulty": by_difficulty,
    }

    out_path = "evaluation/post_training/trained_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved post-training summary to {out_path}")
    print(json.dumps(summary, indent=2))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    print(f"Loading v1 adapter from: {V1_ADAPTER}")
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    v1_dir = snapshot_download(V1_ADAPTER)
    model  = PeftModel.from_pretrained(model, v1_dir, is_trainable=True)
    model.print_trainable_parameters()

    print("Loading and building curriculum dataset...")
    raw_dataset        = load_dataset("rsd-06/pr-regression-audit-grpo", split="train")
    curriculum_dataset = build_curriculum_dataset(raw_dataset)
    print(f"  Columns: {curriculum_dataset.column_names}")

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=4,
        max_prompt_length=1600,
        max_completion_length=400,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,   # 2 instead of 4 → more updates on large GPU
        max_steps=600,                   # 600 steps vs 400 in v1
        learning_rate=2e-6,              # lower LR for fine-tuning on top of trained adapter
        logging_steps=5,
        save_steps=100,
        warmup_steps=15,
        temperature=0.85,                # slightly less random than v1 (model is already trained)
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

    print(f"\nPushing v2 adapter to {OUT_ADAPTER} ...")
    model.push_to_hub(OUT_ADAPTER)
    tokenizer.push_to_hub(OUT_ADAPTER)

    # Write post-training stats to file
    try:
        evaluate_and_save(model, tokenizer, curriculum_dataset)
    except Exception as e:
        print(f"Post-training evaluation failed: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
