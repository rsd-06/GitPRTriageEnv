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

import os
import sys
import json
import re
import requests
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Client & Inference Inlined from prevaluation_env
# ---------------------------------------------------------------------------

url = 'https://huggingface.co/spaces/rsd-06/PRRegressionAuditEnv/resolve/main/prevaluation_env/data/prs.json'
resp = requests.get(url)
PR_TRUTH = {pr["id"]: pr for pr in resp.json()}

def parse_action(raw) -> dict:
    if isinstance(raw, list):
        raw = raw[-1]["content"] if len(raw) > 0 and isinstance(raw[-1], dict) else str(raw)
    elif not isinstance(raw, str):
        raw = str(raw)
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
# Reward Functions
# ---------------------------------------------------------------------------

def compute_env_reward(completions: list[str], pr_ids: list[str], **kwargs) -> list[float]:
    rewards = []
    for comp, pr_id in zip(completions, pr_ids):
        try:
            action = parse_action(comp)
            truth = PR_TRUTH.get(pr_id)
            if not truth:
                rewards.append(0.001)
                continue
            
            level = truth.get("task_level", "easy")
            score = 0.0
            
            decision = action.get("review_decision", "")
            true_decision = truth.get("true_decision", "approve")
            
            if level == "easy":
                if decision == true_decision:
                    score += 0.55
                blocker = (action.get("blocker_type") or "").lower()
                true_blocker = truth.get("true_blocker_type")
                if true_blocker is None and not blocker:
                    score += 0.45
                elif true_blocker and blocker == true_blocker.lower():
                    score += 0.45
                    
            elif level == "medium":
                if decision == "request_changes":
                    score += 0.10
                cat = (action.get("defect_category") or "").lower()
                if cat == (truth.get("true_defect_category") or "").lower():
                    score += 0.40
                fl = action.get("faulty_line")
                tfl = truth.get("true_faulty_line")
                if fl and tfl:
                    try:
                        if int(fl) == tfl: score += 0.35
                        elif abs(int(fl) - tfl) == 1: score += 0.15
                    except: pass
                    
            elif level == "hard":
                if decision == "request_changes":
                    score += 0.05
                cat = (action.get("defect_category") or "").lower()
                if cat == (truth.get("true_defect_category") or "").lower():
                    score += 0.20
                fl = action.get("faulty_line")
                tfl = truth.get("true_faulty_line")
                if fl and tfl:
                    try:
                        if int(fl) == tfl: score += 0.25
                        elif abs(int(fl) - tfl) == 1: score += 0.10
                    except: pass
                team = (action.get("reviewer_team") or "").lower()
                if team == (truth.get("true_reviewer_team") or "").lower():
                    score += 0.25
                suggestion = (action.get("suggested_change") or "").lower()
                keywords = [k.lower() for k in truth.get("true_fix_keywords", [])]
                if suggestion and keywords:
                    matched = sum(1 for k in keywords if k in suggestion)
                    if matched >= 2: score += 0.25
                    elif matched == 1: score += 0.15
                    else: score += 0.05
                    
            rewards.append(min(max(score, 0.001), 0.999))
        except Exception:
            rewards.append(0.001)
    return rewards

def compute_format_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        try:
            if isinstance(comp, list):
                comp = comp[-1]["content"] if len(comp) > 0 and isinstance(comp[-1], dict) else str(comp)
            elif not isinstance(comp, str):
                comp = str(comp)
            text = re.sub(r"```json\s*", "", comp)
            text = re.sub(r"```\s*", "", text).strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(comp)
                
            if isinstance(data, dict):
                decision = data.get("review_decision")
                if decision in ("approve", "request_changes"):
                    rewards.append(0.5)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def compute_reward(completions: list[str], pr_ids: list[str], **kwargs) -> list[float]:
    env_rewards = compute_env_reward(completions, pr_ids, **kwargs)
    fmt_rewards = compute_format_reward(completions, **kwargs)
    final_rewards = []
    for env, fmt in zip(env_rewards, fmt_rewards):
        capped = min(1.0, env + fmt)
        final_rewards.append(capped)
    return final_rewards

# ---------------------------------------------------------------------------
# Main Training Script
# ---------------------------------------------------------------------------

def main():
    MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
    MAX_SEQ_LENGTH = 1400
    LORA_R = 16
    OUTPUT_DIR = "evaluation/checkpoints/"
    FINAL_ADAPTER_DIR = "evaluation/checkpoints/final_adapter/"
    REPO_ID = "rsd-06/pr-regression-audit-grpo-adapter"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)

    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none"
    )
    model.print_trainable_parameters()

    print("Loading dataset from HF Hub...")
    dataset = load_dataset("rsd-06/pr-regression-audit-grpo", split="train")
    print(f"Dataset loaded with {len(dataset)} items.")

    def reward_fn(completions, prompts, pr_id, **kwargs):
        # TRL automatically passes all dataset columns as kwargs, so pr_id is a list of IDs!
        return compute_reward(completions, pr_id, **kwargs)

    def format_reward_fn(completions, **kwargs):
        return compute_format_reward(completions, **kwargs)

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=300,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        max_steps=400,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=50,
        warmup_steps=20,
        report_to="none"
    )

    print("Starting trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn, format_reward_fn],
        args=config,
        train_dataset=dataset
    )

    trainer.train()

    print("Saving model locally...")
    model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    log_history = trainer.state.log_history
    rewards = [log.get("reward", 0) for log in log_history if "reward" in log.keys()]
    last_10 = rewards[-10:] if len(rewards) >= 10 else rewards
    if last_10:
        mean_val = sum(last_10) / len(last_10)
        print(f"Final mean reward (last {len(last_10)} entries): {mean_val:.4f}")
    
    print(f"Pushing adapter to Hugging Face Hub ({REPO_ID})...")
    model.push_to_hub(REPO_ID)
    tokenizer.push_to_hub(REPO_ID)
    
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
