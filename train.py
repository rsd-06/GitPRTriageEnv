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

import os, json, re, requests
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Load ground truth ONCE at module level
url = 'https://huggingface.co/spaces/rsd-06/PRRegressionAuditEnv/resolve/main/prevaluation_env/data/prs.json'
resp = requests.get(url)
PR_TRUTH = {pr["id"]: pr for pr in resp.json()}

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 2048  # increase from 1400
OUTPUT_DIR = "evaluation/checkpoints/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_action(raw) -> dict:
    # --- CRITICAL FIX for TRL >= 0.12.0 ---
    # TRL passes the completion as a list of dicts (the chat template) 
    # rather than a raw string. We must extract the content to avoid TypeError!
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict) and "content" in raw[0]:
            raw = raw[0]["content"]
        elif len(raw) > 0:
            raw = raw[0]
        else:
            raw = ""
    if not isinstance(raw, str):
        raw = str(raw)
    # --------------------------------------
        
    text = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {}

def score_action(action: dict, truth: dict) -> float:
    level = truth["task_level"]
    score = 0.0
    decision = str(action.get("review_decision") or "").lower()
    true_decision = truth["true_decision"]

    if level == "easy":
        if decision == true_decision:
            score += 0.55
        blocker = str(action.get("blocker_type") or "").lower().strip()
        true_blocker = truth.get("true_blocker_type")
        if true_blocker is None:
            if not blocker:
                score += 0.45
        elif blocker == true_blocker.lower():
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
                diff = abs(int(fl) - tfl)
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
                diff = abs(int(fl) - tfl)
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

    return min(max(score, 0.001), 0.999)

def env_reward(completions, pr_id, **kwargs):
    """pr_id is passed automatically by TRL from dataset column."""
    rewards = []
    for comp, pid in zip(completions, pr_id):
        truth = PR_TRUTH.get(pid)
        if not truth:
            rewards.append(0.001)
            continue
        action = parse_action(comp)
        if not action:
            rewards.append(0.001)
            continue
        rewards.append(score_action(action, truth))
    return rewards

def format_reward(completions, **kwargs):
    """Separate reward just for valid JSON with correct decision field."""
    rewards = []
    for comp in completions:
        action = parse_action(comp)
        if action and action.get("review_decision") in ("approve", "request_changes"):
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards

def evaluate_for_ui(model, tokenizer, dataset):
    import numpy as np
    print("\n--- Evaluating Post-Training Stats for UI Dashboard ---")
    stats = {"easy": [], "medium": [], "hard": []}
    FastLanguageModel.for_inference(model)
    
    # Evaluate up to 20 samples per difficulty
    for item in dataset:
        level = item["task_level"]
        if len(stats[level]) >= 20:
            if all(len(v) >= 20 for v in stats.values()):
                break
            continue
        
        try:
            inputs = tokenizer.apply_chat_template(item["prompt"], tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
            outputs = model.generate(inputs, max_new_tokens=256, use_cache=True)
            new_tokens = outputs[0][inputs.shape[1]:]
            resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            action = parse_action(resp)
            truth = PR_TRUTH.get(item["pr_id"])
            if truth and action:
                reward = score_action(action, truth)
                stats[level].append(reward)
        except Exception:
            pass
            
    # Format JSON
    output_json = {}
    for level in ["easy", "medium", "hard"]:
        arr = stats[level]
        n = len(arr)
        avg = float(np.mean(arr)) if n > 0 else 0.0
        std = float(np.std(arr)) if n > 0 else 0.0
        output_json[level] = {"avg": round(avg, 2), "std": round(std, 2), "n": n}
        
    print("\nPASTE THIS JSON INTO THE REACT UI (POST_TRAINING):")
    print(json.dumps(output_json, indent=2))
    print("--------------------------------------------------\n")

def main():
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = load_dataset("rsd-06/pr-regression-audit-grpo", split="train")
    print(f"  {len(dataset)} samples")
    print("  Columns:", dataset.column_names)  # confirm pr_id is present

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=4,
        max_prompt_length=1600,      # increased to allow 2048 - 400 safely
        max_completion_length=400,   # give model more room
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=400,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=100,
        warmup_steps=20,
        temperature=0.9,             # need diversity for GRPO to work
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env_reward, format_reward],
        args=config,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained("evaluation/checkpoints/final_adapter/")
    tokenizer.save_pretrained("evaluation/checkpoints/final_adapter/")
    model.push_to_hub("rsd-06/pr-regression-audit-grpo-adapter")
    tokenizer.push_to_hub("rsd-06/pr-regression-audit-grpo-adapter")
    
    # Evaluate and print JSON stats for the React UI
    try:
        evaluate_for_ui(model, tokenizer, dataset)
    except Exception as e:
        print(f"Failed to generate UI evaluation stats: {e}")
        
    print("Done!")

if __name__ == "__main__":
    main()
