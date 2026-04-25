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

class DevTriageClient:
    def __init__(self, base_url: str = None):
        if base_url is None:
            base_url = os.environ.get("ENV_URL", "http://localhost:7860")
        self.base_url = base_url

    def reset(self) -> dict:
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()

    def step(self, action: dict) -> dict:
        response = requests.post(f"{self.base_url}/step", json=action)
        response.raise_for_status()
        return response.json()

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
# Reward Functions
# ---------------------------------------------------------------------------

def compute_env_reward(completions: list[str], pr_ids: list[str], **kwargs) -> list[float]:
    rewards = []
    for comp, pr_id in zip(completions, pr_ids):
        try:
            action_dict = parse_action(comp)
            if not action_dict:
                rewards.append(0.001)
                continue
                
            client = DevTriageClient()
            matched = False
            for _ in range(20):
                obs = client.reset()
                obs_id = obs.get("pr_id") or obs.get("issue_id")
                if obs_id == pr_id:
                    matched = True
                    break
            
            if not matched:
                rewards.append(0.001)
                continue
            
            result = client.step(action_dict)
            reward_val = float(result.get("reward", 0.001))
            rewards.append(reward_val)
        except Exception as e:
            rewards.append(0.001)
    return rewards

def compute_format_reward(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        try:
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
