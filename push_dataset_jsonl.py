import sys
import os
import json
from huggingface_hub import HfApi

# Make prevaluation_env importable
sys.path.insert(0, "prevaluation_env")
from inference import SYSTEM_PROMPT, _build_prompt

def build_and_push():
    data_path = os.path.join("prevaluation_env", "data", "prs.json")
    output_path = "grpo_dataset.jsonl"
    repo_id = "rsd-06/pr-regression-audit-grpo"
    
    with open(data_path, "r", encoding="utf-8") as f:
        pr_list = json.load(f)

    print(f"Loaded {len(pr_list)} PRs. Formatting...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for pr in pr_list:
            chat_format = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(pr)}
            ]
            row = {
                "prompt": chat_format,
                "pr_id": pr.get("id"),
                "task_level": pr.get("task_level")
            }
            f.write(json.dumps(row) + "\n")

    print(f"Saved JSONL to {output_path}. Pushing to Hugging Face Hub...")
    
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    
    # Upload the JSONL file
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo="data.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add formatted GRPO dataset"
    )
    
    print(f"\n[SUCCESS] Dataset uploaded to https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    build_and_push()
