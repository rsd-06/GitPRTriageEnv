import sys
import os
import json
from collections import Counter
from datasets import Dataset

# Make prevaluation_env importable
sys.path.insert(0, "prevaluation_env")
from inference import SYSTEM_PROMPT, _build_prompt

def build_dataset():
    data_path = os.path.join("prevaluation_env", "data", "prs.json")
    output_dir = os.path.join("evaluation", "grpo_dataset")
    card_path = os.path.join(output_dir, "dataset_card.json")

    # 1. Load PR Data
    with open(data_path, "r", encoding="utf-8") as f:
        pr_list = json.load(f)

    # 2. Build Dataset Columns
    prompts = []
    pr_ids = []
    task_levels = []

    for pr in pr_list:
        chat_format = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(pr)}
        ]
        prompts.append(chat_format)
        pr_ids.append(pr.get("id"))
        task_levels.append(pr.get("task_level"))

    # 3. Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "pr_id": pr_ids,
        "task_level": task_levels
    })

    # 4. Save to Disk
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)

    # 5. Build and Save Summary Card
    task_level_counts = dict(Counter(task_levels))
    
    card_info = {
        "total_count": len(dataset),
        "count_by_task_level": task_level_counts,
        "column_names": dataset.column_names
    }

    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card_info, f, indent=2)

    # 6. Print confirmation
    print("\n[SUCCESS] GRPO Dataset Successfully Built!")
    print(f"Total PRs processed: {card_info['total_count']}")
    print(f"Task Breakdown: {card_info['count_by_task_level']}")
    print(f"Columns generated: {card_info['column_names']}")
    print(f"\nSaved HuggingFace Dataset backends to: {output_dir}")
    print(f"Saved summary card to: {card_path}")

if __name__ == "__main__":
    build_dataset()
