import os
import json
import random
import statistics
import requests

from prevaluation_env.client import DevTriageClient

def run_baseline():
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    print(f"Connecting to environment at {env_url}...")
    
    try:
        client = DevTriageClient(base_url=env_url)
        # Quick health check to ensure we can connect
        client.health()
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to environment: {e}")
        print("Please ensure the FastAPI server is running.")
        return

    num_episodes = 150
    episodes_data = []

    print(f"Running {num_episodes} episodes with random actions...")
    
    for i in range(1, num_episodes + 1):
        # 1. Reset environment to get new PR observation
        obs = client.reset()
        
        pr_id = obs.get("pr_id") or obs.get("issue_id", "unknown")
        task_level = obs.get("task_level", "unknown")

        # 2. Construct random action
        action = {
            "review_decision": random.choice(["approve", "request_changes"]),
            "blocker_type": random.choice([
                "debug_output", "hardcoded_secret", "do_not_merge_comment", 
                "debug_test_bypass", "syntax_error", None
            ]),
            "defect_category": random.choice(["security", "logic", "performance", None]),
            "faulty_line": random.choice(list(range(1, 11)) + [None]),
            "reviewer_team": random.choice([
                "infosec", "devops", "core-frontend", "core-sysdev", "aiml", None
            ]),
            "suggested_change": random.choice(["Fix the issue on this line", None])
        }

        # 3. Step the environment
        result = client.step(action)
        reward = result.get("reward", 0.0)
        
        # 4. Record episode
        episodes_data.append({
            "episode_number": i,
            "pr_id": pr_id,
            "task_level": task_level,
            "reward": reward,
            "reward_breakdown": result.get("reward_breakdown")
        })
        
        if i % 10 == 0:
            print(f"Completed {i}/{num_episodes} episodes...", flush=True)

    # 5. Compute statistics
    rewards_by_level = {"easy": [], "medium": [], "hard": []}
    all_rewards = []
    
    for ep in episodes_data:
        lvl = ep["task_level"].lower()
        if lvl in rewards_by_level:
            rewards_by_level[lvl].append(ep["reward"])
        all_rewards.append(ep["reward"])

    def calc_stats(rewards_list):
        if not rewards_list:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        mean_val = statistics.mean(rewards_list)
        std_val = statistics.stdev(rewards_list) if len(rewards_list) > 1 else 0.0
        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "count": len(rewards_list)
        }

    summary = {
        "overall": calc_stats(all_rewards),
        "by_difficulty": {
            "easy": calc_stats(rewards_by_level["easy"]),
            "medium": calc_stats(rewards_by_level["medium"]),
            "hard": calc_stats(rewards_by_level["hard"])
        }
    }

    # 6. Print summary
    print("\n=== Baseline Training Summary ===")
    print(f"Overall Mean Reward: {summary['overall']['mean']:.4f} ± {summary['overall']['std']:.4f} (Count: {summary['overall']['count']})")
    
    for lvl in ["easy", "medium", "hard"]:
        lvl_stats = summary['by_difficulty'][lvl]
        print(f"  - {lvl.capitalize()} Mean Reward: {lvl_stats['mean']:.4f} ± {lvl_stats['std']:.4f} (Count: {lvl_stats['count']})")

    # 7. Save outputs
    # Ensuring evaluation directories exist
    output_dir = os.path.join(os.path.dirname(__file__), "evaluation", "pre_training")
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, "baseline_results.json")
    summary_path = os.path.join(output_dir, "baseline_summary.json")

    with open(results_path, "w") as f:
        json.dump(episodes_data, f, indent=2)
        
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved episode records to: {results_path}")
    print(f"Saved summary statistics to: {summary_path}")

if __name__ == "__main__":
    run_baseline()
