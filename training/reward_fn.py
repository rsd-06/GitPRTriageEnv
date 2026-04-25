import sys
import json
import re

from prevaluation_env.client import DevTriageClient
from prevaluation_env.inference import parse_action


def compute_env_reward(completions: list[str], pr_ids: list[str], **kwargs) -> list[float]:
    """Scores completions based on the actual environment reward pipeline."""
    rewards = []
    
    for comp, pr_id in zip(completions, pr_ids):
        try:
            action_dict = parse_action(comp)
            # Failsafe if parsed dict is entirely invalid/empty
            if not action_dict:
                rewards.append(0.001)
                continue
                
            client = DevTriageClient()
            matched = False
            
            # Loop until we find the exact PR observation from the dataset distribution
            for _ in range(20):
                obs = client.reset()
                obs_id = obs.get("pr_id") or obs.get("issue_id")
                if obs_id == pr_id:
                    matched = True
                    break
            
            # If the PR ID was not drawn within 20 samples, fail and penalize
            if not matched:
                rewards.append(0.001)
                continue
            
            # Submit action to the deterministic backend 
            result = client.step(action_dict)
            
            # Ensure float extraction
            reward_val = float(result.get("reward", 0.001))
            rewards.append(reward_val)

        except Exception:
            # Silent fallback to heavy penalization
            rewards.append(0.001)
            
    return rewards


def compute_format_reward(completions: list[str], **kwargs) -> list[float]:
    """Incentivizes structurally valid JSON completion blocks independent of correctness."""
    rewards = []
    
    for comp in completions:
        try:
            # Perform a strict JSON parse attempting to gracefully handle backticks
            text = re.sub(r"```json\s*", "", comp)
            text = re.sub(r"```\s*", "", text).strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(comp)
                
            # Reward exclusively hinges upon parsing as a Dict with at least a valid review decision 
            if isinstance(data, dict):
                decision = data.get("review_decision")
                if decision in ("approve", "request_changes"):
                    rewards.append(0.5)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
                
        except Exception:
            # Never raise formatting errors back to the loss pipeline; penalize via 0.0
            rewards.append(0.0)
            
    return rewards


def compute_reward(completions: list[str], pr_ids: list[str], **kwargs) -> list[float]:
    """Combined reward mapping used directly by GRPO pipeline hooks."""
    env_rewards = compute_env_reward(completions, pr_ids, **kwargs)
    fmt_rewards = compute_format_reward(completions, **kwargs)
    
    final_rewards = []
    # Both pipelines must output synchronized dimensional lengths 
    for env, fmt in zip(env_rewards, fmt_rewards):
        # Env (1.0 max) + format (0.5 max) could mathematically hit 1.5. Cap strictly to 1.0.
        capped = min(1.0, env + fmt)
        final_rewards.append(capped)
        
    return final_rewards
