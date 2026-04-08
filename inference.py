"""
inference.py — Baseline LLM Agent for DevTriageEnv
Required env vars:
  API_BASE_URL  - LLM endpoint
  MODEL_NAME    - Model name  
  HF_TOKEN      - API key (works for both HF and Groq)
  ENV_URL       - Environment URL
"""
import os, json, re, statistics, requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")      # reused for Groq key too
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

client = OpenAI(
    base_url=API_BASE_URL, 
    api_key=HF_TOKEN,
    max_retries=3,          # Practical Fix: Limited 3 retries for rate limits
    timeout=30.0            # Practical Fix: 30s timeout for cold starts
)

SYSTEM_PROMPT = """You are a senior software engineer triaging GitHub issues.
Respond with ONLY valid, strict JSON matching this schema:
{
  "type": "object",
  "properties": {
    "thought_process": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Step-by-step reasoning. 1. Analyze the issue. 2. Identify the category. 3. Look for the exact line number of the bug in the code snippet. 4. Determine the internal team. 5. Draft the fix. ALWAYS DO THIS FIRST."
    },
    "classification": {
      "type": "string",
      "enum": ["bug", "feature", "duplicate"],
      "description": "Required. The category of the issue."
    },
    "bug_line": {
      "type": ["integer", "null"],
      "description": "The 1-indexed line number where the bug is located. Set to null if there is no code snippet or no obvious bug."
    },
    "team": {
      "type": ["string", "null"],
      "enum": ["webdev", "devops", "aiml", null],
      "description": "The team best suited to handle this. webdev=frontend/backend/api, devops=infra/ci/docker/k8s, aiml=ml/models. Set to null if unable to determine."
    },
    "suggested_fix": {
      "type": ["string", "null"],
      "description": "One concrete sentence suggesting HOW to fix the bug. Do not just say 'fix the bug'. Set to null if no clear fix."
    }
  },
  "required": ["thought_process", "classification", "bug_line", "team", "suggested_fix"]
}
No markdown formatting, no backticks, no markdown JSON blocks. Output exactly the raw JSON text. Make sure bug_line is exactly an integer or null."""

def parse_action(raw: str) -> dict:
    text = re.sub(r"```json\s*", "", raw)
    text = re.sub(r"```\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"classification": "bug", "bug_line": None,
            "team": None, "suggested_fix": None, "thought_process": []}


def run_episode() -> tuple:
    obs = requests.post(f"{ENV_URL}/reset", timeout=10).json()
    task_level = obs.get("task_level", "easy")
    print(f"[START] task={task_level}", flush=True)

    parts = [f"Title: {obs['title']}", f"Body: {obs['body']}"]
    if obs.get("code_snippet"):
        parts.append(f"\nCode (lines are 1-indexed):\n{obs['code_snippet']}")
    if obs.get("existing_labels"):
        parts.append(f"Labels: {', '.join(obs['existing_labels'])}")
    parts.append(f"\nTask Level: {task_level}\nRespond with ONLY JSON.")
    prompt = "\n".join(parts)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=600,
            temperature=0.1,
        )
        content = response.choices[0].message.content
    except Exception as exc:
        print(f"Model request failed ({exc}). Using fallback action.")
        content = "{}"

    action = parse_action(content)
    action.pop("thought_process", None)  # Clean internal reasoning payload before sending to strict API
    result = requests.post(f"{ENV_URL}/step", json=action, timeout=10).json()
    score = float(result.get("reward") if result.get("reward") is not None else 0.001)
    
    print(f"[STEP] step=1 reward={score}", flush=True)
    print(f"[END] task={task_level} score={score} steps=1", flush=True)
    return score, task_level


def main():
    print(f"Running inference against: {ENV_URL}")
    print(f"Model: {MODEL_NAME}")
    print("-" * 50)

    task_scores = {"easy": [], "medium": [], "hard": []}
    for ep in range(60):
        try:
            score, level = run_episode()
            task_scores[level].append(score)
            print(f"  Ep {ep+1:02d} [{level:6s}] score={score:.3f}")
        except Exception as e:
            print(f"  Ep {ep+1:02d} ERROR: {e}")

    print("\n" + "-" * 50)
    print("BASELINE RESULTS")
    print("-" * 50)
    for level in ["easy", "medium", "hard"]:
        scores = task_scores[level]
        if scores:
            avg = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f"  {level:6s}: {avg:.3f} ± {std:.3f}  (n={len(scores)})")
        else:
            print(f"  {level:6s}: no data")


if __name__ == "__main__":
    main()