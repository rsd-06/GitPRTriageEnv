from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.environment import DevTriageEnvironment
from models import TriageAction

app = FastAPI(title="GitPRTriage Env")

@app.get("/")
def root():
    # Provide a friendly landing message instead of FastAPI's default 404
    return {"message": "GitPRTriage Env API is live! Navigate to /docs to use the interactive Swagger UI.", "status": "healthy"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = DevTriageEnvironment()

@app.get("/health")
def health():
    return {"status": "healthy", "issues_loaded": len(env.all_issues)}

@app.post("/reset")
def reset():
    return env.reset().model_dump()

@app.post("/step")
def step(action: TriageAction):
    return env.step(action).model_dump()

@app.get("/state")
def state():
    return env.state.model_dump()

@app.get("/tasks")
def tasks():
    return [
        {
            "id": "task_easy",
            "name": "Issue Classification",
            "difficulty": "easy",
            "description": "Classify issue as bug, feature, or duplicate. Score 0 or 1."
        },
        {
            "id": "task_medium",
            "name": "Bug Line Identification",
            "difficulty": "medium",
            "description": "Classify + find exact bug line. Partial credit for proximity."
        },
        {
            "id": "task_hard",
            "name": "Full Triage + Fix Suggestion",
            "difficulty": "hard",
            "description": "Classify + bug line + team routing + keyword-checked fix."
        }
    ]

# ---------------------------------------------------------------------------
# Curriculum + Multi-agent monitoring endpoints
# ---------------------------------------------------------------------------

@app.get("/curriculum")
def curriculum_stats():
    """Live curriculum progression -- shows current phase and agent performance.

    Returns the curriculum sampler's full state so you can monitor:
    - Which phase the curriculum is in (bootstrap/intermediate/advanced)
    - Current sampling weights (how often each difficulty is served)
    - Recent average reward per difficulty level
    - History of phase transitions with the episode they occurred

    Use this during RL training to verify the curriculum is progressing.
    """
    return env.get_curriculum_stats()


@app.get("/audit")
def audit_log(n: int = 20):
    """Recent episode audit log for reward-hacking detection.

    Returns the last n episode records. Each record contains:
    - level: difficulty of the issue served (easy/medium/hard)
    - reward: score received (0.001 to 0.999)
    - episode: episode number

    Monitor this for suspicious reward patterns (e.g. always 0.999
    on hard issues from episode 1 -- that's reward hacking).

    Args:
        n: Number of recent episodes to return (default 20, max 50)
    """
    return env.get_recent_audit(n)


@app.get("/agents/info")
def agents_info():
    """Multi-agent architecture description for the triage system.

    Documents the 4-agent pipeline that produces triage actions:
    1. ClassifierAgent   -> bug / feature / duplicate
    2. BugLocatorAgent   -> exact bug line number
    3. TeamRouterAgent   -> webdev / devops / aiml
    4. FixSuggesterAgent -> concrete one-sentence fix

    Each agent has a focused system prompt and keyword fallback.
    The orchestrator passes upstream results as context to downstream agents.
    """
    return {
        "architecture": "multi-agent-sequential",
        "description": (
            "Four specialist LLM agents triage issues in dependency order. "
            "Each agent receives only the information relevant to its sub-task. "
            "Upstream results are injected as context for downstream agents."
        ),
        "pipeline": [
            {
                "step": 1,
                "agent": "ClassifierAgent",
                "input_fields": ["title", "body", "existing_labels"],
                "output_field": "classification",
                "valid_values": ["bug", "feature", "duplicate"],
                "reward_weight": {"easy": 1.0, "medium": 0.40, "hard": 0.25},
            },
            {
                "step": 2,
                "agent": "BugLocatorAgent",
                "input_fields": ["body", "code_snippet", "classification_context"],
                "output_field": "bug_line",
                "valid_values": "positive integer or null",
                "reward_weight": {"easy": 0.0, "medium": 0.60, "hard": 0.25},
            },
            {
                "step": 3,
                "agent": "TeamRouterAgent",
                "input_fields": ["title", "body", "existing_labels", "classification_context"],
                "output_field": "team",
                "valid_values": ["webdev", "devops", "aiml", "null"],
                "reward_weight": {"easy": 0.0, "medium": 0.0, "hard": 0.25},
            },
            {
                "step": 4,
                "agent": "FixSuggesterAgent",
                "input_fields": ["body", "code_snippet", "bug_line_context", "team_context"],
                "output_field": "suggested_fix",
                "valid_values": "string or null",
                "reward_weight": {"easy": 0.0, "medium": 0.0, "hard": 0.25},
            },
        ],
        "anti_reward_hacking": {
            "independent_verifiers": 4,
            "curriculum_phases": ["bootstrap", "intermediate", "advanced"],
            "confidence_thresholding": True,
            "timeout_penalty": True,
            "audit_log": True,
        },
        "curriculum": {
            "phase_weights": {
                "bootstrap":    {"easy": 0.70, "medium": 0.20, "hard": 0.10},
                "intermediate": {"easy": 0.20, "medium": 0.60, "hard": 0.20},
                "advanced":     {"easy": 0.10, "medium": 0.30, "hard": 0.60},
            },
            "transition_triggers": {
                "bootstrap_to_intermediate": "easy avg reward >= 0.80 over last 10 episodes",
                "intermediate_to_advanced":  "medium avg reward >= 0.65 over last 10 episodes",
            },
        },
    }

import uvicorn


def main():
    uvicorn.run('server.app:app', host='0.0.0.0', port=7860)

if __name__ == '__main__':
    main()

