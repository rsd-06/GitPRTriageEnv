from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from .environment import PRRegressionAuditEnvironment
from ..models import ReviewAction

app = FastAPI(title="PRRegressionAuditEnv")

assets_dir = os.path.join(os.path.dirname(__file__), "../../assets")
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = PRRegressionAuditEnvironment()


@app.get("/", response_class=HTMLResponse)
def root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PR Review RL Agent Dashboard</h1><p>Dashboard HTML not found.</p>"


@app.get("/health")
def health():
    return {"status": "healthy", "prs_loaded": len(env.all_prs)}


@app.post("/reset")
def reset():
    return env.reset().model_dump()


from pydantic import BaseModel

class StatelessGradeRequest(BaseModel):
    pr_id: str
    action: dict
    elapsed_ms: float = 150.0

@app.post("/grade_stateless")
def grade_stateless(req: StatelessGradeRequest):
    result = env.grade_stateless(req.pr_id, req.action, req.elapsed_ms)
    if "error" in result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/step")
def step(action: ReviewAction):
    return env.step(action).model_dump()


@app.get("/state")
def state():
    return env.state.model_dump()


@app.get("/tasks")
def tasks():
    return [
        {
            "id": "task_easy",
            "name": "PR Safety Gate",
            "difficulty": "easy",
            "description": (
                "Approve clean PRs or request_changes on obviously flagged ones. "
                "Two-component reward: review_decision (0.55) + blocker_type (0.45)."
            ),
        },
        {
            "id": "task_medium",
            "name": "Regression Localization",
            "difficulty": "medium",
            "description": (
                "Identify the defect category and exact faulty line in the proposed code. "
                "Static analysis — defect is self-contained in proposed_code. "
                "Partial credit for proximity (±1 line)."
            ),
        },
        {
            "id": "task_hard",
            "name": "Full Audit & Integration Review",
            "difficulty": "hard",
            "description": (
                "Integration analysis — proposed_code looks fine in isolation, but a "
                "context_snippet reveals a compatibility break with the existing system. "
                "Five-component reward: decision, defect category, faulty line, "
                "reviewer team routing, and a concise suggested change."
            ),
        },
    ]


@app.get("/curriculum")
def curriculum_stats():
    """Live curriculum progression — shows current phase and agent performance.

    Returns the curriculum sampler's full state including:
    - Current phase (bootstrap / intermediate / advanced)
    - Current sampling weights per difficulty
    - Recent average reward per difficulty level
    - History of phase transitions with episode numbers

    Use during RL training to verify curriculum is progressing.
    """
    return env.get_curriculum_stats()


@app.get("/audit")
def audit_log(n: int = 10000):
    """Recent episode audit log for reward-hacking detection.

    Returns the last n episode records. Each record contains:
    - level: difficulty of the PR served (easy/medium/hard)
    - reward: scalar score received (0.001 to 0.999)
    - episode: episode number

    Watch for suspicious patterns (e.g. always 0.999 on hard from episode 1).

    Args:
        n: Number of recent episodes to return (default 20, max 50)
    """
    return env.get_recent_audit(n)


@app.get("/agents/info")
def agents_info():
    """Multi-agent architecture description for the PR review system.

    Documents the 4-agent sequential pipeline:
    1. SafetyGateAgent    -> review_decision + blocker_type (Easy)
    2. DefectLocatorAgent -> defect_category + faulty_line (Medium + Hard)
    3. ReviewerRouterAgent-> reviewer_team (Hard only)
    4. ReviewCommentAgent -> suggested_change (Hard only)

    Each agent has a focused system prompt and keyword fallback.
    Upstream results are injected as context into downstream agents.
    """
    return {
        "architecture": "multi-agent-sequential",
        "description": (
            "Four specialist LLM agents review PRs in dependency order. "
            "Each agent receives only the information relevant to its sub-task. "
            "Upstream results are injected as context for downstream agents."
        ),
        "pipeline": [
            {
                "step": 1,
                "agent": "SafetyGateAgent",
                "input_fields": ["title", "description", "proposed_code", "labels"],
                "output_fields": ["review_decision", "blocker_type"],
                "reward_weight": {"easy": 1.0, "medium": 0.10, "hard": 0.05},
            },
            {
                "step": 2,
                "agent": "DefectLocatorAgent",
                "input_fields": ["description", "proposed_code", "decision_context"],
                "output_fields": ["defect_category", "faulty_line"],
                "reward_weight": {"easy": 0.0, "medium": 0.75, "hard": 0.45},
            },
            {
                "step": 3,
                "agent": "ReviewerRouterAgent",
                "input_fields": ["title", "proposed_code", "context_snippet", "defect_context"],
                "output_fields": ["reviewer_team"],
                "reward_weight": {"easy": 0.0, "medium": 0.0, "hard": 0.25},
            },
            {
                "step": 4,
                "agent": "ReviewCommentAgent",
                "input_fields": [
                    "description", "proposed_code", "context_snippet",
                    "faulty_line_context", "reviewer_team_context"
                ],
                "output_fields": ["suggested_change"],
                "reward_weight": {"easy": 0.0, "medium": 0.0, "hard": 0.25},
            },
        ],
        "anti_reward_hacking": {
            "blocker_type_required_for_clean_prs": "null — prevents always-request_changes exploit",
            "suggested_change_max_chars": 200,
            "curriculum_phases": ["bootstrap", "intermediate", "advanced"],
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


@app.get("/guards")
def guard_summary():
    """Anti-reward-hacking guard suite statistics.

    Returns a summary of all reward-hacking detection activity:
    - total_episodes: how many episodes have been guard-evaluated
    - total_penalties_applied: how many times any guard fired
    - penalty_rate: fraction of episodes where reward was adjusted
    - fast_response_rate: fraction of responses under 200ms threshold
    - guards: description of each active guard

    A rising penalty_rate during RL training is a healthy sign —
    it means the guards are catching exploitation attempts.
    """
    return env.get_guard_summary()


@app.get("/guards/audit")
def guard_audit(n: int = 20):
    """Recent guard firing log — shows reward adjustments in detail.

    Each entry shows:
    - episode: episode number when guard fired
    - original_reward: reward before guard penalties
    - adjusted_reward: reward after guard penalties
    - penalty_multiplier: the combined penalty factor applied
    - guards_triggered: list of guard names that fired
    - reasons: human-readable explanation for each guard firing

    Example of keyword stuffing being caught:
      original_reward: 0.999 -> adjusted_reward: 0.499
      guard: KeywordStuffingDetector
      reason: suggested_change keyword density 0.67 exceeds 0.40 threshold

    Args:
        n: Number of recent entries (default 20)
    """
    return env.get_guard_audit(n)


def main():
    uvicorn.run("prevaluation_env.server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
