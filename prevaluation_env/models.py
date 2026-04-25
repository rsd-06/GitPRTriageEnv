from pydantic import BaseModel
from typing import Optional, List, Literal


class ReviewAction(BaseModel):
    review_decision: Literal["approve", "request_changes"]
    blocker_type: Optional[Literal[
        "debug_output",
        "hardcoded_secret",
        "do_not_merge_comment",
        "debug_test_bypass",
        "syntax_error"
    ]] = None
    defect_category: Optional[Literal["security", "logic", "performance"]] = None
    faulty_line: Optional[int] = None
    reviewer_team: Optional[Literal["infosec", "devops", "core-frontend", "core-sysdev", "aiml"]] = None
    suggested_change: Optional[str] = None


class ReviewObservation(BaseModel):
    pr_id: str
    title: str
    description: str
    proposed_code: Optional[str] = None
    context_snippet: Optional[str] = None
    labels: List[str] = []
    task_level: Literal["easy", "medium", "hard"]
    done: bool
    reward: Optional[float] = None
    reward_breakdown: Optional[dict] = None


class ReviewState(BaseModel):
    episode_id: str
    step_count: int
    task_level: Literal["easy", "medium", "hard"]
    current_pr_id: str
