from pydantic import BaseModel
from typing import Optional, List, Literal

class TriageAction(BaseModel):
    classification: Literal["bug", "feature", "duplicate"]
    bug_line: Optional[int] = None
    team: Optional[Literal["webdev", "devops", "aiml"]] = None
    suggested_fix: Optional[str] = None

class TriageObservation(BaseModel):
    issue_id: str
    title: str
    body: str
    code_snippet: Optional[str] = None
    existing_labels: List[str] = []
    task_level: Literal["easy", "medium", "hard"]
    done: bool
    reward: Optional[float] = None

class TriageState(BaseModel):
    episode_id: str
    step_count: int
    task_level: Literal["easy", "medium", "hard"]
    current_issue_id: str
