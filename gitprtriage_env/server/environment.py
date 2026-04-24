import json
import random
import uuid
from typing import Any, Dict, List

from server.curriculum import CurriculumSampler

from models import TriageAction, TriageObservation, TriageState

def grade_easy(action: dict, truth: dict) -> float:
    classification = (action.get("classification") or "").lower().strip()
    true_label = (truth.get("true_label") or "").lower().strip()
    if not classification:
        return 0.001
    return 0.999 if classification == true_label else 0.001

def grade_medium(action: dict, truth: dict) -> float:
    classification = (action.get("classification") or "").lower().strip()
    true_label = (truth.get("true_label") or "").lower().strip()
    bug_line = action.get("bug_line")
    true_bug_line = truth.get("true_bug_line")

    score = 0.0
    if classification == true_label:
        score += 0.40

    if true_bug_line is not None and bug_line is not None:
        try:
            bug_line_int = int(bug_line)
            if bug_line_int == true_bug_line:
                score += 0.40
            elif abs(bug_line_int - true_bug_line) == 1:
                score += 0.20  # proximity bonus
        except (TypeError, ValueError):
            pass

    return min(max(round(score, 3), 0.001), 0.999)

def grade_hard(action: dict, truth: dict) -> float:
    classification = (action.get("classification") or "").lower().strip()
    true_label = (truth.get("true_label") or "").lower().strip()
    bug_line = action.get("bug_line")
    true_bug_line = truth.get("true_bug_line")
    team = (action.get("team") or "").lower().strip()
    true_team = (truth.get("true_team") or "").lower().strip()
    suggested_fix = (action.get("suggested_fix") or "").lower().strip()
    fix_keywords = [k.lower() for k in truth.get("true_fix_keywords", [])]

    score = 0.0

    if classification == true_label:
        score += 0.25

    if true_bug_line is not None and bug_line is not None:
        try:
            bug_line_int = int(bug_line)
            if bug_line_int == true_bug_line:
                score += 0.25
            elif abs(bug_line_int - true_bug_line) == 1:
                score += 0.10
        except (TypeError, ValueError):
            pass

    if team == true_team:
        score += 0.25

    if suggested_fix:
        if fix_keywords:
            matched = sum(1 for kw in fix_keywords if kw in suggested_fix)
            if matched >= 2:
                score += 0.25
            elif matched == 1:
                score += 0.15
            else:
                score += 0.05  # effort credit only
        else:
            score += 0.05

    return min(max(round(score, 3), 0.001), 0.999)

def grade(action: dict, truth: dict) -> float:
    level = truth.get("task_level", "easy")
    if level == "easy":
        return grade_easy(action, truth)
    elif level == "medium":
        return grade_medium(action, truth)
    elif level == "hard":
        return grade_hard(action, truth)
    raise ValueError(f"Unknown task_level: {level}")

class DevTriageEnvironment:
    def __init__(self):
        with open("data/issues.json", "r") as f:
            self.all_issues = json.load(f)

        # Group issues by difficulty level for the curriculum sampler
        self._issues_by_level: Dict[str, List[dict]] = {"easy": [], "medium": [], "hard": []}
        for issue in self.all_issues:
            lvl = issue.get("task_level", "easy")
            if lvl in self._issues_by_level:
                self._issues_by_level[lvl].append(issue)

        # Curriculum sampler replaces random sampling
        self._curriculum = CurriculumSampler(self._issues_by_level, history_window=10)

        self._reset_state()

    def _reset_state(self):
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self._current = self._curriculum.sample()

    def reset(self) -> TriageObservation:
        self._reset_state()
        return TriageObservation(
            issue_id=self._current["id"],
            title=self._current["title"],
            body=self._current["body"],
            code_snippet=self._current["code_snippet"],
            existing_labels=self._current["labels"],
            task_level=self._current["task_level"],
            done=False,
            reward=None
        )

    def step(self, action: TriageAction) -> TriageObservation:
        self.step_count += 1
        reward = grade(action.model_dump(), self._current)
        # Record performance for curriculum phase tracking
        self._curriculum.record(self._current["task_level"], reward)
        return TriageObservation(
            issue_id=self._current["id"],
            title=self._current["title"],
            body=self._current["body"],
            code_snippet=self._current["code_snippet"],
            existing_labels=self._current["labels"],
            task_level=self._current["task_level"],
            done=True,
            reward=reward
        )

    @property
    def state(self) -> TriageState:
        return TriageState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            task_level=self._current["task_level"],
            current_issue_id=self._current["id"]
        )

    def get_curriculum_stats(self) -> dict:
        """Returns current curriculum progression statistics.

        Used by the /curriculum FastAPI endpoint and for monitoring
        agent improvement during RL training. Returns the curriculum
        sampler's full state including current phase, phase weights,
        recent performance by level, and transition history.

        Returns:
            dict with keys: current_phase, episode, phase_weights,
            phase_transitions, recent_performance, history_window,
            total_episodes_recorded, issue_pool_size
        """
        return self._curriculum.get_stats()

    def get_recent_audit(self, n: int = 20) -> list:
        """Returns the last n episode records from the curriculum history.

        Used by the /audit FastAPI endpoint for reward-hacking detection.
        Each entry contains level, reward, and episode number.

        Args:
            n: Number of recent entries to return (default 20, max 50)

        Returns:
            List of dicts with keys: level, reward, episode
        """
        n = min(n, 50)
        history = list(self._curriculum._history)
        return history[-n:]
