import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .curriculum import CurriculumSampler
from .guards import GuardSuite
from ..models import ReviewAction, ReviewObservation, ReviewState


# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------

def _normalize(score: float) -> float:
    """Clamp and normalize a raw [0, 1] score to the strict (0.001, 0.999) range."""
    return min(max(round(score, 3), 0.001), 0.999)


# ---------------------------------------------------------------------------
# Task 1 — PR Safety Gate (Easy)
# Two independent components: review_decision (0.55) + blocker_type (0.45)
# Prevents always-request_changes exploit: wrong blocker_type on clean PRs = -0.45
# ---------------------------------------------------------------------------

def grade_easy(action: dict, truth: dict) -> Tuple[float, dict]:
    decision = (action.get("review_decision") or "").lower().strip()
    blocker = (action.get("blocker_type") or "").lower().strip() or None
    true_decision = (truth.get("true_decision") or "").lower().strip()
    true_blocker = truth.get("true_blocker_type")

    breakdown = {"review_decision": 0.0, "blocker_type": 0.0}
    score = 0.0

    if decision == true_decision:
        score += 0.55
        breakdown["review_decision"] = 0.55

    # blocker_type must be null for clean PRs and correct string for flagged ones
    if true_blocker is None:
        # Clean PR — blocker_type must be null/empty
        if not blocker:
            score += 0.45
            breakdown["blocker_type"] = 0.45
    else:
        # Flagged PR — blocker_type must match exactly
        if blocker == true_blocker.lower():
            score += 0.45
            breakdown["blocker_type"] = 0.45

    return _normalize(score), breakdown


# ---------------------------------------------------------------------------
# Task 2 — Regression Localization (Medium)
# Static defect analysis: agent reads proposed_code only.
# Components: review_decision (0.10) + defect_category (0.40) + faulty_line (0.35 / 0.15 proximity)
# ---------------------------------------------------------------------------

def grade_medium(action: dict, truth: dict) -> Tuple[float, dict]:
    decision = (action.get("review_decision") or "").lower().strip()
    category = (action.get("defect_category") or "").lower().strip()
    faulty_line = action.get("faulty_line")
    true_category = (truth.get("true_defect_category") or "").lower().strip()
    true_line = truth.get("true_faulty_line")

    breakdown = {"review_decision": 0.0, "defect_category": 0.0, "faulty_line": 0.0}
    score = 0.0

    # All medium PRs are flawed — must be request_changes
    if decision == "request_changes":
        score += 0.10
        breakdown["review_decision"] = 0.10

    if category == true_category:
        score += 0.40
        breakdown["defect_category"] = 0.40

    if true_line is not None and faulty_line is not None:
        try:
            line_int = int(faulty_line)
            if line_int == true_line:
                score += 0.35
                breakdown["faulty_line"] = 0.35
            elif abs(line_int - true_line) == 1:
                score += 0.15
                breakdown["faulty_line"] = 0.15
        except (TypeError, ValueError):
            pass

    return _normalize(score), breakdown


# ---------------------------------------------------------------------------
# Task 3 — Full Audit & Integration Review (Hard)
# Integration defect analysis: agent reads proposed_code + context_snippet.
# Components: review_decision (0.05) + defect_category (0.20) +
#             faulty_line (0.25 / 0.10 proximity) + reviewer_team (0.25) +
#             suggested_change (0.25, keyword-scored, anti-stuffing guard)
# ---------------------------------------------------------------------------

def grade_hard(action: dict, truth: dict) -> Tuple[float, dict]:
    decision = (action.get("review_decision") or "").lower().strip()
    category = (action.get("defect_category") or "").lower().strip()
    faulty_line = action.get("faulty_line")
    team = (action.get("reviewer_team") or "").lower().strip()
    suggested = (action.get("suggested_change") or "").lower().strip()
    true_category = (truth.get("true_defect_category") or "").lower().strip()
    true_line = truth.get("true_faulty_line")
    true_team = (truth.get("true_reviewer_team") or "").lower().strip()
    fix_keywords = [k.lower() for k in truth.get("true_fix_keywords", [])]

    breakdown = {
        "review_decision": 0.0,
        "defect_category": 0.0,
        "faulty_line": 0.0,
        "reviewer_team": 0.0,
        "suggested_change": 0.0,
    }
    score = 0.0

    # All hard PRs are flawed — must be request_changes
    if decision == "request_changes":
        score += 0.05
        breakdown["review_decision"] = 0.05

    if category == true_category:
        score += 0.20
        breakdown["defect_category"] = 0.20

    if true_line is not None and faulty_line is not None:
        try:
            line_int = int(faulty_line)
            if line_int == true_line:
                score += 0.25
                breakdown["faulty_line"] = 0.25
            elif abs(line_int - true_line) == 1:
                score += 0.10
                breakdown["faulty_line"] = 0.10
        except (TypeError, ValueError):
            pass

    if team == true_team:
        score += 0.25
        breakdown["reviewer_team"] = 0.25

    # Anti-reward-hacking: suggested_change >200 chars → 0.0 (no keyword stuffing)
    if suggested:
        if len(suggested) > 200:
            fix_score = 0.0
        elif fix_keywords:
            matched = sum(1 for kw in fix_keywords if kw in suggested)
            if matched >= 2:
                fix_score = 0.25
            elif matched == 1:
                fix_score = 0.15
            else:
                fix_score = 0.05  # effort credit
        else:
            fix_score = 0.05
        score += fix_score
        breakdown["suggested_change"] = fix_score

    return _normalize(score), breakdown


def grade(action: dict, truth: dict) -> Tuple[float, dict]:
    level = truth.get("task_level", "easy")
    if level == "easy":
        return grade_easy(action, truth)
    elif level == "medium":
        return grade_medium(action, truth)
    elif level == "hard":
        return grade_hard(action, truth)
    raise ValueError(f"Unknown task_level: {level}")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PRRegressionAuditEnvironment:
    def __init__(self):
        data_candidates = [
            Path(__file__).resolve().parents[1] / "data" / "prs.json",
            Path("data/prs.json"),
            Path("prevaluation_env/data/prs.json"),
        ]
        data_path = next((p for p in data_candidates if p.exists()), None)

        if data_path is None:
            print("Warning: prs.json not found. Environment will have no PRs.")
            self.all_prs = []
        else:
            with data_path.open("r", encoding="utf-8") as f:
                self.all_prs = json.load(f)

        # Group PRs by difficulty for curriculum sampler
        self._prs_by_level: Dict[str, List[dict]] = {"easy": [], "medium": [], "hard": []}
        for pr in self.all_prs:
            lvl = pr.get("task_level", "easy")
            if lvl in self._prs_by_level:
                self._prs_by_level[lvl].append(pr)

        self._curriculum = CurriculumSampler(self._prs_by_level, history_window=10)

        # Anti-reward-hacking guard suite (ported from GitHub)
        self._guards = GuardSuite()

        self._reset_state()

    def _reset_state(self):
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self._current = self._curriculum.sample()
        self._step_start: float = time.perf_counter()

    def reset(self) -> ReviewObservation:
        self._reset_state()
        self._step_start = time.perf_counter()  # start timing when agent receives obs
        return self._build_observation(done=False, reward=None, breakdown=None)

    def step(self, action: ReviewAction) -> ReviewObservation:
        self.step_count += 1
        elapsed_ms = (time.perf_counter() - self._step_start) * 1000.0
        raw_reward, breakdown = grade(action.model_dump(), self._current)
        # Apply anti-reward-hacking guards (multiplicative penalties)
        reward, _guard_results = self._guards.evaluate(
            action.model_dump(), self._current, raw_reward, elapsed_ms
        )
        self._curriculum.record(self._current["task_level"], reward)
        return self._build_observation(done=True, reward=reward, breakdown=breakdown)

    def _build_observation(
        self,
        done: bool,
        reward: Any,
        breakdown: Any,
    ) -> ReviewObservation:
        return ReviewObservation(
            pr_id=self._current["id"],
            title=self._current["title"],
            description=self._current["description"],
            proposed_code=self._current.get("proposed_code"),
            context_snippet=self._current.get("context_snippet"),
            labels=self._current.get("labels", []),
            task_level=self._current["task_level"],
            done=done,
            reward=reward,
            reward_breakdown=breakdown,
        )

    @property
    def state(self) -> ReviewState:
        return ReviewState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            task_level=self._current["task_level"],
            current_pr_id=self._current["id"],
        )

    def get_curriculum_stats(self) -> dict:
        """Returns live curriculum progression stats. Used by /curriculum endpoint."""
        return self._curriculum.get_stats()

    def get_recent_audit(self, n: int = 20) -> list:
        """Returns the last n episode records for reward-hacking detection."""
        n = min(n, 50)
        return list(self._curriculum._history)[-n:]

    def get_guard_summary(self) -> dict:
        """Returns anti-reward-hacking guard statistics.

        Exposes the GuardSuite's summary including total penalties applied,
        penalty rate, fast response rate, and descriptions of each guard.
        Used by the /guards FastAPI endpoint.
        """
        return self._guards.get_summary()

    def get_guard_audit(self, n: int = 20) -> list:
        """Returns recent guard firing log entries.

        Each entry contains episode number, original vs adjusted reward,
        penalty multiplier, and which guards triggered and why.
        Used by the /guards/audit FastAPI endpoint.

        Args:
            n: Number of recent entries to return (default 20)
        """
        return self._guards.get_audit_log(n)
