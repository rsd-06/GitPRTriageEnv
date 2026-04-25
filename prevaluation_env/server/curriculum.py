"""
server/curriculum.py
--------------------
Adaptive curriculum sampler for the PREvaluationEnv RL environment.

The CurriculumSampler maintains a three-phase difficulty progression:

    bootstrap    → heavy easy weighting while the agent learns basics
    intermediate → balanced medium focus once easy is mastered
    advanced     → hard-heavy once medium performance is solid

Phase transitions are driven by observed reward history and are
one-directional (never regress). The sampler is fully self-contained
and does not import any other project modules.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional


class CurriculumSampler:
    """Adaptive difficulty sampler for three-level GitHub issue triage training.

    Serves issue dicts from a fixed pool using probability weights that shift
    automatically as the agent's observed reward climbs through thresholds.

    Phases
    ------
    bootstrap    — 70 % easy / 20 % medium / 10 % hard
    intermediate — 20 % easy / 60 % medium / 20 % hard
    advanced     — 10 % easy / 30 % medium / 60 % hard

    Transitions are evaluated at the *start* of each ``sample()`` call so
    that a newly recorded reward is visible on the very next episode.

    Attributes:
        PHASE_WEIGHTS:          Class-level weight table keyed by phase name.
        TRANSITION_THRESHOLDS:  Class-level thresholds that trigger advancement.
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    PHASE_WEIGHTS: Dict[str, Dict[str, float]] = {
        "bootstrap":    {"easy": 0.70, "medium": 0.20, "hard": 0.10},
        "intermediate": {"easy": 0.20, "medium": 0.60, "hard": 0.20},
        "advanced":     {"easy": 0.10, "medium": 0.30, "hard": 0.60},
    }

    TRANSITION_THRESHOLDS: Dict[str, Dict] = {
        "bootstrap_to_intermediate": {"level": "easy",   "min_reward": 0.80, "min_samples": 5},
        "intermediate_to_advanced":  {"level": "medium", "min_reward": 0.65, "min_samples": 5},
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        issues_by_level: Dict[str, List[dict]],
        history_window: int = 10,
    ) -> None:
        """Initialise the curriculum sampler.

        Args:
            issues_by_level: Dict with keys ``"easy"``, ``"medium"``,
                             ``"hard"``, each mapping to a list of issue dicts.
                             Each issue dict must contain at least ``"id"``
                             (str) and ``"task_level"`` (str).
            history_window:  Number of recent episodes to inspect when
                             evaluating phase transitions. Default is 10.
        """
        self._issues: Dict[str, List[dict]] = issues_by_level
        self._history: deque = deque(maxlen=50)
        self._phase: str = "bootstrap"
        self._episode: int = 0
        self._last_issue_id: Optional[str] = None
        self._history_window: int = history_window
        self._phase_transitions: List[dict] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(self) -> dict:
        """Select and return the next issue to serve.

        Increments the episode counter, checks for a phase transition, then
        picks a difficulty level according to the current phase weights before
        randomly choosing an issue from that level.

        An anti-repetition guard prevents the same issue from appearing in
        two consecutive episodes (falls back to the full pool if necessary).

        Returns:
            One issue dict from the internal pool.

        Note:
            Never raises — any internal error falls back to a random easy issue.
        """
        try:
            self._episode += 1
            self._maybe_transition_phase()

            weights_dict = self.PHASE_WEIGHTS[self._phase]
            chosen_level: str = random.choices(
                ["easy", "medium", "hard"],
                weights=[weights_dict["easy"], weights_dict["medium"], weights_dict["hard"]],
                k=1,
            )[0]

            candidates: List[dict] = self._issues[chosen_level]

            # Anti-repetition: exclude the immediately previous issue.
            filtered = [i for i in candidates if i["id"] != self._last_issue_id]
            if not filtered:
                filtered = candidates  # full pool fallback — never get stuck

            issue: dict = random.choice(filtered)
            self._last_issue_id = issue["id"]
            return issue

        except Exception:  # noqa: BLE001
            # Last-resort fallback: any easy issue.
            try:
                return random.choice(self._issues["easy"])
            except Exception:  # noqa: BLE001
                return {}

    def record(self, level: str, reward: float) -> None:
        """Record the outcome of a completed episode.

        Appends an entry to the rolling history buffer. Phase transition
        evaluation is deferred to the next ``sample()`` call.

        Args:
            level:  The difficulty level of the episode (``"easy"``,
                    ``"medium"``, or ``"hard"``).
            reward: The scalar reward received, typically in [0.0, 1.0].

        Note:
            Never raises — any error is silently swallowed.
        """
        try:
            self._history.append({
                "level": level,
                "reward": round(float(reward), 4),
                "episode": self._episode,
            })
        except Exception:  # noqa: BLE001
            pass

    def get_stats(self) -> dict:
        """Return a complete snapshot of curriculum state for monitoring.

        Returns:
            A dict with keys:
                ``current_phase``, ``episode``, ``phase_weights``,
                ``phase_transitions``, ``recent_performance``,
                ``history_window``, ``total_episodes_recorded``,
                ``issue_pool_size``.

        Note:
            Never raises — returns ``{"error": str(exc)}`` on failure.
        """
        try:
            return {
                "current_phase": self._phase,
                "episode": self._episode,
                "phase_weights": self.PHASE_WEIGHTS[self._phase],
                "phase_transitions": self._phase_transitions,
                "recent_performance": {
                    "easy":   self._level_avg("easy"),
                    "medium": self._level_avg("medium"),
                    "hard":   self._level_avg("hard"),
                },
                "history_window": self._history_window,
                "total_episodes_recorded": len(self._history),
                "issue_pool_size": {
                    level: len(issues) for level, issues in self._issues.items()
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> str:
        """The name of the active curriculum phase.

        Returns:
            One of ``"bootstrap"``, ``"intermediate"``, ``"advanced"``.
        """
        return self._phase

    @property
    def phase_weights(self) -> Dict[str, float]:
        """The difficulty sampling weights for the active phase.

        Returns:
            Dict with keys ``"easy"``, ``"medium"``, ``"hard"`` whose values
            sum to 1.0.
        """
        return self.PHASE_WEIGHTS[self._phase]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_transition_phase(self) -> None:
        """Evaluate and apply a phase transition if thresholds are met.

        Called at the start of every ``sample()`` call. Only advances forward
        (bootstrap → intermediate → advanced). The terminal phase never
        transitions.

        Note:
            Never raises — any error is silently swallowed.
        """
        try:
            if self._phase == "bootstrap":
                cfg = self.TRANSITION_THRESHOLDS["bootstrap_to_intermediate"]
                recent_easy = self._recent_level_entries(cfg["level"])
                if (
                    len(recent_easy) >= cfg["min_samples"]
                    and self._mean(recent_easy) >= cfg["min_reward"]
                ):
                    mean_val = self._mean(recent_easy)
                    self._transition_to(
                        "intermediate",
                        trigger=f"easy_avg_{mean_val:.2f}",
                    )

            elif self._phase == "intermediate":
                cfg = self.TRANSITION_THRESHOLDS["intermediate_to_advanced"]
                recent_medium = self._recent_level_entries(cfg["level"])
                if (
                    len(recent_medium) >= cfg["min_samples"]
                    and self._mean(recent_medium) >= cfg["min_reward"]
                ):
                    mean_val = self._mean(recent_medium)
                    self._transition_to(
                        "advanced",
                        trigger=f"medium_avg_{mean_val:.2f}",
                    )

            # "advanced" is the terminal phase — nothing to do.

        except Exception:  # noqa: BLE001
            pass

    def _transition_to(self, new_phase: str, trigger: str) -> None:
        """Record and apply a phase transition.

        Args:
            new_phase: The phase to move into.
            trigger:   A short descriptive string explaining why the transition
                       fired (e.g. ``"easy_avg_0.85"``).
        """
        self._phase_transitions.append({
            "from": self._phase,
            "to": new_phase,
            "episode": self._episode,
            "trigger": trigger,
        })
        print(
            f"\n[Curriculum] Phase transition: {self._phase} -> {new_phase} "
            f"(episode {self._episode}, trigger: {trigger})\n"
        )
        self._phase = new_phase

    def _level_avg(self, level: str) -> Optional[float]:
        """Mean reward for one difficulty level over the recent history window.

        Args:
            level: One of ``"easy"``, ``"medium"``, ``"hard"``.

        Returns:
            Rounded mean float, or ``None`` if no matching entries exist in
            the window.
        """
        entries = self._recent_level_entries(level)
        if not entries:
            return None
        return round(
            sum(r["reward"] for r in entries) / len(entries),
            3,
        )

    def _recent_level_entries(self, level: str) -> List[dict]:
        """Return history entries for a given level within the history window.

        Scans the last ``self._history_window`` entries of the ring buffer and
        keeps only those matching ``level``.

        Args:
            level: Difficulty level to filter on.

        Returns:
            List of matching history entry dicts (may be empty).
        """
        window: List[dict] = list(self._history)[-self._history_window:]
        return [e for e in window if e["level"] == level]

    @staticmethod
    def _mean(entries: List[dict]) -> float:
        """Compute mean reward from a list of history entry dicts.

        Args:
            entries: Non-empty list of dicts each containing ``"reward"`` key.

        Returns:
            Arithmetic mean of ``reward`` values as a float.
        """
        return sum(e["reward"] for e in entries) / len(entries)
