"""
server/guards.py
~~~~~~~~~~~~~~~~
Anti-reward-hacking guards for the PRRegressionAuditEnv RL environment.

Guards post-process the reward returned by the core grader without
touching its internal logic. Each guard is stateless (KeywordStuffingDetector,
FixQualityValidator, TimingGuard) or lightly stateful (RepetitionDetector).
They are orchestrated by GuardSuite, which multiplies penalties from every
triggered guard, clamps the result to [0.001, 0.999], and maintains an
audit deque for inspection via the FastAPI /guards/audit endpoint.

Penalty semantics
-----------------
penalty == 1.0  →  no change to the reward
penalty == 0.5  →  reward is halved
Penalties are multiplicative: two guards at 0.8 → 0.64× the original reward.

Usage
-----
    suite = GuardSuite()
    adjusted, results = suite.evaluate(action, truth, original_reward, elapsed_ms)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# GuardResult
# ---------------------------------------------------------------------------


class GuardResult:
    """Outcome of a single guard check.

    Attributes
    ----------
    triggered:
        True when the guard detected a gaming strategy and a penalty
        should be applied.
    penalty:
        Reward multiplier. 1.0 means no change; values < 1.0 reduce the
        reward proportionally. Only applied when triggered is True.
    reason:
        Human-readable explanation of why the guard fired.
    guard_name:
        Identifier of the guard that produced this result.
    """

    __slots__ = ("triggered", "penalty", "reason", "guard_name")

    def __init__(
        self,
        triggered: bool,
        penalty: float,
        reason: str,
        guard_name: str,
    ) -> None:
        self.triggered = triggered
        self.penalty = penalty
        self.reason = reason
        self.guard_name = guard_name


def _ok(guard_name: str) -> GuardResult:
    """Convenience constructor for a non-triggered, no-penalty result."""
    return GuardResult(triggered=False, penalty=1.0, reason="", guard_name=guard_name)


# ---------------------------------------------------------------------------
# KeywordStuffingDetector
# ---------------------------------------------------------------------------


class KeywordStuffingDetector:
    """Detect when an agent dumps many keywords into suggested_change without
    coherent sentence structure — a pure reward-gaming strategy.

    The guard computes the fraction of words in the fix that are ground-truth
    keywords. If that density exceeds 0.40 and every keyword is present,
    it applies a scaled penalty.

    Note: The hard 200-char limit in the grader is the primary defence.
    This guard catches cases that pass the char limit but are still keyword-dense.
    """

    name: str = "KeywordStuffingDetector"

    def check(self, action: dict, truth: dict) -> GuardResult:
        """Evaluate keyword density in the agent's suggested change.

        Parameters
        ----------
        action:
            The agent's submitted action dict (checks suggested_change).
        truth:
            Ground-truth dict (checks true_fix_keywords list).
        """
        try:
            suggested: str = (action.get("suggested_change") or "").strip()
            fix_keywords: List[str] = [
                k.lower() for k in truth.get("true_fix_keywords", [])
            ]

            if not suggested or not fix_keywords:
                return _ok(self.name)

            fix_words: List[str] = suggested.lower().split()

            if len(fix_words) < 3:
                return _ok(self.name)

            keyword_hits: int = sum(
                1 for kw in fix_keywords if kw in suggested.lower()
            )
            word_count: int = len(fix_words)

            # Only evaluate density when all keywords appear
            if keyword_hits >= len(fix_keywords) and word_count > 0:
                keyword_density: float = keyword_hits / max(word_count, 1)

                if keyword_density > 0.4:
                    penalty: float = max(0.5, 1.0 - keyword_density)
                    return GuardResult(
                        triggered=True,
                        penalty=penalty,
                        reason=(
                            f"suggested_change keyword density {keyword_density:.2f} exceeds 0.40 "
                            f"threshold ({keyword_hits}/{word_count} words are keywords)"
                        ),
                        guard_name=self.name,
                    )

            return _ok(self.name)

        except Exception:  # noqa: BLE001
            return _ok(self.name)


# ---------------------------------------------------------------------------
# RepetitionDetector
# ---------------------------------------------------------------------------


class RepetitionDetector:
    """Detect agents that output the same action repeatedly across episodes.

    Repeated identical fingerprints across a rolling window are a strong
    signal of caching or hard-coded responses rather than genuine inference.

    Parameters
    ----------
    window:
        How many recent episodes to consider (the deque maxlen).
    max_repeats:
        Number of identical fingerprints in the window before a penalty fires.
    """

    name: str = "RepetitionDetector"

    def __init__(self, window: int = 10, max_repeats: int = 3) -> None:
        self._window: int = window
        self._max_repeats: int = max_repeats
        self._history: deque = deque(maxlen=window)

    def _fingerprint(self, action: dict) -> str:
        """Create a compact, hashable identifier for a ReviewAction.

        Uses review_decision, defect_category, faulty_line, and reviewer_team.
        suggested_change is intentionally excluded because its variability
        would mask true repetition.
        """
        return (
            f"{action.get('review_decision', '?')}"
            f"|{action.get('defect_category', '?')}"
            f"|{action.get('faulty_line', '?')}"
            f"|{action.get('reviewer_team', '?')}"
        )

    def check(self, action: dict) -> GuardResult:
        """Evaluate whether the current action fingerprint is repeated too often."""
        try:
            fp: str = self._fingerprint(action)
            repeat_count: int = sum(1 for h in self._history if h == fp)
            self._history.append(fp)

            if repeat_count >= self._max_repeats:
                excess: int = repeat_count - self._max_repeats + 1
                penalty: float = max(0.5, 1.0 - excess * 0.1)
                return GuardResult(
                    triggered=True,
                    penalty=penalty,
                    reason=(
                        f"Action fingerprint '{fp}' repeated {repeat_count} times "
                        f"in last {self._window} episodes "
                        f"(max allowed: {self._max_repeats})"
                    ),
                    guard_name=self.name,
                )

            return _ok(self.name)

        except Exception:  # noqa: BLE001
            return _ok(self.name)


# ---------------------------------------------------------------------------
# FixQualityValidator
# ---------------------------------------------------------------------------


class FixQualityValidator:
    """Enforce minimum quality on suggested_change strings.

    Rejects:
    * Very short suggestions (< 4 words).
    * ALL-CAPS responses (shouted keyword lists).
    * Suggestions with no action verb (keyword lists, not sentences).
    """

    name: str = "FixQualityValidator"

    COMMON_VERBS: List[str] = [
        "add", "use", "move", "replace", "change", "remove", "fix", "update",
        "ensure", "check", "set", "call", "wrap", "avoid", "pass", "initialize",
        "load", "cache", "return", "raise", "convert", "apply", "enable",
        "disable", "import", "export", "create", "delete", "append", "insert",
    ]

    def check(self, action: dict) -> GuardResult:
        """Validate the quality of the agent's suggested change."""
        try:
            fix: str = (action.get("suggested_change") or "").strip()

            if not fix:
                return _ok(self.name)

            words: List[str] = fix.split()

            if len(words) < 4:
                return GuardResult(
                    triggered=True,
                    penalty=0.7,
                    reason=(
                        "suggested_change too short: fewer than 4 words — "
                        "likely a keyword dump not a sentence"
                    ),
                    guard_name=self.name,
                )

            fix_lower: str = fix.lower()
            has_verb: bool = any(verb in fix_lower for verb in self.COMMON_VERBS)
            if not has_verb:
                return GuardResult(
                    triggered=True,
                    penalty=0.8,
                    reason=(
                        "suggested_change contains no action verb — "
                        "appears to be a keyword list not a fix description"
                    ),
                    guard_name=self.name,
                )

            if fix == fix.upper() and len(fix) > 3:
                return GuardResult(
                    triggered=True,
                    penalty=0.7,
                    reason=(
                        "suggested_change is all uppercase — "
                        "likely not a genuine fix description"
                    ),
                    guard_name=self.name,
                )

            return _ok(self.name)

        except Exception:  # noqa: BLE001
            return _ok(self.name)


# ---------------------------------------------------------------------------
# TimingGuard
# ---------------------------------------------------------------------------


class TimingGuard:
    """Flag suspiciously fast responses that suggest cached / hardcoded outputs.

    Soft guard: applies only a mild 5% penalty. Primarily surfaces a signal
    in the audit log rather than strongly penalising the agent.

    Parameters
    ----------
    min_ms:
        Minimum expected inference latency in milliseconds. Responses
        arriving faster than this threshold are flagged.
    """

    name: str = "TimingGuard"

    def __init__(self, min_ms: float = 10.0) -> None:
        self._min_ms: float = min_ms
        self._fast_response_count: int = 0
        self._total_checked: int = 0

    @property
    def fast_response_rate(self) -> float:
        """Fraction of checked episodes that were faster than min_ms."""
        return self._fast_response_count / max(self._total_checked, 1)

    def check(self, elapsed_ms: float) -> GuardResult:
        """Evaluate whether the response arrived suspiciously quickly.

        Parameters
        ----------
        elapsed_ms:
            Wall-clock time (ms) from action submission to reward computation.
            Values <= 0 are treated as unmeasured and are not flagged.
        """
        try:
            self._total_checked += 1

            if 0 < elapsed_ms < self._min_ms:
                self._fast_response_count += 1
                return GuardResult(
                    triggered=True,
                    penalty=0.95,
                    reason=(
                        f"Response in {elapsed_ms:.0f}ms "
                        f"(threshold: {self._min_ms:.0f}ms) — "
                        f"possible cached response "
                        f"({self._fast_response_count}/{self._total_checked} "
                        f"fast responses so far)"
                    ),
                    guard_name=self.name,
                )

            return _ok(self.name)

        except Exception:  # noqa: BLE001
            return _ok(self.name)


# ---------------------------------------------------------------------------
# GuardSuite
# ---------------------------------------------------------------------------


class GuardSuite:
    """Orchestrate all guards, compute the final adjusted reward, and maintain
    an audit log of every guard firing.

    All four guards are composed here. Penalties from triggered guards are
    multiplied together (independent factors), and the result is clamped to
    [0.001, 0.999] to keep the reward always in-range.
    """

    def __init__(self) -> None:
        self.keyword_guard: KeywordStuffingDetector = KeywordStuffingDetector()
        self.repetition_guard: RepetitionDetector = RepetitionDetector()
        self.fix_quality_guard: FixQualityValidator = FixQualityValidator()
        self.timing_guard: TimingGuard = TimingGuard()

        self._audit_log: deque = deque(maxlen=200)
        self._total_penalties_applied: int = 0
        self._episode: int = 0

    def evaluate(
        self,
        action: dict,
        truth: dict,
        original_reward: float,
        elapsed_ms: float = 0.0,
    ) -> Tuple[float, List[GuardResult]]:
        """Run all guards and return the adjusted reward.

        Parameters
        ----------
        action:
            The agent's submitted ReviewAction dict.
        truth:
            Ground-truth PR dict (used by keyword guard).
        original_reward:
            The reward produced by the core grader, in [0.001, 0.999].
        elapsed_ms:
            Wall-clock latency of the agent's response in milliseconds.

        Returns
        -------
        Tuple[float, List[GuardResult]]
            (adjusted_reward, results). If an exception occurs, original
            reward is returned unchanged.
        """
        try:
            self._episode += 1

            results: List[GuardResult] = [
                self.keyword_guard.check(action, truth),
                self.repetition_guard.check(action),
                self.fix_quality_guard.check(action),
                self.timing_guard.check(elapsed_ms),
            ]

            penalty_multiplier: float = 1.0
            for r in results:
                if r.triggered:
                    penalty_multiplier *= r.penalty

            adjusted_reward: float = original_reward * penalty_multiplier
            adjusted_reward = min(max(adjusted_reward, 0.001), 0.999)

            triggered_results = [r for r in results if r.triggered]
            if triggered_results:
                self._total_penalties_applied += 1
                audit_entry: Dict = {
                    "episode": self._episode,
                    "original_reward": round(original_reward, 4),
                    "adjusted_reward": round(adjusted_reward, 4),
                    "penalty_multiplier": round(penalty_multiplier, 4),
                    "guards_triggered": [r.guard_name for r in triggered_results],
                    "reasons": [r.reason for r in triggered_results],
                }
                self._audit_log.append(audit_entry)
                print(
                    f"  [GuardSuite] Ep{self._episode}: "
                    f"reward {original_reward:.3f}->{adjusted_reward:.3f} "
                    f"(x{penalty_multiplier:.3f}) -- "
                    f"{[r.guard_name for r in triggered_results]}"
                )

            return (adjusted_reward, results)

        except Exception:  # noqa: BLE001
            return (original_reward, [])

    def get_audit_log(self, n: int = 20) -> list:
        """Return the last n audit entries as a plain list."""
        entries = list(self._audit_log)
        return entries[-n:]

    def get_summary(self) -> dict:
        """Return a high-level summary of guard activity."""
        return {
            "total_episodes": self._episode,
            "total_penalties_applied": self._total_penalties_applied,
            "penalty_rate": round(
                self._total_penalties_applied / max(self._episode, 1), 3
            ),
            "fast_response_rate": self.timing_guard.fast_response_rate,
            "guards": {
                "KeywordStuffingDetector": (
                    "Penalises suggested_change with >40% keyword density"
                ),
                "RepetitionDetector": (
                    "Penalises identical action fingerprints repeated "
                    ">3 times in 10 episodes"
                ),
                "FixQualityValidator": (
                    "Penalises suggested_change with <4 words or no action verb"
                ),
                "TimingGuard": (
                    "Flags responses under 200ms as possible cached outputs"
                ),
            },
        }
