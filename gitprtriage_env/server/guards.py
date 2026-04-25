"""
server/guards.py
~~~~~~~~~~~~~~~~
Anti-reward-hacking guards for the GitPRTriage RL environment.

Guards post-process the reward returned by the core grader without
touching its internal logic.  Each guard is stateless (KeywordStuffingDetector,
FixQualityValidator, TimingGuard) or lightly stateful (RepetitionDetector).
They are orchestrated by GuardSuite, which multiplies penalties from every
triggered guard, clamps the result to [0.001, 0.999], and maintains an
audit deque for inspection via the FastAPI /audit endpoint.

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

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GuardResult
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    """Outcome of a single guard check.

    Attributes
    ----------
    triggered:
        ``True`` when the guard detected a gaming strategy and a penalty
        should be applied.
    penalty:
        Reward multiplier.  1.0 means no change; values < 1.0 reduce the
        reward proportionally (e.g. 0.5 halves it).  Only applied when
        ``triggered`` is ``True``.
    reason:
        Human-readable explanation of why the guard fired.
    guard_name:
        Identifier of the guard that produced this result.
    """

    triggered: bool
    penalty: float
    reason: str
    guard_name: str


def _ok(guard_name: str) -> GuardResult:
    """Convenience constructor for a non-triggered, no-penalty result."""
    return GuardResult(triggered=False, penalty=1.0, reason="", guard_name=guard_name)


# ---------------------------------------------------------------------------
# KeywordStuffingDetector
# ---------------------------------------------------------------------------


class KeywordStuffingDetector:
    """Detect when an agent dumps many keywords into *suggested_fix* without
    coherent sentence structure — a pure reward-gaming strategy.

    The guard computes the fraction of words in the fix that are ground-truth
    keywords.  If that density exceeds 0.40 **and** every keyword is present,
    it applies a scaled penalty.
    """

    name: str = "KeywordStuffingDetector"

    def check(self, action: dict, truth: dict) -> GuardResult:
        """Evaluate keyword density in the agent's suggested fix.

        Parameters
        ----------
        action:
            The agent's submitted action dict (must contain ``suggested_fix``).
        truth:
            Ground-truth dict (must contain ``true_fix_keywords`` list).

        Returns
        -------
        GuardResult
            Triggered with scaled penalty if keyword density > 0.40, else
            non-triggered.
        """
        try:
            suggested_fix: str = (action.get("suggested_fix") or "").strip()
            fix_keywords: List[str] = [
                k.lower() for k in truth.get("true_fix_keywords", [])
            ]

            # Nothing to evaluate
            if not suggested_fix or not fix_keywords:
                return _ok(self.name)

            fix_words: List[str] = suggested_fix.lower().split()

            # Too short to classify as stuffing
            if len(fix_words) < 3:
                return _ok(self.name)

            keyword_hits: int = sum(
                1 for kw in fix_keywords if kw in suggested_fix.lower()
            )
            word_count: int = len(fix_words)

            # Only evaluate density when all keywords appear (the stuffing
            # condition — the agent crammed every expected keyword in)
            if keyword_hits >= len(fix_keywords) and word_count > 0:
                keyword_density: float = keyword_hits / max(word_count, 1)

                if keyword_density > 0.4:
                    penalty: float = max(0.5, 1.0 - keyword_density)
                    return GuardResult(
                        triggered=True,
                        penalty=penalty,
                        reason=(
                            f"Fix keyword density {keyword_density:.2f} exceeds 0.40 "
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
        How many recent episodes to consider (the deque ``maxlen``).
    max_repeats:
        Number of identical fingerprints in the window before a penalty fires.
    """

    name: str = "RepetitionDetector"

    def __init__(self, window: int = 10, max_repeats: int = 3) -> None:
        self._window: int = window
        self._max_repeats: int = max_repeats
        # Stores compact string fingerprints for the last *window* episodes.
        self._history: deque = deque(maxlen=window)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fingerprint(self, action: dict) -> str:
        """Create a compact, hashable identifier for an action.

        Only structural fields are included; ``suggested_fix`` is intentionally
        excluded because its natural variability would mask true repetition.

        Parameters
        ----------
        action:
            The agent's submitted action dict.

        Returns
        -------
        str
            A pipe-separated fingerprint string.
        """
        return (
            f"{action.get('classification', '?')}"
            f"|{action.get('bug_line', '?')}"
            f"|{action.get('team', '?')}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, action: dict) -> GuardResult:
        """Evaluate whether the current action fingerprint is repeated too often.

        Parameters
        ----------
        action:
            The agent's submitted action dict.

        Returns
        -------
        GuardResult
            Triggered with a scaled penalty if the fingerprint appears
            ``>= max_repeats`` times in the recent history window.
        """
        try:
            fp: str = self._fingerprint(action)

            # Count occurrences *before* appending so the current episode
            # contributes to the next check (not this one).
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
    """Enforce minimum quality on fix suggestions.

    Rejects:
    * Single-word or very short dumps (< 4 words).
    * All-caps responses (shouted keyword lists).
    * Fixes that contain no action verb (keyword lists, not sentences).
    """

    name: str = "FixQualityValidator"

    COMMON_VERBS: List[str] = [
        "add",
        "use",
        "move",
        "replace",
        "change",
        "remove",
        "fix",
        "update",
        "ensure",
        "check",
        "set",
        "call",
        "wrap",
        "avoid",
        "pass",
        "initialize",
        "load",
        "cache",
        "return",
        "raise",
        "convert",
        "apply",
        "enable",
        "disable",
        "import",
        "export",
        "create",
        "delete",
        "append",
        "insert",
    ]

    def check(self, action: dict) -> GuardResult:
        """Validate the quality of the agent's suggested fix.

        Parameters
        ----------
        action:
            The agent's submitted action dict (checked for ``suggested_fix``).

        Returns
        -------
        GuardResult
            Triggered with appropriate penalty if any quality check fails.
        """
        try:
            fix: str = (action.get("suggested_fix") or "").strip()

            # No fix attempted — the core grader handles this separately.
            if not fix:
                return _ok(self.name)

            words: List[str] = fix.split()

            # Check 1 — minimum length
            if len(words) < 4:
                return GuardResult(
                    triggered=True,
                    penalty=0.7,
                    reason=(
                        "Fix too short: fewer than 4 words — "
                        "likely a keyword dump not a sentence"
                    ),
                    guard_name=self.name,
                )

            # Check 2 — no action verb present
            fix_lower: str = fix.lower()
            has_verb: bool = any(verb in fix_lower for verb in self.COMMON_VERBS)
            if not has_verb:
                return GuardResult(
                    triggered=True,
                    penalty=0.8,
                    reason=(
                        "Fix contains no action verb — "
                        "appears to be a keyword list not a fix description"
                    ),
                    guard_name=self.name,
                )

            # Check 3 — all uppercase
            if fix == fix.upper() and len(fix) > 3:
                return GuardResult(
                    triggered=True,
                    penalty=0.7,
                    reason=(
                        "Fix is all uppercase — "
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

    This is a *soft* guard: it applies only a mild 5 % penalty and is
    primarily intended to surface a signal in the audit log rather than to
    strongly penalise the agent.

    Parameters
    ----------
    min_ms:
        Minimum expected inference latency in milliseconds.  Responses
        arriving faster than this threshold are flagged.
    """

    name: str = "TimingGuard"

    def __init__(self, min_ms: float = 200.0) -> None:
        self._min_ms: float = min_ms
        self._fast_response_count: int = 0
        self._total_checked: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fast_response_rate(self) -> float:
        """Fraction of checked episodes that were faster than *min_ms*."""
        return self._fast_response_count / max(self._total_checked, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, elapsed_ms: float) -> GuardResult:
        """Evaluate whether the response arrived suspiciously quickly.

        Parameters
        ----------
        elapsed_ms:
            Wall-clock time (milliseconds) from action submission to reward
            computation.  Values <= 0 are treated as missing / unmeasured
            and are not flagged.

        Returns
        -------
        GuardResult
            Triggered with a mild 0.95 penalty if ``elapsed_ms`` is below
            the configured threshold.
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

    All four guards are composed here.  Penalties from triggered guards are
    *multiplied* together (i.e. they are independent factors), and the result
    is clamped to ``[0.001, 0.999]`` to keep the reward always in-range.

    The audit deque (``maxlen=200``) is exposed via :meth:`get_audit_log` for
    the FastAPI ``/audit`` endpoint.
    """

    def __init__(self) -> None:
        self.keyword_guard: KeywordStuffingDetector = KeywordStuffingDetector()
        self.repetition_guard: RepetitionDetector = RepetitionDetector()
        self.fix_quality_guard: FixQualityValidator = FixQualityValidator()
        self.timing_guard: TimingGuard = TimingGuard()

        # Audit deque — stores dicts for the last 200 penalised episodes.
        self._audit_log: deque = deque(maxlen=200)
        self._total_penalties_applied: int = 0
        self._episode: int = 0

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        action: dict,
        truth: dict,
        original_reward: float,
        elapsed_ms: float = 0.0,
    ) -> Tuple[float, List[GuardResult]]:
        """Run all guards and return the adjusted reward.

        This is the primary entry-point called once per environment step.

        Parameters
        ----------
        action:
            The agent's submitted action dict.
        truth:
            Ground-truth dict used by the grader (and keyword guard).
        original_reward:
            The reward produced by the core grader, in ``[0.001, 0.999]``.
        elapsed_ms:
            Wall-clock latency of the agent's response in milliseconds.
            Pass ``0.0`` (default) if not measured.

        Returns
        -------
        Tuple[float, List[GuardResult]]
            ``(adjusted_reward, results)`` where *results* is the list of
            :class:`GuardResult` objects (one per guard, in declaration order).
            If an unexpected exception occurs, the original reward is returned
            unchanged and *results* is an empty list.
        """
        try:
            self._episode += 1

            results: List[GuardResult] = [
                self.keyword_guard.check(action, truth),
                self.repetition_guard.check(action),
                self.fix_quality_guard.check(action),
                self.timing_guard.check(elapsed_ms),
            ]

            # Multiply penalties only from triggered guards.
            penalty_multiplier: float = 1.0
            for r in results:
                if r.triggered:
                    penalty_multiplier *= r.penalty

            adjusted_reward: float = original_reward * penalty_multiplier
            # Clamp to environment reward range.
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
                    f"reward {original_reward:.3f}→{adjusted_reward:.3f} "
                    f"(×{penalty_multiplier:.3f}) — "
                    f"{[r.guard_name for r in triggered_results]}"
                )

            return (adjusted_reward, results)

        except Exception:  # noqa: BLE001
            # Never lose the original reward due to a guard bug.
            return (original_reward, [])

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_audit_log(self, n: int = 20) -> list:
        """Return the last *n* audit entries as a plain list.

        Parameters
        ----------
        n:
            Maximum number of most-recent entries to return.

        Returns
        -------
        list
            A list of audit-entry dicts, newest-last.
        """
        entries = list(self._audit_log)
        return entries[-n:]

    def get_summary(self) -> dict:
        """Return a high-level summary of guard activity.

        Returns
        -------
        dict
            Summary statistics and guard descriptions suitable for the
            FastAPI ``/audit`` summary endpoint.
        """
        return {
            "total_episodes": self._episode,
            "total_penalties_applied": self._total_penalties_applied,
            "penalty_rate": round(
                self._total_penalties_applied / max(self._episode, 1), 3
            ),
            "fast_response_rate": self.timing_guard.fast_response_rate,
            "guards": {
                "KeywordStuffingDetector": (
                    "Penalises fix suggestions with >40% keyword density"
                ),
                "RepetitionDetector": (
                    "Penalises identical action fingerprints repeated "
                    ">3 times in 10 episodes"
                ),
                "FixQualityValidator": (
                    "Penalises fix suggestions with <4 words or no action verb"
                ),
                "TimingGuard": (
                    "Flags responses under 200ms as possible cached outputs"
                ),
            },
        }
