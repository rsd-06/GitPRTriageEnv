"""
agents/specialists.py
---------------------
Specialist agent implementations for the GitPRTriageEnv multi-agent triage system.

Four agents are defined here, each responsible for one component of the final
action submitted to the environment:

    ClassifierAgent   — bug / feature / duplicate
    BugLocatorAgent   — exact 1-indexed bug line number (or null)
    TeamRouterAgent   — webdev / devops / aiml (or null)
    FixSuggesterAgent — one-sentence concrete fix description (or null)

All agents inherit BaseAgent, use temperature=0.0 for determinism, and
guarantee that _parse_response and _keyword_fallback never raise exceptions.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agents.base import AgentResult, BaseAgent, safe_json_parse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_CLASSIFICATIONS: frozenset[str] = frozenset({"bug", "feature", "duplicate"})
_VALID_TEAMS: frozenset[Optional[str]] = frozenset({"webdev", "devops", "aiml", None})


def _clamp(value: float) -> float:
    """Clamp a float confidence value to the range [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# 1. ClassifierAgent
# ---------------------------------------------------------------------------


class ClassifierAgent(BaseAgent):
    """Classifies a GitHub issue as 'bug', 'feature', or 'duplicate'.

    This agent uses the issue title, body, and existing labels to produce
    a classification with an associated confidence score and reasoning.
    """

    @property
    def name(self) -> str:
        """Return the display name of this agent.

        Returns:
            ``"ClassifierAgent"``
        """
        return "ClassifierAgent"

    def _get_system_prompt(self) -> str:
        """Return the system-level instruction for the classifier.

        Returns:
            A prompt instructing the model to return a raw JSON classification.
        """
        return (
            "You are an expert at classifying GitHub issues into exactly one of three "
            "categories. Respond ONLY with valid JSON: "
            '{"classification": "bug"|"feature"|"duplicate", "confidence": 0.0-1.0, '
            '"reasoning": "one sentence"}. '
            "Rules: bug=something broken/crashing/wrong behavior; "
            "feature=new capability requested; "
            'duplicate=explicitly references another issue number or says "same as". '
            "Never use markdown. Output raw JSON only."
        )

    def build_prompt(self, observation: dict) -> str:
        """Build the user prompt from the observation.

        Uses title, body, and existing_labels only.

        Args:
            observation: Environment observation dict.

        Returns:
            Formatted string for the user turn.
        """
        title: str = observation.get("title", "") or ""
        body: str = observation.get("body", "") or ""
        labels: list = observation.get("existing_labels", []) or []
        labels_str = ", ".join(labels) if labels else "none"
        return (
            f"Title: {title}\n"
            f"Body: {body}\n"
            f"Labels: {labels_str}\n\n"
            "Classify this issue."
        )

    def _parse_response(self, raw: str) -> AgentResult:
        """Parse the LLM response into an AgentResult.

        Validates that the classification is one of the three legal values.
        Falls back to None if it is not. Clamps confidence to [0.0, 1.0].

        Args:
            raw: Verbatim LLM output string.

        Returns:
            AgentResult with value set to the classification string or None.
        """
        try:
            data: dict = safe_json_parse(raw)
            raw_cls = data.get("classification")
            classification: Optional[str] = (
                raw_cls if raw_cls in _VALID_CLASSIFICATIONS else None
            )
            confidence: float = _clamp(float(data.get("confidence", 0.5)))
            reasoning: str = str(data.get("reasoning", ""))
            return AgentResult(
                value=classification,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw,
                agent_name=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Parsing failed: {exc}",
                raw_response=raw,
                agent_name=self.name,
            )

    def _keyword_fallback(self, observation: dict) -> AgentResult:
        """Classify using keyword heuristics when the LLM is unavailable.

        Extends the base heuristic by also checking the ``existing_labels``
        list: a label of ``"duplicate"`` immediately raises confidence to 0.6.

        Priority order: duplicate label → duplicate keywords → feature keywords
        → bug keywords → default bug.

        Args:
            observation: Environment observation dict.

        Returns:
            AgentResult with confidence=0.4 (or 0.6 for label-based duplicate).
        """
        try:
            title: str = observation.get("title", "") or ""
            body: str = observation.get("body", "") or ""
            labels: list = observation.get("existing_labels", []) or []
            text = (title + " " + body).lower()

            # Label-based duplicate detection (higher confidence).
            if "duplicate" in [lbl.lower() for lbl in labels]:
                return AgentResult(
                    value="duplicate",
                    confidence=0.6,
                    reasoning="Existing label 'duplicate' detected.",
                    raw_response="[keyword_fallback]",
                    agent_name=self.name,
                )

            duplicate_signals = ["duplicate", "same as", "already filed", "#"]
            feature_signals = ["add", "support", "feature", "request", "would be nice"]
            bug_signals = ["crash", "error", "fail", "broken", "not working"]

            if any(kw in text for kw in duplicate_signals):
                classification, reasoning = "duplicate", "Keyword heuristic matched duplicate signal words."
            elif any(kw in text for kw in feature_signals):
                classification, reasoning = "feature", "Keyword heuristic matched feature-request signal words."
            elif any(kw in text for kw in bug_signals):
                classification, reasoning = "bug", "Keyword heuristic matched bug signal words."
            else:
                classification, reasoning = "bug", "No strong signal found; defaulting to bug."

            return AgentResult(
                value=classification,
                confidence=0.4,
                reasoning=reasoning,
                raw_response="[keyword_fallback]",
                agent_name=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Keyword fallback failed: {exc}",
                raw_response="[keyword_fallback_error]",
                agent_name=self.name,
            )


# ---------------------------------------------------------------------------
# 2. BugLocatorAgent
# ---------------------------------------------------------------------------


class BugLocatorAgent(BaseAgent):
    """Identifies the exact 1-indexed line number where a bug resides.

    If no code snippet is present or no bug is identifiable, the agent
    returns ``value=None``.
    """

    @property
    def name(self) -> str:
        """Return the display name of this agent.

        Returns:
            ``"BugLocatorAgent"``
        """
        return "BugLocatorAgent"

    def _get_system_prompt(self) -> str:
        """Return the system-level instruction for the bug locator.

        Returns:
            A prompt instructing the model to return the bug line as raw JSON.
        """
        return (
            "You are an expert code reviewer who identifies the exact line number "
            "containing a bug. Respond ONLY with JSON: "
            '{"bug_line": integer_or_null, "confidence": 0.0-1.0, '
            '"reasoning": "one sentence explaining what is wrong on that line"}. '
            "If there is no code snippet or no bug, set bug_line to null. "
            "Line numbers are 1-indexed as shown in the snippet. "
            "Output raw JSON only, no markdown."
        )

    def build_prompt(self, observation: dict) -> str:
        """Build the user prompt from the observation.

        Uses body and code_snippet only. If the snippet is absent, prompts the
        model to return ``bug_line: null``.

        Args:
            observation: Environment observation dict.

        Returns:
            Formatted string for the user turn.
        """
        body: str = observation.get("body", "") or ""
        snippet: Optional[str] = observation.get("code_snippet")

        if snippet is None:
            return "No code snippet provided. Return bug_line: null."

        return (
            f"Issue description: {body}\n\n"
            f"Code (lines are 1-indexed):\n{snippet}\n\n"
            "Identify the exact line containing the bug."
        )

    def _parse_response(self, raw: str) -> AgentResult:
        """Parse the LLM response into an AgentResult.

        Converts ``bug_line`` to a positive integer, or sets it to None if the
        value is absent, non-numeric, or non-positive. Clamps confidence.

        Args:
            raw: Verbatim LLM output string.

        Returns:
            AgentResult with value set to a positive int or None.
        """
        try:
            data: dict = safe_json_parse(raw)
            raw_line = data.get("bug_line")
            bug_line: Optional[int] = None

            if raw_line is not None:
                try:
                    candidate = int(raw_line)
                    if candidate > 0:
                        bug_line = candidate
                except (TypeError, ValueError):
                    bug_line = None

            confidence: float = _clamp(float(data.get("confidence", 0.5)))
            reasoning: str = str(data.get("reasoning", ""))
            return AgentResult(
                value=bug_line,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw,
                agent_name=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Parsing failed: {exc}",
                raw_response=raw,
                agent_name=self.name,
            )

    def _keyword_fallback(self, observation: dict) -> AgentResult:
        """Return a zero-confidence result when no LLM is available.

        Code analysis cannot be performed without an LLM, so this fallback
        always returns ``value=None``.

        Args:
            observation: Environment observation dict (unused).

        Returns:
            AgentResult with value=None and confidence=0.0.
        """
        return AgentResult(
            value=None,
            confidence=0.0,
            reasoning="No code analysis available without LLM.",
            raw_response="[keyword_fallback]",
            agent_name=self.name,
        )


# ---------------------------------------------------------------------------
# 3. TeamRouterAgent
# ---------------------------------------------------------------------------


class TeamRouterAgent(BaseAgent):
    """Routes a GitHub issue to the correct internal engineering team.

    Valid routing targets are ``"webdev"``, ``"devops"``, ``"aiml"``, or
    ``None`` when the team cannot be determined.
    """

    @property
    def name(self) -> str:
        """Return the display name of this agent.

        Returns:
            ``"TeamRouterAgent"``
        """
        return "TeamRouterAgent"

    def _get_system_prompt(self) -> str:
        """Return the system-level instruction for the team router.

        Returns:
            A prompt detailing team responsibilities and expected JSON output.
        """
        return (
            "You are an expert at routing engineering issues to the correct internal team. "
            "Teams are: "
            "webdev (frontend, backend, REST APIs, web security, databases, sessions, auth), "
            "devops (Docker, Kubernetes, CI/CD, GitHub Actions, infrastructure, deployment, monitoring), "
            "aiml (machine learning models, training pipelines, GPU/CUDA, embeddings, NLP, data preprocessing). "
            'Respond ONLY with JSON: {"team": "webdev"|"devops"|"aiml"|null, '
            '"confidence": 0.0-1.0, "reasoning": "one sentence"}. '
            "Set null if truly cannot determine. Output raw JSON only."
        )

    def build_prompt(self, observation: dict) -> str:
        """Build the user prompt from the observation.

        Uses title, body, existing_labels, and the optional
        ``classification_context`` key injected by the orchestrator.

        Args:
            observation: Environment observation dict. May contain an extra key
                         ``classification_context`` (str) with the previously
                         determined issue classification.

        Returns:
            Formatted string for the user turn.
        """
        title: str = observation.get("title", "") or ""
        body: str = observation.get("body", "") or ""
        labels: list = observation.get("existing_labels", []) or []
        labels_str = ", ".join(labels) if labels else "none"
        cls_context: str = observation.get("classification_context") or "unknown"

        return (
            f"Title: {title}\n"
            f"Body: {body}\n"
            f"Labels: {labels_str}\n"
            f"Issue type: {cls_context}\n\n"
            "Route to the correct team."
        )

    def _parse_response(self, raw: str) -> AgentResult:
        """Parse the LLM response into an AgentResult.

        Validates that ``team`` is one of the four legal values (including
        None). Clamps confidence to [0.0, 1.0].

        Args:
            raw: Verbatim LLM output string.

        Returns:
            AgentResult with value set to the team string or None.
        """
        try:
            data: dict = safe_json_parse(raw)
            raw_team = data.get("team")
            team: Optional[str] = raw_team if raw_team in _VALID_TEAMS else None
            confidence: float = _clamp(float(data.get("confidence", 0.5)))
            reasoning: str = str(data.get("reasoning", ""))
            return AgentResult(
                value=team,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw,
                agent_name=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Parsing failed: {exc}",
                raw_response=raw,
                agent_name=self.name,
            )

    def _keyword_fallback(self, observation: dict) -> AgentResult:
        """Route using keyword heuristics when the LLM is unavailable.

        Checks title and body (both lowercased) for domain-specific signal
        words in priority order: devops → aiml → webdev (default).

        Args:
            observation: Environment observation dict.

        Returns:
            AgentResult with confidence=0.45 and the heuristic team routing.
        """
        try:
            title: str = observation.get("title", "") or ""
            body: str = observation.get("body", "") or ""
            text = (title + " " + body).lower()

            devops_signals = ["docker", "kubernetes", "k8s", "ci", "deploy", "pipeline", "helm"]
            aiml_signals = ["model", "training", "cuda", "gpu", "embedding", "torch", "tensorflow", "ml", "neural"]

            if any(kw in text for kw in devops_signals):
                team, reasoning = "devops", "Keyword heuristic matched DevOps signal words."
            elif any(kw in text for kw in aiml_signals):
                team, reasoning = "aiml", "Keyword heuristic matched AIML signal words."
            else:
                team, reasoning = "webdev", "No DevOps/AIML signal; defaulting to webdev."

            return AgentResult(
                value=team,
                confidence=0.45,
                reasoning=reasoning,
                raw_response="[keyword_fallback]",
                agent_name=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Keyword fallback failed: {exc}",
                raw_response="[keyword_fallback_error]",
                agent_name=self.name,
            )


# ---------------------------------------------------------------------------
# 4. FixSuggesterAgent
# ---------------------------------------------------------------------------


class FixSuggesterAgent(BaseAgent):
    """Generates a concise, technically specific one-sentence fix suggestion.

    Uses the issue body, code snippet, and optional context from the
    orchestrator (bug line number and routed team) to produce an actionable
    fix description.
    """

    @property
    def name(self) -> str:
        """Return the display name of this agent.

        Returns:
            ``"FixSuggesterAgent"``
        """
        return "FixSuggesterAgent"

    def _get_system_prompt(self) -> str:
        """Return the system-level instruction for the fix suggester.

        Returns:
            A prompt instructing the model to produce a specific one-sentence
            fix description as raw JSON.
        """
        return (
            "You are a senior engineer who writes concise, actionable bug fix suggestions. "
            'Respond ONLY with JSON: {"suggested_fix": "one concrete sentence describing '
            'exactly what to change and why", "confidence": 0.0-1.0, '
            '"reasoning": "one sentence"}. '
            "The fix must be specific — name the exact function, variable, or concept to change. "
            "Do not say 'fix the bug'. Mention the technical solution. "
            "Output raw JSON only."
        )

    def build_prompt(self, observation: dict) -> str:
        """Build the user prompt from the observation.

        Uses body, code_snippet, and optional orchestrator-injected keys
        ``bug_line_context`` (int or None) and ``team_context`` (str or None).

        Args:
            observation: Environment observation dict. May contain extra keys
                         ``bug_line_context`` and ``team_context`` injected by
                         the orchestrator.

        Returns:
            Formatted string for the user turn.
        """
        body: str = observation.get("body", "") or ""
        snippet: Optional[str] = observation.get("code_snippet")
        bug_line: Optional[int] = observation.get("bug_line_context")
        team: Optional[str] = observation.get("team_context")

        snippet_text = snippet if snippet is not None else "No code provided"
        line_text = str(bug_line) if bug_line is not None else "unknown"
        team_text = team if team is not None else "unknown"

        return (
            f"Issue: {body}\n\n"
            f"Code:\n{snippet_text}\n\n"
            f"The bug is on line {line_text}. Team: {team_text}.\n\n"
            "Suggest a specific one-sentence fix."
        )

    def _parse_response(self, raw: str) -> AgentResult:
        """Parse the LLM response into an AgentResult.

        Extracts the ``suggested_fix`` string. If it is absent or empty, sets
        value to None. Clamps confidence to [0.0, 1.0].

        Args:
            raw: Verbatim LLM output string.

        Returns:
            AgentResult with value set to the fix string or None.
        """
        try:
            data: dict = safe_json_parse(raw)
            fix: Optional[str] = data.get("suggested_fix") or None
            if fix is not None:
                fix = fix.strip() or None
            confidence: float = _clamp(float(data.get("confidence", 0.5)))
            reasoning: str = str(data.get("reasoning", ""))
            return AgentResult(
                value=fix,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw,
                agent_name=self.name,
            )
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Parsing failed: {exc}",
                raw_response=raw,
                agent_name=self.name,
            )

    def _keyword_fallback(self, observation: dict) -> AgentResult:
        """Return a zero-confidence result when the LLM is unavailable.

        Generating a specific fix suggestion without an LLM is not feasible,
        so this always returns ``value=None``.

        Args:
            observation: Environment observation dict (unused).

        Returns:
            AgentResult with value=None and confidence=0.0.
        """
        return AgentResult(
            value=None,
            confidence=0.0,
            reasoning="Fix suggestion unavailable without LLM.",
            raw_response="[keyword_fallback]",
            agent_name=self.name,
        )
