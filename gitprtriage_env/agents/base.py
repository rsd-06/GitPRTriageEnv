"""
agents/base.py
--------------
Base abstractions for the multi-agent GitHub issue triage system.

All concrete agents in this package inherit from BaseAgent, implement the
four abstract methods, and rely on the shared AgentResult dataclass and
safe_json_parse helper defined here.
"""

from __future__ import annotations

import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Encapsulates everything produced by a single agent invocation.

    Attributes:
        value:        The agent's parsed answer — a string, integer, or None
                      depending on the agent's role (e.g. "bug", 42, None).
        confidence:   Self-reported confidence in [0.0, 1.0]. Agents should
                      calibrate this honestly; the orchestrator uses it for
                      weighted voting.
        reasoning:    One-sentence explanation of why the agent chose this
                      value. Used for logging and interpretability.
        raw_response: The verbatim LLM output before any parsing. Stored for
                      debugging and replay.
        agent_name:   Display name of the agent that produced this result.
    """

    value: Any
    confidence: float
    reasoning: str
    raw_response: str
    agent_name: str


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base for every triage agent in the GitPRTriageEnv system.

    Subclasses must implement:
        - ``name``            – a display name property
        - ``build_prompt``    – converts an environment observation to a prompt
        - ``_get_system_prompt`` – returns the agent's system-level instruction
        - ``_parse_response`` – converts raw LLM text into an AgentResult

    The ``run`` method orchestrates the full call cycle and guarantees that
    an AgentResult is always returned — even when the LLM call fails.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        temperature: float = 0.0,
    ) -> None:
        """Initialise the agent with an OpenAI-compatible client.

        Args:
            client:      An object exposing ``client.chat.completions.create``.
                         Compatible with the official ``openai`` SDK and any
                         drop-in replacement (e.g. Groq, Together, LiteLLM).
            model_name:  The model identifier string passed to the API.
            temperature: Sampling temperature.  Always pass 0.0 from outside
                         callers to guarantee reproducibility across runs.
        """
        self.client = client
        self.model_name = model_name
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name for this agent.

        Returns:
            A short string identifying the agent, e.g. ``"ClassifierAgent"``.
        """

    @abstractmethod
    def build_prompt(self, observation: dict) -> str:
        """Build the user-turn prompt from an environment observation.

        Args:
            observation: The dict returned by the environment's reset() or
                         step() call. Keys include ``issue_id``, ``title``,
                         ``body``, ``code_snippet``, ``existing_labels``, and
                         ``task_level``.

        Returns:
            A formatted string ready to be sent as the user message.
        """

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system-level instruction for this agent.

        Returns:
            A string that will be sent as the ``system`` role message in the
            chat completion request.
        """

    @abstractmethod
    def _parse_response(self, raw: str) -> AgentResult:
        """Parse raw LLM output into a structured AgentResult.

        Args:
            raw: The verbatim string returned by the LLM.

        Returns:
            An AgentResult populated with the agent's value, confidence,
            reasoning, raw_response, and agent_name.
        """

    # ------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------

    def run(self, observation: dict) -> AgentResult:
        """Execute a full agent cycle: build prompt → call LLM → parse result.

        This method is deliberately defensive: any exception during the LLM
        call (network error, rate limit, malformed response, etc.) is caught
        and a zero-confidence fallback AgentResult is returned so that the
        orchestrator can continue.

        Args:
            observation: The environment observation dict for the current step.

        Returns:
            An AgentResult — never raises an exception.
        """
        user_prompt = self.build_prompt(observation)
        system_prompt = self._get_system_prompt()
        raw_response = ""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_response = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            # Surface the exception text for diagnostics without propagating.
            raw_response = f"[LLM_ERROR] {type(exc).__name__}: {exc}"
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning="LLM call failed; returning fallback result.",
                raw_response=raw_response,
                agent_name=self.name,
            )

        try:
            return self._parse_response(raw_response)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Response parsing failed: {type(exc).__name__}: {exc}",
                raw_response=raw_response,
                agent_name=self.name,
            )

    def _keyword_fallback(self, observation: dict) -> AgentResult:
        """Classify an issue using simple keyword heuristics.

        Used when the LLM is unavailable or returns an unparseable response.
        The heuristic scans the concatenated title and body for known signal
        words and returns a low-confidence AgentResult.

        Heuristic priority order (first match wins):
            1. duplicate — "duplicate", "same as", "already filed", "#"
            2. feature   — "add", "support", "feature", "request",
                           "would be nice"
            3. bug       — "crash", "error", "fail", "broken", "not working"

        Args:
            observation: The environment observation dict containing at least
                         ``title`` and ``body``.

        Returns:
            An AgentResult with confidence=0.4 and the heuristic
            classification as its value. Never raises an exception.
        """
        try:
            title: str = observation.get("title", "") or ""
            body: str = observation.get("body", "") or ""
            text = (title + " " + body).lower()

            duplicate_signals = ["duplicate", "same as", "already filed", "#"]
            feature_signals = ["add", "support", "feature", "request", "would be nice"]
            bug_signals = ["crash", "error", "fail", "broken", "not working"]

            if any(kw in text for kw in duplicate_signals):
                classification = "duplicate"
                reasoning = "Keyword heuristic matched duplicate signal words."
            elif any(kw in text for kw in feature_signals):
                classification = "feature"
                reasoning = "Keyword heuristic matched feature-request signal words."
            elif any(kw in text for kw in bug_signals):
                classification = "bug"
                reasoning = "Keyword heuristic matched bug signal words."
            else:
                classification = "bug"
                reasoning = "No strong signal found; defaulting to bug classification."

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
                reasoning=f"Keyword fallback itself failed: {exc}",
                raw_response="[keyword_fallback_error]",
                agent_name=self.name,
            )


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def safe_json_parse(text: str) -> dict:
    """Extract and parse the first JSON object found in a string.

    Handles common LLM response patterns:
        - Bare JSON objects
        - JSON wrapped in markdown code fences (```json ... ```)
        - JSON embedded in surrounding prose

    Args:
        text: Any string that may contain a JSON object.

    Returns:
        A parsed dict if a valid JSON object was found; otherwise an empty
        dict ``{}``. Never raises an exception.
    """
    try:
        # Strip markdown code fences if present.
        stripped = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")

        # Find the first {...} block (greedy, dot-all so newlines are matched).
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return {}

        return json.loads(match.group())
    except Exception:  # noqa: BLE001
        return {}
