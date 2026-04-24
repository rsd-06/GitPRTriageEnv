"""
agents/orchestrator.py
----------------------
Multi-agent orchestrator for the GitPRTriageEnv triage system.

The MultiAgentOrchestrator coordinates all four specialist agents in a
fixed pipeline: classify → (locate bug + route team in parallel) →
suggest fix. Each step enriches the shared observation dict so that
downstream agents have context from upstream decisions.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agents.base import AgentResult, BaseAgent
from agents.specialists import (
    BugLocatorAgent,
    ClassifierAgent,
    FixSuggesterAgent,
    TeamRouterAgent,
)


class MultiAgentOrchestrator:
    """Coordinates all four specialist agents to produce a complete triage action.

    The pipeline runs in dependency order:

    1. ``ClassifierAgent``   — determines bug / feature / duplicate
    2. ``BugLocatorAgent``   — identifies the exact bug line (uses classification)
    2. ``TeamRouterAgent``   — selects the responsible team (uses classification)
    3. ``FixSuggesterAgent`` — proposes a concrete fix (uses line + team)

    All agent results are stored in ``agent_trace`` for post-hoc analysis.

    Attributes:
        classifier:    The ClassifierAgent instance.
        bug_locator:   The BugLocatorAgent instance.
        team_router:   The TeamRouterAgent instance.
        fix_suggester: The FixSuggesterAgent instance.
        agent_trace:   Ordered list of AgentResult objects from the current episode.
        episode_count: Total number of episodes run since instantiation.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        temperature: float = 0.0,
    ) -> None:
        """Instantiate the orchestrator and all four specialist agents.

        Args:
            client:      An OpenAI-compatible client exposing
                         ``client.chat.completions.create``.
            model_name:  Model identifier string forwarded to every agent.
            temperature: Sampling temperature forwarded to every agent.
                         Pass 0.0 (the default) for fully deterministic runs.
        """
        self.classifier = ClassifierAgent(client, model_name, temperature)
        self.bug_locator = BugLocatorAgent(client, model_name, temperature)
        self.team_router = TeamRouterAgent(client, model_name, temperature)
        self.fix_suggester = FixSuggesterAgent(client, model_name, temperature)

        self.agent_trace: list[AgentResult] = []
        self.episode_count: int = 0

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, observation: dict) -> dict:
        """Execute the full four-agent triage pipeline on one observation.

        Steps:
            A. Reset the per-episode trace and increment the episode counter.
            B. Run ClassifierAgent.
            C. Enrich the observation with the classification result.
            D. Run BugLocatorAgent and TeamRouterAgent (both use classification context).
            E. Enrich the observation with bug line and team results.
            F. Run FixSuggesterAgent (uses line + team context).
            G. Assemble and return the final action dict.

        Args:
            observation: The environment observation dict for the current step.
                         Expected keys: title, body, code_snippet, existing_labels,
                         task_level. Additional context keys are injected internally.

        Returns:
            A dict with keys ``classification``, ``bug_line``, ``team``,
            and ``suggested_fix``, ready to submit to the environment's step().

        Note:
            This method never raises. Any unhandled internal exception causes
            a safe fallback action to be returned with all optional fields nulled.
        """
        try:
            # --- Step A: reset trace ---
            self.agent_trace = []
            self.episode_count += 1

            # --- Step B: classify ---
            result_clf = self._run_agent(self.classifier, observation)
            self.agent_trace.append(result_clf)

            # --- Step C: enrich with classification ---
            enriched_obs = observation.copy()
            enriched_obs["classification_context"] = str(result_clf.value or "unknown")

            # --- Step D: locate bug line + route team ---
            result_line = self._run_agent(self.bug_locator, enriched_obs)
            result_team = self._run_agent(self.team_router, enriched_obs)
            self.agent_trace.append(result_line)
            self.agent_trace.append(result_team)

            # --- Step E: enrich with line + team ---
            enriched_obs["bug_line_context"] = result_line.value
            enriched_obs["team_context"] = str(result_team.value or "unknown")

            # --- Step F: suggest fix ---
            result_fix = self._run_agent(self.fix_suggester, enriched_obs)
            self.agent_trace.append(result_fix)

            # --- Step G: assemble action ---
            return {
                "classification": result_clf.value or "bug",
                "bug_line": result_line.value,       # int or None
                "team": result_team.value,            # str or None
                "suggested_fix": result_fix.value,    # str or None
            }

        except Exception as exc:  # noqa: BLE001
            # Safety net: environment must always receive a valid action dict.
            return {
                "classification": "bug",
                "bug_line": None,
                "team": None,
                "suggested_fix": None,
                "_orchestrator_error": str(exc),
            }

    # ------------------------------------------------------------------
    # Agent runner
    # ------------------------------------------------------------------

    def _run_agent(self, agent: BaseAgent, observation: dict) -> AgentResult:
        """Run a single agent with timing and automatic fallback on failure.

        Wraps ``agent.run()`` in a try/except. If the call raises (which
        should not happen given the base class guarantees, but can occur in
        exotic failure modes), ``agent._keyword_fallback()`` is called instead.

        The result is printed to stdout in a structured log line.

        Args:
            agent:       The specialist agent to invoke.
            observation: The (potentially enriched) observation dict.

        Returns:
            The AgentResult produced by the agent or its keyword fallback.
        """
        start_time = time.perf_counter()
        try:
            result = agent.run(observation)
        except Exception:  # noqa: BLE001
            result = agent._keyword_fallback(observation)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print(
            f"[{agent.name}] value={result.value!r} "
            f"confidence={result.confidence:.2f} "
            f"({elapsed_ms:.0f}ms)"
        )
        return result

    # ------------------------------------------------------------------
    # Trace utilities
    # ------------------------------------------------------------------

    def get_trace_summary(self) -> dict:
        """Return a structured summary of the most recent episode's agent trace.

        Useful for logging, debugging, and RL reward attribution.

        Returns:
            A dict with keys:
                - ``episode``        — current episode number (int)
                - ``agents_run``     — number of agent results in the trace (int)
                - ``results``        — list of per-agent dicts with agent, value,
                                       confidence, and reasoning
                - ``avg_confidence`` — mean confidence across all agents (float)

        Note:
            Call ``run()`` before this method. Returns zero counts if the trace
            is empty. Never raises an exception.
        """
        try:
            results = [
                {
                    "agent": r.agent_name,
                    "value": r.value,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                }
                for r in self.agent_trace
            ]
            avg_confidence = (
                round(
                    sum(r.confidence for r in self.agent_trace) / len(self.agent_trace),
                    3,
                )
                if self.agent_trace
                else 0.0
            )
            return {
                "episode": self.episode_count,
                "agents_run": len(self.agent_trace),
                "results": results,
                "avg_confidence": avg_confidence,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "episode": self.episode_count,
                "agents_run": 0,
                "results": [],
                "avg_confidence": 0.0,
                "_error": str(exc),
            }

    def get_confidence_weighted_action(self) -> dict:
        """Return a conservative action dict that nulls out low-confidence fields.

        Applies per-field confidence thresholds after a ``run()`` call:

        - ``classification`` < 0.5  → apply keyword fallback from ClassifierAgent
        - ``bug_line``       < 0.4  → set to None (abstain rather than guess)
        - ``team``           < 0.4  → set to None
        - ``suggested_fix``  < 0.3  → set to None

        This trades recall for precision on ambiguous inputs — useful when the
        downstream grader rewards abstaining over wrong answers.

        Returns:
            An action dict with the same shape as ``run()``'s return value, but
            with low-confidence fields replaced by None (or the fallback
            classification).

        Note:
            Requires ``run()`` to have been called first; uses ``self.agent_trace``.
            Never raises an exception.
        """
        try:
            if len(self.agent_trace) < 4:
                # Trace is incomplete — cannot derive weighted action.
                return {
                    "classification": "bug",
                    "bug_line": None,
                    "team": None,
                    "suggested_fix": None,
                }

            # agent_trace is always appended in order: clf, line, team, fix.
            result_clf: AgentResult = self.agent_trace[0]
            result_line: AgentResult = self.agent_trace[1]
            result_team: AgentResult = self.agent_trace[2]
            result_fix: AgentResult = self.agent_trace[3]

            # Classification: fall back to keyword heuristic when uncertain.
            if result_clf.confidence < 0.5:
                # Reconstruct a minimal observation from whatever the trace holds.
                # The orchestrator does not store the raw observation, so we
                # build the fallback from the classifier's reasoning as context.
                fallback = self.classifier._keyword_fallback(
                    {"title": result_clf.reasoning, "body": "", "existing_labels": []}
                )
                classification: str = fallback.value or "bug"
            else:
                classification = result_clf.value or "bug"

            bug_line: Optional[int] = result_line.value if result_line.confidence >= 0.4 else None
            team: Optional[str] = result_team.value if result_team.confidence >= 0.4 else None
            suggested_fix: Optional[str] = result_fix.value if result_fix.confidence >= 0.3 else None

            return {
                "classification": classification,
                "bug_line": bug_line,
                "team": team,
                "suggested_fix": suggested_fix,
            }

        except Exception as exc:  # noqa: BLE001
            return {
                "classification": "bug",
                "bug_line": None,
                "team": None,
                "suggested_fix": None,
                "_error": str(exc),
            }
