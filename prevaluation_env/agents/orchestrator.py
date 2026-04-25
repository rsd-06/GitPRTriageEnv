"""
agents/orchestrator.py
----------------------
Multi-agent orchestrator for the PRRegressionAuditEnv review system.

The MultiAgentOrchestrator coordinates four specialist agents in a fixed
dependency pipeline:

    SafetyGateAgent    (step 1) → review_decision, blocker_type
    DefectLocatorAgent (step 2) → defect_category, faulty_line
    ReviewerRouterAgent(step 3) → reviewer_team     [Hard only]
    ReviewCommentAgent (step 4) → suggested_change  [Hard only]

Each step enriches the shared observation so downstream agents receive
upstream results as context. The final assembled dict maps 1:1 to
ReviewAction fields.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agents.base import AgentResult, BaseAgent
from agents.specialists import (
    DefectLocatorAgent,
    ReviewCommentAgent,
    ReviewerRouterAgent,
    SafetyGateAgent,
)


class MultiAgentOrchestrator:
    """Coordinates all four specialist agents to produce a complete ReviewAction.

    Pipeline dependency order:
        1. SafetyGateAgent     — review_decision + blocker_type
        2. DefectLocatorAgent  — defect_category + faulty_line (uses decision context)
        3. ReviewerRouterAgent — reviewer_team (uses defect context)
        4. ReviewCommentAgent  — suggested_change (uses faulty_line + team context)

    All agent results are stored in ``agent_trace`` for post-hoc analysis
    and reward attribution.

    Attributes:
        safety_gate:    The SafetyGateAgent instance.
        defect_locator: The DefectLocatorAgent instance.
        router:         The ReviewerRouterAgent instance.
        comment_agent:  The ReviewCommentAgent instance.
        agent_trace:    Ordered list of AgentResult objects from the current episode.
        episode_count:  Total number of episodes run since instantiation.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        temperature: float = 0.0,
    ) -> None:
        """Instantiate the orchestrator and all four specialist agents.

        Args:
            client:      An OpenAI-compatible client with chat.completions.create.
            model_name:  Model identifier forwarded to every agent.
            temperature: Sampling temperature (default 0.0 for determinism).
        """
        self.safety_gate = SafetyGateAgent(client, model_name, temperature)
        self.defect_locator = DefectLocatorAgent(client, model_name, temperature)
        self.router = ReviewerRouterAgent(client, model_name, temperature)
        self.comment_agent = ReviewCommentAgent(client, model_name, temperature)

        self.agent_trace: list[AgentResult] = []
        self.episode_count: int = 0

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, observation: dict) -> dict:
        """Execute the full four-agent review pipeline on one observation.

        Steps:
            A. Reset the per-episode trace and increment episode counter.
            B. Run SafetyGateAgent → review_decision + blocker_type.
            C. Enrich observation with decision context.
            D. Run DefectLocatorAgent → defect_category + faulty_line.
            E. Enrich observation with defect context.
            F. Run ReviewerRouterAgent → reviewer_team.
            G. Run ReviewCommentAgent → suggested_change.
            H. Assemble and return the final ReviewAction dict.

        Args:
            observation: The environment observation dict. Expected keys:
                         pr_id, title, description, proposed_code,
                         context_snippet, labels, task_level.

        Returns:
            A dict with all ReviewAction keys:
                review_decision, blocker_type, defect_category,
                faulty_line, reviewer_team, suggested_change.

        Note:
            Never raises. Unhandled exceptions return a safe fallback dict.
        """
        try:
            # --- Step A: reset trace ---
            self.agent_trace = []
            self.episode_count += 1

            # --- Step B: safety gate ---
            result_gate = self._run_agent(self.safety_gate, observation)
            self.agent_trace.append(result_gate)
            gate_value: dict = result_gate.value or {}
            review_decision = gate_value.get("review_decision") or "approve"
            blocker_type = gate_value.get("blocker_type")

            # --- Step C: enrich with decision context ---
            enriched_obs = observation.copy()
            enriched_obs["decision_context"] = review_decision

            # --- Step D: locate defect ---
            result_defect = self._run_agent(self.defect_locator, enriched_obs)
            self.agent_trace.append(result_defect)
            defect_value: dict = result_defect.value or {}
            defect_category = defect_value.get("defect_category")
            faulty_line = defect_value.get("faulty_line")

            # --- Step E: enrich with defect context ---
            enriched_obs["defect_context"] = str(defect_category or "unknown")
            enriched_obs["faulty_line_context"] = faulty_line

            # --- Step F: route to reviewer team ---
            result_team = self._run_agent(self.router, enriched_obs)
            self.agent_trace.append(result_team)
            reviewer_team = result_team.value

            # --- Step G: generate suggested change ---
            enriched_obs["reviewer_team_context"] = str(reviewer_team or "unknown")
            result_comment = self._run_agent(self.comment_agent, enriched_obs)
            self.agent_trace.append(result_comment)
            suggested_change = result_comment.value

            # --- Step H: assemble ReviewAction dict ---
            return {
                "review_decision": review_decision,
                "blocker_type": blocker_type,
                "defect_category": defect_category,
                "faulty_line": faulty_line,
                "reviewer_team": reviewer_team,
                "suggested_change": suggested_change,
            }

        except Exception as exc:  # noqa: BLE001
            # Safety net: environment must always receive a valid action dict.
            return {
                "review_decision": "approve",
                "blocker_type": None,
                "defect_category": None,
                "faulty_line": None,
                "reviewer_team": None,
                "suggested_change": None,
                "_orchestrator_error": str(exc),
            }

    # ------------------------------------------------------------------
    # Agent runner
    # ------------------------------------------------------------------

    def _run_agent(self, agent: BaseAgent, observation: dict) -> AgentResult:
        """Run a single agent with timing and automatic fallback on failure.

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
            A dict with keys: episode, agents_run, results, avg_confidence.
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
        """Return an action dict that nulls out low-confidence fields.

        Per-field confidence thresholds:
            review_decision  < 0.5  → force "approve" (low confidence = conservative)
            defect_category  < 0.4  → set to None
            faulty_line      < 0.4  → set to None
            reviewer_team    < 0.4  → set to None
            suggested_change < 0.3  → set to None

        Returns:
            Action dict with the same keys as run(), low-confidence fields nulled.
        """
        try:
            if len(self.agent_trace) < 4:
                return {
                    "review_decision": "approve",
                    "blocker_type": None,
                    "defect_category": None,
                    "faulty_line": None,
                    "reviewer_team": None,
                    "suggested_change": None,
                }

            # agent_trace order: gate, defect, router, comment
            result_gate: AgentResult = self.agent_trace[0]
            result_defect: AgentResult = self.agent_trace[1]
            result_team: AgentResult = self.agent_trace[2]
            result_comment: AgentResult = self.agent_trace[3]

            gate_value: dict = result_gate.value or {}
            review_decision = gate_value.get("review_decision") if result_gate.confidence >= 0.5 else "approve"
            blocker_type = gate_value.get("blocker_type") if result_gate.confidence >= 0.5 else None

            defect_value: dict = result_defect.value or {}
            defect_category: Optional[str] = defect_value.get("defect_category") if result_defect.confidence >= 0.4 else None
            faulty_line: Optional[int] = defect_value.get("faulty_line") if result_defect.confidence >= 0.4 else None

            reviewer_team: Optional[str] = result_team.value if result_team.confidence >= 0.4 else None
            suggested_change: Optional[str] = result_comment.value if result_comment.confidence >= 0.3 else None

            return {
                "review_decision": review_decision,
                "blocker_type": blocker_type,
                "defect_category": defect_category,
                "faulty_line": faulty_line,
                "reviewer_team": reviewer_team,
                "suggested_change": suggested_change,
            }

        except Exception as exc:  # noqa: BLE001
            return {
                "review_decision": "approve",
                "blocker_type": None,
                "defect_category": None,
                "faulty_line": None,
                "reviewer_team": None,
                "suggested_change": None,
                "_error": str(exc),
            }
