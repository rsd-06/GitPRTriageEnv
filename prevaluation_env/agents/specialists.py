"""
agents/specialists.py
---------------------
Specialist agent implementations for the PRRegressionAuditEnv multi-agent review system.

Four agents are defined here, each responsible for one component of the final
ReviewAction submitted to the environment:

    SafetyGateAgent    — review_decision + blocker_type (Easy)
    DefectLocatorAgent — defect_category + faulty_line (Medium + Hard)
    ReviewerRouterAgent— reviewer_team (Hard)
    ReviewCommentAgent — suggested_change (Hard)

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

_VALID_DECISIONS: frozenset[str] = frozenset({"approve", "request_changes"})
_VALID_BLOCKER_TYPES: frozenset[Optional[str]] = frozenset({
    "debug_output", "hardcoded_secret", "do_not_merge_comment",
    "debug_test_bypass", "syntax_error", None
})
_VALID_DEFECT_CATEGORIES: frozenset[Optional[str]] = frozenset({
    "security", "logic", "performance", None
})
_VALID_TEAMS: frozenset[Optional[str]] = frozenset({
    "infosec", "devops", "core-frontend", "core-sysdev", "aiml", None
})


def _clamp(value: float) -> float:
    """Clamp a float confidence value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# 1. SafetyGateAgent
# ---------------------------------------------------------------------------

class SafetyGateAgent(BaseAgent):
    """Makes the binary review decision and identifies obvious blockers.

    For Easy tasks: decides approve vs request_changes and names the blocker
    type if one is present (debug_output, hardcoded_secret, etc.).
    """

    @property
    def name(self) -> str:
        return "SafetyGateAgent"

    def _get_system_prompt(self) -> str:
        return (
            "You are a senior code reviewer performing an initial safety scan of a Pull Request. "
            "Your job: decide if the PR is safe to pass to reviewers, or if it has an obvious blocker. "
            "Respond ONLY with valid JSON: "
            '{"review_decision": "approve"|"request_changes", '
            '"blocker_type": "debug_output"|"hardcoded_secret"|"do_not_merge_comment"|"debug_test_bypass"|"syntax_error"|null, '
            '"confidence": 0.0-1.0, "reasoning": "one sentence"}. '
            "Rules: "
            "debug_output=print/console.log of secrets or auth data left in; "
            "hardcoded_secret=literal API key, password, or token in code; "
            "do_not_merge_comment=TODO: DO NOT MERGE or WIP HACK comment in code; "
            "debug_test_bypass=hard-wired condition like 'or True' bypassing auth or gates; "
            "syntax_error=unmatched bracket, invalid syntax that breaks the file. "
            "Set blocker_type to null if the PR is clean. "
            "Output raw JSON only, no markdown."
        )

    def build_prompt(self, observation: dict) -> str:
        title: str = observation.get("title", "") or ""
        description: str = observation.get("description", "") or ""
        proposed_code: Optional[str] = observation.get("proposed_code")
        labels: list = observation.get("labels", []) or []
        labels_str = ", ".join(labels) if labels else "none"

        parts = [
            f"PR Title: {title}",
            f"Description: {description}",
            f"Labels: {labels_str}",
        ]
        if proposed_code:
            parts.append(f"\nProposed Code (1-indexed lines):\n{proposed_code}")
        parts.append("\nScan this PR for obvious blockers and decide approve or request_changes.")
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> AgentResult:
        try:
            data: dict = safe_json_parse(raw)
            raw_decision = data.get("review_decision")
            decision: Optional[str] = raw_decision if raw_decision in _VALID_DECISIONS else None
            raw_blocker = data.get("blocker_type")
            blocker: Optional[str] = raw_blocker if raw_blocker in _VALID_BLOCKER_TYPES else None
            confidence: float = _clamp(float(data.get("confidence", 0.5)))
            reasoning: str = str(data.get("reasoning", ""))
            return AgentResult(
                value={"review_decision": decision, "blocker_type": blocker},
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw,
                agent_name=self.name,
            )
        except Exception as exc:
            return AgentResult(
                value={"review_decision": None, "blocker_type": None},
                confidence=0.0,
                reasoning=f"Parsing failed: {exc}",
                raw_response=raw,
                agent_name=self.name,
            )

    def _keyword_fallback(self, observation: dict) -> AgentResult:
        try:
            code: str = (observation.get("proposed_code") or "").lower()
            title: str = (observation.get("title") or "").lower()

            debug_signals = ["console.log", "print(", "# debug", "// debug", "alert("]
            secret_signals = ["sk_live_", "sk_test_", "aws_secret", "api_key =", "password ="]
            no_merge_signals = ["do not merge", "do_not_merge", "wip hack", "// wip"]
            bypass_signals = ["or true", "or True", "if true:", "== true # bypass"]
            syntax_signals = ["syntax_error", "def (\n", "(;", "[;"]

            if any(s in code for s in no_merge_signals):
                blocker = "do_not_merge_comment"
            elif any(s in code for s in secret_signals):
                blocker = "hardcoded_secret"
            elif any(s in code for s in bypass_signals):
                blocker = "debug_test_bypass"
            elif any(s in code for s in debug_signals):
                blocker = "debug_output"
            else:
                blocker = None

            decision = "request_changes" if blocker else "approve"
            return AgentResult(
                value={"review_decision": decision, "blocker_type": blocker},
                confidence=0.4,
                reasoning="Keyword heuristic safety scan.",
                raw_response="[keyword_fallback]",
                agent_name=self.name,
            )
        except Exception as exc:
            return AgentResult(
                value={"review_decision": "approve", "blocker_type": None},
                confidence=0.0,
                reasoning=f"Keyword fallback failed: {exc}",
                raw_response="[keyword_fallback_error]",
                agent_name=self.name,
            )


# ---------------------------------------------------------------------------
# 2. DefectLocatorAgent
# ---------------------------------------------------------------------------

class DefectLocatorAgent(BaseAgent):
    """Identifies the defect category and exact faulty line in proposed_code.

    Used for Medium (proposed_code only) and Hard (proposed_code + context_snippet).
    The defect category is one of: security, logic, performance.
    """

    @property
    def name(self) -> str:
        return "DefectLocatorAgent"

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert code reviewer identifying defects in Pull Request code. "
            "Respond ONLY with valid JSON: "
            '{"defect_category": "security"|"logic"|"performance", '
            '"faulty_line": integer, "confidence": 0.0-1.0, '
            '"reasoning": "one sentence explaining what is wrong on that line"}. '
            "Categories: "
            "security=vulnerability that can be exploited (injection, insecure config, data exposure, auth bypass); "
            "logic=incorrect behavior (wrong condition, wrong operator, off-by-one, wrong order, data leakage); "
            "performance=resource inefficiency (unnecessary load, repeated work, memory accumulation, wasted compute). "
            "faulty_line is the 1-indexed line number in the Proposed Code where the defect lives. "
            "For Hard tasks, look at how the proposed code interacts with the Context Snippet to identify the defect. "
            "Output raw JSON only, no markdown."
        )

    def build_prompt(self, observation: dict) -> str:
        description: str = observation.get("description", "") or ""
        proposed_code: Optional[str] = observation.get("proposed_code")
        context_snippet: Optional[str] = observation.get("context_snippet")
        decision_context: str = observation.get("decision_context") or "request_changes"

        parts = [
            f"PR Description: {description}",
            f"Review Decision (upstream): {decision_context}",
        ]
        if proposed_code:
            parts.append(f"\nProposed Code (1-indexed):\n{proposed_code}")
        if context_snippet:
            parts.append(
                f"\nContext Snippet (existing code/config this PR interacts with):\n{context_snippet}"
            )
        parts.append(
            "\nIdentify the defect category and the exact 1-indexed line in Proposed Code containing the flaw."
        )
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> AgentResult:
        try:
            data: dict = safe_json_parse(raw)
            raw_cat = data.get("defect_category")
            category: Optional[str] = raw_cat if raw_cat in _VALID_DEFECT_CATEGORIES else None
            raw_line = data.get("faulty_line")
            faulty_line: Optional[int] = None
            if raw_line is not None:
                try:
                    candidate = int(raw_line)
                    if candidate > 0:
                        faulty_line = candidate
                except (TypeError, ValueError):
                    pass
            confidence: float = _clamp(float(data.get("confidence", 0.5)))
            reasoning: str = str(data.get("reasoning", ""))
            return AgentResult(
                value={"defect_category": category, "faulty_line": faulty_line},
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw,
                agent_name=self.name,
            )
        except Exception as exc:
            return AgentResult(
                value={"defect_category": None, "faulty_line": None},
                confidence=0.0,
                reasoning=f"Parsing failed: {exc}",
                raw_response=raw,
                agent_name=self.name,
            )

    def _keyword_fallback(self, observation: dict) -> AgentResult:
        try:
            code: str = (observation.get("proposed_code") or "").lower()
            context: str = (observation.get("context_snippet") or "").lower()
            text = code + " " + context

            security_signals = ["inject", "exec(", "eval(", "f\"select", "secure=false", "algorithm", "authorization"]
            performance_signals = ["pipeline(", "load_model", "/ 255", "loss +=", "fits_transform"]

            if any(s in text for s in security_signals):
                category = "security"
            elif any(s in text for s in performance_signals):
                category = "performance"
            else:
                category = "logic"

            return AgentResult(
                value={"defect_category": category, "faulty_line": None},
                confidence=0.3,
                reasoning="Keyword heuristic defect classification.",
                raw_response="[keyword_fallback]",
                agent_name=self.name,
            )
        except Exception as exc:
            return AgentResult(
                value={"defect_category": None, "faulty_line": None},
                confidence=0.0,
                reasoning=f"Keyword fallback failed: {exc}",
                raw_response="[keyword_fallback_error]",
                agent_name=self.name,
            )


# ---------------------------------------------------------------------------
# 3. ReviewerRouterAgent
# ---------------------------------------------------------------------------

class ReviewerRouterAgent(BaseAgent):
    """Routes the flagged PR to the correct expert reviewer team.

    Valid teams: infosec, devops, core-frontend, core-sysdev, aiml.
    Used for Hard tasks only. Receives defect_category context from upstream.
    """

    @property
    def name(self) -> str:
        return "ReviewerRouterAgent"

    def _get_system_prompt(self) -> str:
        return (
            "You are a senior engineering lead routing a flagged Pull Request to the correct expert team. "
            "Respond ONLY with valid JSON: "
            '{"reviewer_team": "infosec"|"devops"|"core-frontend"|"core-sysdev"|"aiml", '
            '"confidence": 0.0-1.0, "reasoning": "one sentence"}. '
            "Team responsibilities: "
            "infosec=authentication, authorization, JWT, secrets management, XSS, injection, crypto; "
            "devops=Docker, Kubernetes, CI/CD pipelines, GitHub Actions, build systems, monitoring; "
            "core-frontend=frontend code, REST API routes, CORS, middleware, web auth flows, templates; "
            "core-sysdev=backend/system logic, database connections, ORM, connection pooling, server-side validation; "
            "aiml=ML models, training loops, GPU/CUDA, data pipelines, model loading, embeddings. "
            "Output raw JSON only, no markdown."
        )

    def build_prompt(self, observation: dict) -> str:
        title: str = observation.get("title", "") or ""
        proposed_code: Optional[str] = observation.get("proposed_code")
        context_snippet: Optional[str] = observation.get("context_snippet")
        defect_context: str = observation.get("defect_context") or "unknown"
        labels: list = observation.get("labels", []) or []
        labels_str = ", ".join(labels) if labels else "none"

        parts = [
            f"PR Title: {title}",
            f"Labels: {labels_str}",
            f"Defect Category (upstream): {defect_context}",
        ]
        if proposed_code:
            parts.append(f"\nProposed Code:\n{proposed_code}")
        if context_snippet:
            parts.append(f"\nContext Snippet:\n{context_snippet}")
        parts.append("\nRoute this PR to the correct expert reviewer team.")
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> AgentResult:
        try:
            data: dict = safe_json_parse(raw)
            raw_team = data.get("reviewer_team")
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
        except Exception as exc:
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Parsing failed: {exc}",
                raw_response=raw,
                agent_name=self.name,
            )

    def _keyword_fallback(self, observation: dict) -> AgentResult:
        try:
            title: str = (observation.get("title") or "").lower()
            code: str = (observation.get("proposed_code") or "").lower()
            context: str = (observation.get("context_snippet") or "").lower()
            defect: str = (observation.get("defect_context") or "").lower()
            text = title + " " + code + " " + context

            infosec_signals = ["jwt", "auth", "secret", "token", "password", "injection", "xss", "cors"]
            devops_signals = ["docker", "kubernetes", "k8s", "ci/cd", "github actions", "deploy", "dockerfile"]
            aiml_signals = ["model", "training", "cuda", "gpu", "pipeline(", "torch", "sklearn", "epoch"]
            frontend_signals = ["cors", "template", "oauth", "redirect", "middleware", "route"]

            if any(s in text for s in infosec_signals) or "security" in defect:
                team = "infosec"
            elif any(s in text for s in devops_signals):
                team = "devops"
            elif any(s in text for s in aiml_signals):
                team = "aiml"
            elif any(s in text for s in frontend_signals):
                team = "core-frontend"
            else:
                team = "core-sysdev"

            return AgentResult(
                value=team,
                confidence=0.4,
                reasoning="Keyword heuristic team routing.",
                raw_response="[keyword_fallback]",
                agent_name=self.name,
            )
        except Exception as exc:
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Keyword fallback failed: {exc}",
                raw_response="[keyword_fallback_error]",
                agent_name=self.name,
            )


# ---------------------------------------------------------------------------
# 4. ReviewCommentAgent
# ---------------------------------------------------------------------------

class ReviewCommentAgent(BaseAgent):
    """Generates a concise, technically specific suggested code change.

    The suggestion must be one sentence and under 200 characters to prevent
    keyword stuffing. Used for Hard tasks only.
    """

    @property
    def name(self) -> str:
        return "ReviewCommentAgent"

    def _get_system_prompt(self) -> str:
        return (
            "You are a senior engineer writing a precise, actionable code review comment. "
            "Respond ONLY with valid JSON: "
            '{"suggested_change": "one concrete sentence under 200 characters describing exactly what to change and why", '
            '"confidence": 0.0-1.0, "reasoning": "one sentence"}. '
            "Rules: "
            "Your suggested_change must be UNDER 200 characters total. "
            "Name the exact function, variable, line, or config key to change. "
            "Do not just say 'fix the bug' — specify the technical correction. "
            "Output raw JSON only, no markdown."
        )

    def build_prompt(self, observation: dict) -> str:
        description: str = observation.get("description", "") or ""
        proposed_code: Optional[str] = observation.get("proposed_code")
        context_snippet: Optional[str] = observation.get("context_snippet")
        faulty_line: Optional[int] = observation.get("faulty_line_context")
        team: Optional[str] = observation.get("reviewer_team_context")

        parts = [f"PR Description: {description}"]
        if proposed_code:
            parts.append(f"\nProposed Code:\n{proposed_code}")
        if context_snippet:
            parts.append(f"\nContext Snippet:\n{context_snippet}")

        line_text = str(faulty_line) if faulty_line is not None else "unknown"
        team_text = team if team else "unknown"
        parts.append(
            f"\nThe defect is on line {line_text} of the proposed code. Assigned to team: {team_text}."
        )
        parts.append("Write a concise one-sentence suggested change under 200 characters.")
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> AgentResult:
        try:
            data: dict = safe_json_parse(raw)
            suggestion: Optional[str] = data.get("suggested_change") or None
            if suggestion is not None:
                suggestion = suggestion.strip() or None
            confidence: float = _clamp(float(data.get("confidence", 0.5)))
            reasoning: str = str(data.get("reasoning", ""))
            return AgentResult(
                value=suggestion,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw,
                agent_name=self.name,
            )
        except Exception as exc:
            return AgentResult(
                value=None,
                confidence=0.0,
                reasoning=f"Parsing failed: {exc}",
                raw_response=raw,
                agent_name=self.name,
            )

    def _keyword_fallback(self, observation: dict) -> AgentResult:
        return AgentResult(
            value=None,
            confidence=0.0,
            reasoning="Suggested change unavailable without LLM.",
            raw_response="[keyword_fallback]",
            agent_name=self.name,
        )
