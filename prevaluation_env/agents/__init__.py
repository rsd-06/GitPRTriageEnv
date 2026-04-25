"""
Multi-agent PR review system for PRRegressionAuditEnv.

Agents:
    SafetyGateAgent      — review_decision + blocker_type (Easy)
    DefectLocatorAgent   — defect_category + faulty_line (Medium + Hard)
    ReviewerRouterAgent  — reviewer_team routing (Hard only)
    ReviewCommentAgent   — concise suggested_change under 200 chars (Hard only)

Orchestrator:
    MultiAgentOrchestrator — coordinates all 4 agents in dependency order,
                             passes upstream results as context downstream,
                             returns a dict matching ReviewAction fields.
"""
from agents.base import AgentResult, BaseAgent, safe_json_parse
from agents.specialists import (
    SafetyGateAgent,
    DefectLocatorAgent,
    ReviewerRouterAgent,
    ReviewCommentAgent,
)
from agents.orchestrator import MultiAgentOrchestrator

__all__ = [
    "AgentResult",
    "BaseAgent",
    "safe_json_parse",
    "SafetyGateAgent",
    "DefectLocatorAgent",
    "ReviewerRouterAgent",
    "ReviewCommentAgent",
    "MultiAgentOrchestrator",
]
