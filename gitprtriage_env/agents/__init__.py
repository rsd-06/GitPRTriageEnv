"""
Multi-agent triage system for DevTriageEnv.

Agents:
  ClassifierAgent   — classifies issue as bug/feature/duplicate
  BugLocatorAgent   — identifies exact bug line in code snippet
  TeamRouterAgent   — routes issue to webdev/devops/aiml team
  FixSuggesterAgent — generates a concrete one-sentence fix suggestion

Orchestrator:
  MultiAgentOrchestrator — coordinates all 4 agents in sequence,
                           passes upstream results as context downstream,
                           returns merged action dict
"""
from agents.base import AgentResult, BaseAgent, safe_json_parse
from agents.specialists import (
    ClassifierAgent,
    BugLocatorAgent,
    TeamRouterAgent,
    FixSuggesterAgent,
)
from agents.orchestrator import MultiAgentOrchestrator

__all__ = [
    "AgentResult",
    "BaseAgent",
    "safe_json_parse",
    "ClassifierAgent",
    "BugLocatorAgent",
    "TeamRouterAgent",
    "FixSuggesterAgent",
    "MultiAgentOrchestrator",
]
