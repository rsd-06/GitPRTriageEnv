---
title: GitPRTriage Env
emoji: 🔍
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
tags:
  - openenv
  - developer-tools
  - code-review
  - multi-agent
  - reinforcement-learning
---

# GitPRTriage Env 🔍

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Problem Statement

Enterprise engineering teams spend thousands of collective hours annually triaging GitHub issue inboxes — classifying bugs vs feature requests vs duplicates, isolating the exact broken line across large codebases, routing to the right specialist team (DevOps vs WebDev vs AIML), and writing actionable fix suggestions. This is high-friction, repetitive expert work that compounds at scale.

GitPRTriageEnv models this as a structured RL benchmark: agents interact with an OpenEnv-compatible FastAPI environment that serves realistic GitHub issues, grades responses deterministically, and applies curriculum-based difficulty progression. The environment fills a gap in agentic benchmarks by targeting a genuine professional workflow rather than toy games.

Built for the **Meta × Scaler OpenEnv Hackathon 2026**.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **3-Task Benchmark** | Easy (classify), Medium (classify + bug line), Hard (full triage) |
| 🤖 **Multi-Agent System** | 4 specialist agents coordinated by an orchestrator |
| 📚 **Curriculum Learning** | Adaptive difficulty: Bootstrap → Intermediate → Advanced |
| 🛡️ **Anti-Reward-Hacking** | 4 independent guards preventing reward exploitation |
| 🔢 **45 Diverse Issues** | 15 per level across WebDev, DevOps, and AIML domains |
| 📊 **Live Monitoring** | 8 API endpoints including curriculum stats and guard audit |
| 🔄 **Deterministic Grading** | Same action on same issue always scores identically |

---

## Dataset & Diversification

The environment ships **45 hand-crafted GitHub issues** — 15 per difficulty level — spanning three engineering domains.

### Easy (15 issues) — Classification only, no code snippet

Binary scoring. Labels: 6 bugs, 5 features, 4 duplicates.

| Domain | Example topics |
|--------|---------------|
| Mobile crashes, web frontend, backend API | Auth/security, developer tooling |
| Analytics, infrastructure, integrations | UI/UX, configuration |

### Medium (15 issues) — Classification + exact bug line

All bugs. Every issue includes a numbered code snippet; the agent must identify the exact broken line.

| Domain | Issues |
|--------|--------|
| **WebDev (5)** | SQL injection, cookie security, Content-Type error, input validation, UTF-8 encoding |
| **DevOps (5)** | Dockerfile cache, env bool parsing, health check path, log rotation timezone, unpinned dependency |
| **AIML (5)** | CNN flatten missing, double normalization, label indexing, gradient accumulation, checkpoint saving order |

### Hard (15 issues) — Full triage: classify + bug line + team routing + fix suggestion

All bugs. Requires all four action fields to score maximum reward.

| Domain | Issues |
|--------|--------|
| **WebDev (5)** | XSS, CORS wildcard+credentials, DB connection leak, JWT alg:none, OAuth redirect loop |
| **DevOps (5)** | K8s liveness probe auth, CI/CD test bypass, multi-stage Docker copy, secrets in ENV layer, JVM heap OOMKilled |
| **AIML (5)** | Transformer pipeline inside handler, CUDA OOM in eval loop, gradient clip after optimizer.step, zero-initialized embeddings, data leakage before split |

---

## Environment Description

The RL loop follows a single-step episode structure:

```
POST /reset  →  agent receives observation  →  POST /step with action  →  reward returned
```

Each episode serves one issue. The agent submits exactly one action, receives its reward, and the episode ends.

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `issue_id` | string | Unique issue identifier |
| `title` | string | Issue title |
| `body` | string | Issue description |
| `code_snippet` | string or null | Numbered lines (medium/hard only) |
| `existing_labels` | list[string] | GitHub-style labels |
| `task_level` | string | `easy` / `medium` / `hard` |
| `done` | bool | Whether episode is complete |
| `reward` | float or null | Score after `step()`, null after `reset()` |

### Action Space

| Field | Type | Required | Valid Values |
|-------|------|----------|--------------|
| `classification` | string | Always | `bug`, `feature`, `duplicate` |
| `bug_line` | int or null | Medium + Hard | Line number (1-indexed) |
| `team` | string or null | Hard only | `webdev`, `devops`, `aiml` |
| `suggested_fix` | string or null | Hard only | One-sentence fix description |

---

## Task Definitions & Scoring

### Task 1 — Easy (Classification)
Binary score: **0.999** if correct, **0.001** otherwise.

### Task 2 — Medium (Bug Line Identification)
Partial credit available:

| Component | Points |
|-----------|--------|
| Correct classification | 0.40 |
| Exact bug line | 0.40 |
| Proximity (±1 line) | 0.20 |

### Task 3 — Hard (Full Triage)
Four equal components, each worth **0.25**:

| Component | Scoring detail |
|-----------|---------------|
| Classification | Correct = 0.25 |
| Bug line | Exact = 0.25, ±1 = 0.10 |
| Team routing | Correct = 0.25 |
| Suggested fix | 2+ keywords = 0.25, 1 keyword = 0.15, non-empty = 0.05 |

---

## Multi-Agent Architecture

Four specialist agents run in a dependency-ordered pipeline coordinated by the `MultiAgentOrchestrator`.

```
Issue Observation
      │
      ▼
┌─────────────────┐
│  Orchestrator   │
└────────┬────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  Step 1: ClassifierAgent                │
    │  Input: title + body + labels           │
    │  Output: bug / feature / duplicate      │
    └────┬────────────────────────────────────┘
         │ injects classification_context
    ┌────▼────────────────┐ ┌──────────────────────────┐
    │ Step 2:             │ │ Step 3:                  │
    │ BugLocatorAgent     │ │ TeamRouterAgent          │
    │ Input: code+body    │ │ Input: title+body+labels │
    │ Output: line number │ │ Output: webdev/devops/aiml│
    └────┬────────────────┘ └────────┬─────────────────┘
         └──────────┬────────────────┘
                    │ injects bug_line_context + team_context
         ┌──────────▼──────────────────────────────┐
         │  Step 4: FixSuggesterAgent              │
         │  Input: code + body + line + team       │
         │  Output: one-sentence fix suggestion    │
         └─────────────────────────────────────────┘
```

**Key design decisions:**
- Each agent receives **only** the fields relevant to its sub-task
- Upstream results are injected as context for downstream agents
- Each agent has a focused system prompt + keyword fallback for when LLM is unavailable
- Low-confidence outputs are nulled rather than guessed
- Orchestrator never crashes — all exceptions fall back to keyword heuristics

**Running modes:**

```bash
python inference.py --mode single    # baseline single-agent
python inference.py --mode multi     # multi-agent orchestrator
python inference.py --mode compare   # side-by-side comparison
python inference.py --mode multi --episodes 30 --verbose
```

---

## Curriculum Learning

The `CurriculumSampler` manages difficulty progression across three phases using a rolling performance window.

```
Phase 1: BOOTSTRAP          Phase 2: INTERMEDIATE       Phase 3: ADVANCED
Easy:   70%                 Easy:   20%                 Easy:   10%
Medium: 20%           →     Medium: 60%           →     Medium: 30%
Hard:   10%                 Hard:   20%                 Hard:   60%

Trigger: easy avg ≥ 0.80    Trigger: medium avg ≥ 0.65  (terminal phase)
         over 10 episodes            over 10 episodes
```

- **Anti-repetition:** the same issue is never served twice in a row
- Phase transitions are logged with episode number and trigger value
- Live state is exposed via `GET /curriculum`

---

## Anti-Reward-Hacking Guards

The `GuardSuite` post-processes every reward after grading without touching any grading logic. Penalties are multiplicative and the final reward is clamped to `[0.001, 0.999]`.

| Guard | What It Catches | Penalty |
|-------|----------------|---------|
| `KeywordStuffingDetector` | Fix suggestions with >40% keyword density | 0.50–0.90× scaled |
| `RepetitionDetector` | Same action fingerprint repeated >3× in 10 episodes | 0.50–0.90× scaled |
| `FixQualityValidator` | Fix with <4 words or no action verb | 0.70–0.80× |
| `TimingGuard` | Response under 200ms (possible cached output) | 0.95× (soft) |

All guard firings are logged to `GET /guards/audit` with before/after reward, penalty multiplier, and human-readable reason.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check landing page |
| `/health` | GET | Returns status + issue count |
| `/reset` | POST | Start new episode, returns observation |
| `/step` | POST | Submit action, returns observation + reward |
| `/state` | GET | Current episode metadata |
| `/tasks` | GET | List all 3 tasks with descriptions |
| `/curriculum` | GET | Live curriculum phase + performance stats |
| `/audit` | GET | Last N episode records (level + reward) |
| `/agents/info` | GET | Full multi-agent pipeline architecture |
| `/guards` | GET | Guard suite statistics + penalty rate |
| `/guards/audit` | GET | Guard firing log with before/after rewards |

---

## Setup and Usage

**Prerequisites:** Python 3.11, Docker

### Running Locally

```bash
# Set environment variables
export HF_TOKEN="your_groq_api_key"
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export ENV_URL="http://localhost:7860"

# Install and start
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Running via Docker

```bash
docker build -t gitprtriage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_groq_key \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.3-70b-versatile \
  gitprtriage-env
```

### Running Inference

```bash
python inference.py --mode multi --episodes 60
python inference.py --mode compare
```

### Project Structure

```
gitprtriage_env/
├── agents/
│   ├── __init__.py
│   ├── base.py          # AgentResult + BaseAgent ABC
│   ├── specialists.py   # 4 specialist agents
│   └── orchestrator.py  # MultiAgentOrchestrator
├── server/
│   ├── app.py           # FastAPI app + all endpoints
│   ├── environment.py   # Core RL environment
│   ├── curriculum.py    # CurriculumSampler (3-phase)
│   └── guards.py        # Anti-reward-hacking guards
├── data/
│   └── issues.json      # 45 diverse GitHub issues
├── models.py            # Pydantic schemas
├── inference.py         # Single/multi/compare modes
├── Dockerfile
└── requirements.txt
```

---

## Baseline Results

Evaluated using `llama-3.3-70b-versatile` via Groq API, 60-episode automated run, chain-of-thought reasoning enabled.

| Task | Mode | Avg Score | Std | Episodes |
|------|------|-----------|-----|----------|
| Easy | Single-Agent | 0.890 | 0.287 | 20 |
| Easy | Multi-Agent | 0.940 | 0.211 | 20 |
| Medium | Single-Agent | 0.612 | 0.198 | 20 |
| Medium | Multi-Agent | 0.731 | 0.156 | 20 |
| Hard | Single-Agent | 0.421 | 0.187 | 20 |
| Hard | Multi-Agent | 0.583 | 0.163 | 20 |

Multi-agent mode outperforms single-agent across all difficulty levels, with the largest gain on Hard (+16.2 points) where task decomposition matters most.

---

## License

MIT © 2026 rsd-06
