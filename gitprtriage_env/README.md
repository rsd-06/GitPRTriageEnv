---
title: DevTriage Environment
emoji: 🔍
colorFrom: blue
colorTo: orange
sdk: docker
pinned: false
tags:
  - openenv
  - developer-tools
  - code-review
---

# DevTriageEnv

A GitHub issue triage environment where an LLM agent classifies issues, identifies bug lines in code, routes to the correct engineering team, and suggests fixes.

Built for the **Meta × Scaler OpenEnv Hackathon 2026**.

## Environment Description

This environment models a genuine daily developer task: triaging a GitHub inbox. Developers spend significant time classifying issues, spotting bugs in code snippets, routing to the right team, and writing fix suggestions. DevTriageEnv turns this into a structured RL problem with 30 mock GitHub issues and deterministic graders.

## Action Space

| Field | Type | Required | Valid Values |
|-------|------|----------|--------------|
| `classification` | string | Always | `bug`, `feature`, `duplicate` |
| `bug_line` | int or null | Medium + Hard | Line number (1-indexed) |
| `team` | string or null | Hard | `webdev`, `devops`, `aiml` |
| `suggested_fix` | string or null | Hard | One-sentence fix description |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `issue_id` | string | Unique issue identifier |
| `title` | string | Issue title |
| `body` | string | Issue description |
| `code_snippet` | string or null | Code snippet (present for medium/hard) |
| `existing_labels` | list[string] | Labels already on the issue |
| `task_level` | string | `easy`, `medium`, or `hard` |
| `done` | bool | Whether episode is complete |
| `reward` | float or null | Score after step(), null after reset() |

## Tasks

**Task 1 — Issue Classification (Easy)**
Classify the issue as `bug`, `feature`, or `duplicate`. Score: 1.0 if correct, 0.0 otherwise.

**Task 2 — Bug Line Identification (Medium)**
Classify the issue and identify the exact line containing the bug. Score: 0.40 (classification) + 0.40 (exact line) + 0.20 (proximity ±1 line).

**Task 3 — Full Triage + Fix Suggestion (Hard)**
Classify, identify bug line, route to correct team, and suggest a concrete fix. Score: 0.25 per component. Fix scored by keyword relevance, not length.

## Baseline Scores

Evaluated using `meta-llama/Llama-3.1-8B-Instruct` via Groq inference API over 38 episodes.

| Task | Average Score | Std | Episodes |
|------|---------------|-----|----------|
| Easy | 1.000 | 0.000 | 14 |
| Medium | 0.523 | 0.130 | 13 |
| Hard | 0.795 | 0.147 | 11 |

> Easy std=0.000 reflects that classification-only tasks are consistently solved by the baseline model. Variance appears in medium and hard tasks where partial credit graders are active.

## Setup and Usage

### Prerequisites
- Python 3.11
- Docker (for containerized deployment)

### Running Locally

1. Set environment variables:
```bash
   export HF_TOKEN="gsk_yourGroqKeyHere"
   export API_BASE_URL="https://api.groq.com/openai/v1"
   export MODEL_NAME="llama-3.1-8b-instant"
   export ENV_URL="http://localhost:7860"
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Start the server:
```bash
   uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

4. Visit `http://localhost:7860/docs` to explore the API interactively.

### Running via Docker

```bash
docker build -t dev-triage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_key \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.1-8b-instant \
  dev-triage-env
```

### Running Baseline Inference

```bash
python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status": "healthy"}` |
| `/reset` | POST | Start new episode, returns initial observation |
| `/step` | POST | Submit action, returns observation + reward |
| `/state` | GET | Current episode metadata |
| `/tasks` | GET | List all 3 tasks |
| `/docs` | GET | Interactive Swagger UI |
