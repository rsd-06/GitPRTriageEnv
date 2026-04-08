---
title: DevTriage Environment
emoji: 🔍
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
tags:
  - openenv
  - developer-tools
  - code-review
---

# GitPRTriage Env

A GitHub issue triage and issue routing environment powered by RL agents. Built for the **Meta × Scaler OpenEnv Hackathon 2026**.

## Problem Statement

Enterprise engineering teams spend thousands of collective hours annually reading through unstructured GitHub issues, identifying whether they are genuine bugs or duplicate feature requests, isolating the faulty lines of code across massive repositories, and determining which specialized internal team (e.g., DevOps, WebDev, AIML) should urgently handle them.

**GitPRTriage Env** transforms this grueling, high-friction manual workflow into a rigorous, interactive OpenEnv reinforcement learning benchmark. By providing agents with raw markdown descriptions and broken code snippets extracted directly from realistic development pipelines, the environment natively tests an LLM's capacity to orchestrate complex developer operations autonomously. It explicitly fills a crucial gap in modern agentic capability evaluations by shifting away from standard toy box games and firmly into professional, multi-layered software development workflows.

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

## Grader Design

Graders are deterministic, non-gameable, and return floats strictly in [0.0, 1.0].

**Easy:** Binary — 1.0 for correct classification, 0.0 otherwise.

**Medium:** Partial credit — 0.40 classification + 0.40 exact bug line + 0.20 proximity bonus (±1 line). String bug_line values are cast to int to handle LLM formatting quirks.

**Hard:** Four equal components (0.25 each) — classification, bug line (0.10 for ±1 proximity), team routing, fix quality. Fix is scored by keyword relevance against `true_fix_keywords` in the dataset: 2+ keywords = 0.25, 1 keyword = 0.15, non-empty but no match = 0.05 effort credit.

## Dataset

30 mock GitHub issues across three difficulty levels (10 each). Each issue contains:
- `true_label` — ground truth classification
- `true_bug_line` — exact bug line for medium/hard
- `true_team` — correct routing team for hard
- `true_fix_keywords` — relevant fix keywords for hard grader

Ground truth fields are never exposed in observations — only used internally by graders.

## Baseline Scores

Evaluated using `llama-3.1-8b-instant` via Groq inference API over a full 60-episode automated environment run. Chain-of-Thought (CoT) reasoning was integrated into the prompt, requiring the model to produce a `thought_process` array before the classification payload, improving contextual understanding on medium and hard tasks.

| Task   | Average Score | Std   | Episodes |
|--------|---------------|-------|----------|
| Easy   | 1.000         | 0.000 | 23       |
| Medium | 0.800         | 0.000 | 18       |
| Hard   | 0.774         | 0.079 | 19       |

## Setup and Usage

### Prerequisites
- Python 3.11
- Docker (for containerized deployment)

### Running Locally

1. Set environment variables:
```bash
export HF_TOKEN="your_api_key_here"
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
  -e HF_TOKEN=your_api_key_here \
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