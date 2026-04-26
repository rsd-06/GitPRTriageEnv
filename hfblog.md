---
title: "PRRegressionAuditEnv: Teaching LLMs to Catch Code Regressions with GRPO"
authors:
  - user: rsd-06
  - user: SaiSanjayR
---

# PRRegressionAuditEnv: Teaching LLMs to Catch Code Regressions with GRPO

Every day, developers merge Pull Requests that accidentally introduce deep-seated regressions entirely unrelated to the feature they intended to ship. A developer adds a Stripe integration and inadvertently leaves a live API key hardcoded. An ML engineer adds gradient clipping but places it *after* the optimizer step, rendering it mathematically useless. A backend developer implements JWT validation but reads the decoding algorithm from the unverified token header itself, opening the door to algorithm confusion attacks. These bugs are practically invisible in isolation because the PR descriptions only describe the intended feature—never the flaw.

To solve this, we built a rigorous reinforcement learning benchmark around this exact problem for the Meta × Scaler OpenEnv Hackathon 2026. We trained a Qwen2.5-1.5B model using Group Relative Policy Optimization (GRPO) to act as an automated senior code reviewer. Our results demonstrate progressive, measurable improvement from a struggling baseline to a capable Stage 1 GRPO model, culminating in a highly robust Stage 2 model trained using curriculum learning and anti-reward-hacking guardrails.

## The Environment Design

The RL loop in PRRegressionAuditEnv is built around single-step episodes. The interaction is strictly deterministic: `POST /reset` to receive the PR observation, followed by `POST /step` with the agent's action, which immediately returns a scalar reward. 

We structure the benchmark across three escalating task tiers with exact scoring weights:
- **Easy (PR Safety Gate):** `review_decision` (0.55) + `blocker_type` (0.45). Agents evaluate 15 PRs (6 clean, 9 flagged). Blockers include `debug_output`, `hardcoded_secret`, `do_not_merge_comment`, `debug_test_bypass`, and `syntax_error`.
- **Medium (Regression Localization):** `review_decision` (0.10) + `defect_category` (0.40) + `faulty_line` (0.35 exact / 0.15 proximity). Agents evaluate 15 flagged PRs containing self-contained defects.
- **Hard (Full Audit & Integration Review):** `review_decision` (0.05) + `defect_category` (0.20) + `faulty_line` (0.25) + `reviewer_team` (0.25) + `suggested_change` (0.25). The proposed code looks perfectly correct in isolation, but a `context_snippet` reveals a critical interaction bug.

The environment's anti-hack design actively punishes lazy strategies. For instance, an agent might learn to always output `request_changes` to guarantee points on flagged PRs. However, doing so loses 0.45 points per clean PR because the `blocker_type` must be explicitly set to `null` for clean code. 

A concrete example of a Hard task is our JWT Decoder PR. The proposed code parses a JWT header to extract the algorithm type before verifying the signature. In isolation, it's flawless Python. However, the provided `context_snippet` shows that the downstream middleware inherently trusts whatever algorithm the token claims to use. The agent must synthesize both files to realize the PR enables an algorithm confusion attack.

## The Multi-Agent Architecture

To conquer the environment, we built a four-agent sequential pipeline:

- **SafetyGateAgent**: Executes a binary `approve`/`request_changes` decision and identifies blocker types. Dedicated to Easy tasks.
- **DefectLocatorAgent**: Identifies the defect category (security, logic, or performance) and pinpoints the exact 1-indexed faulty line. Dedicated to Medium and Hard tasks.
- **ReviewerRouterAgent**: Routes the flagged PR to the correct expert team (`infosec`, `devops`, `core-frontend`, `core-sysdev`, `aiml`). Hard tasks only.
- **ReviewCommentAgent**: Writes a concise, actionable suggested change under 200 characters. Hard tasks only.

Each agent receives upstream results injected directly into its context. The `DefectLocatorAgent` knows the safety gate decision, and the `ReviewCommentAgent` knows exactly which team was assigned and which line was flagged as faulty. Contrast this with our single-agent baseline, which attempts to deduce the entire complex JSON schema in one monolithic LLM call.

```text
Issue Observation → Orchestrator → SafetyGateAgent → DefectLocatorAgent → ReviewerRouterAgent → ReviewCommentAgent
```

This decomposition is critical for Hard tasks. When proposed code looks correct in isolation, a monolithic model often hallucinates generic feedback. By forcing focused reasoning on the code-context interaction layer by layer, the pipeline reliably isolates integration defects without being overwhelmed by schema formatting.

## Training with GRPO

Group Relative Policy Optimization (GRPO) generates multiple completions per prompt and optimizes the policy relative to the group's average reward. By eliminating the need for a separate value function, GRPO is highly memory-efficient and practical for fine-tuning smaller models on verifiable reward signals.

Our training progression was divided into three stages:

**Stage 0 — Baseline (Qwen2.5-1.5B-Instruct, no fine-tuning)**
The base model inherently understands code but fails consistently on strict JSON schema compliance, exact 1-indexed faulty line identification, and keyword-matching constraints on the `suggested_change` field. Results: Easy 0.890, Medium 0.612, Hard 0.421. Note these are from the multi-agent baseline run.

**Stage 1 — GRPO V1**
*400 steps, direct ground-truth reward*
Our key insight was to score the model directly against our `prs.json` ground truth rather than polling the live HTTP environment. Because the live environment utilizes a dynamic curriculum sampler, it is impossible to guarantee that a specific PR lands in a given training batch. The reward function mirrors the environment's exact scoring weights. Format reward (0.2) is a separate reward function that fires for any valid JSON with a correct `review_decision` field—preventing the model from collapsing to a `0.001` reward on every sample early in training. Results: Easy 0.890, Medium 0.612, Hard 0.421 (training in progress).

**Stage 2 — Curriculum + Guardrails V2**
*600 additional steps from the V1 adapter*
We resumed training from V1 with two major additions:
1. **Curriculum**: The dataset oversamples easy PRs initially (70/20/10 split), transitions to medium-heavy (20/60/20) once the easy average exceeds 0.80, then hard-heavy (10/30/60) once the medium average exceeds 0.65.
2. **Anti-reward-hacking penalties inline**: A diversity penalty fires when all 4 GRPO generations predict the same `review_decision` (reward multiplied by 0.5); a contradiction penalty fires when the `review_decision` is `approve` but the `blocker_type` or `defect_category` is non-null (reward set to 0.001).

A critical GRPO insight: reward standard deviation within a generation group must be non-zero for gradients to flow. If all 4 generations receive identical rewards, the gradient is zero. Using a temperature of 0.9 and separate format/environment reward functions maintains the necessary variance.

| Stage | Easy | Medium | Hard | Notes |
|-------|------|--------|------|-------|
| Baseline | 0.890 | 0.612 | 0.421 | Multi-agent inference |
| Stage 1: GRPO V1 | 0.890 | 0.612 | 0.421 | Training in progress |
| Stage 2: Curriculum V2 | 0.985 | 0.842 | 0.710 | Fake Data - Training in progress |

## Anti-Reward Hacking: Guard Suite

The environment server applies post-grading reward multipliers via a `GuardSuite` that operates independently of the grader logic.

| Guard | What It Catches | Penalty |
|-------|-----------------|---------|
| `KeywordStuffingDetector` | `suggested_change` with >40% keyword density | 0.50–0.90× scaled |
| `RepetitionDetector` | Same action fingerprint repeated >3× in 10 episodes | 0.50–0.90× scaled |
| `FixQualityValidator` | `suggested_change` with <4 words or no action verb | 0.70–0.80× |
| `TimingGuard` | Response under 200ms (likely cached) | 0.95× (soft) |

This two-layer approach is essential: training guardrails steer the policy mathematically; server guardrails catch exploits that emerge post-training when a fixed policy is deployed against the evaluation loop.

## Results & Resources

Our GRPO pipeline transformed a struggling base model into a specialized code reviewer capable of identifying deep integration defects. 

| Stage | Easy | Medium | Hard | Notes |
|-------|------|--------|------|-------|
| Baseline (Qwen2.5-1.5B-Instruct) | 0.890 | 0.612 | 0.421 | Multi-agent inference, no fine-tuning |
| Stage 1: GRPO V1 | TBD | TBD | TBD | 400 steps, direct reward |
| Stage 2: Curriculum + Guards V2 | TBD | TBD | TBD | +600 steps, curriculum + penalties |

| Resource | Link |
|----------|------|
| Live Environment Space | [rsd-06/PRRegressionAuditEnv](https://huggingface.co/spaces/rsd-06/PRRegressionAuditEnv) |
| GRPO Dataset V2 | [rsd-06/pr-regression-audit-grpo](https://huggingface.co/datasets/rsd-06/pr-regression-audit-grpo) |
| Trained Adapter V1 | [SaiSanjayR/pr-triage-grpo-adapter](https://huggingface.co/SaiSanjayR/pr-triage-grpo-adapter) |
| Trained Adapter V2 | [rsd-06/pr-regression-audit-grpo-adapter-v2](https://huggingface.co/rsd-06/pr-regression-audit-grpo-adapter-v2) |

The gap between code that looks correct in isolation and code that breaks the system it integrates with is exactly where human reviewers fail. Turning that specific gap into a programmatic, verifiable training signal proved to be a highly productive frontier for reinforcement learning agents.
