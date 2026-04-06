---
title: Automated Code Review Environment
emoji: review
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Automated Code Review Environment

This repository implements a deterministic OpenEnv environment that simulates a real pull request review workflow. The agent reviews a PR by exploring diffs, running static analysis, checking CI, reading repository policy, requesting AI review notes, posting review comments, and making a final approve or reject decision.

The environment is designed for Round 1 OpenEnv submissions where the benchmark must model a real task, expose the standard `reset()`, `step()`, and `state()` APIs, and include deterministic graders with partial-progress reward shaping.

## Why This Environment

Code review is a real, repeated workflow performed by developers and platform teams every day. Good reviewers do more than read diffs:

- they inspect the right files
- run the right checks
- separate true blockers from noise
- cite actionable issues
- choose the correct merge decision

This environment captures that behavior in a safe deterministic simulation.

## Action Space

The environment accepts a typed `CodeReviewAction` with these `action_type` values:

- `view_pr`
- `inspect_file`
- `run_static_analysis`
- `run_tests`
- `read_policy`
- `request_ai_review`
- `submit_comment`
- `set_decision`

Optional structured fields:

- `file_path`: target file for inspection or comment placement
- `finding_ids`: structured finding references for comments
- `comment_text`: review comment body
- `decision`: `approve` or `reject`

## Observation Space

Each `CodeReviewObservation` returns:

- task metadata and objective
- PR summary and changed file paths
- all revealed artifacts so far
- discovered structured findings
- comments posted so far
- remaining steps
- `last_action_error`
- reward and done status

The `state()` API exposes deterministic grader state including discovered findings, checks run, comments posted, cumulative reward, and action history.

## Tasks

### 1. `clean_refactor_approve`

Easy task. A formatter refactor updates helper naming and docs with no behavior change. The correct outcome is approval after lightweight verification.

### 2. `secret_leak_reject`

Medium task. A production analytics token is committed directly into source code. The correct outcome is rejection with a blocking comment covering the secret leak.

### 3. `auth_policy_reject`

Hard task. An impersonation change removes a required admin scope check, misses denied-path regression coverage, and includes one misleading low-severity warning. The correct outcome is rejection with coverage of both required findings while not confusing the false positive with a blocker.

## Reward Design

Rewards are deterministic and normalized to `[0.0, 1.0]` across the full episode.

- first-time useful review actions earn partial progress
- inspecting the most relevant file earns more than inspecting noise
- posting comments that cover required findings earns structured reward
- duplicate or wasteful actions consume steps and usually earn `0.0`
- final decision reward depends on correctness and required finding coverage

An optimal trajectory reaches a cumulative reward of `1.0`.

## Project Layout

```text
.
|- __init__.py
|- client.py
|- inference.py
|- models.py
|- openenv.yaml
|- policy.py
|- pyproject.toml
|- scoring.py
|- task_fixtures.py
|- fixtures/
|  `- tasks/
|- engines/
|- server/
|  |- app.py
|  |- code_review_environment.py
|  |- Dockerfile
|  `- requirements.txt
`- tests/
```

## Setup

Install dependencies:

```bash
python -m pip install openenv-core openai pytest httpx pyyaml uv
```

Generate the lockfile and install the package:

```bash
uv lock
uv sync
```

Run local validation:

```bash
openenv validate
pytest
```

## Local Usage

Run the server locally:

```bash
uv run server --host 0.0.0.0 --port 8000
```

Or with Uvicorn directly:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

Build the environment image:

```bash
docker build -t code-review-env:latest -f server/Dockerfile .
```

Run it:

```bash
docker run --rm -p 8000:8000 code-review-env:latest
```

## Inference Script

The required `inference.py` lives at the repository root and:

- uses the OpenAI client for model calls
- reads `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`, and `LOCAL_IMAGE_NAME` or `IMAGE_NAME`
- runs all 3 tasks sequentially
- prints exact `[START]`, `[STEP]`, and `[END]` log lines
- falls back to a deterministic structured policy when the model output is invalid

Example:

```bash
set HF_TOKEN=your_token
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
set LOCAL_IMAGE_NAME=code-review-env:latest
python inference.py
```

## Baseline Scores

Local deterministic fallback baseline:

| Task | Score |
| --- | ---: |
| `clean_refactor_approve` | `1.000` |
| `secret_leak_reject` | `1.000` |
| `auth_policy_reject` | `1.000` |
| Average | `1.000` |

These scores come from the built-in structured fallback policy used when the live model output is invalid or unavailable. Live model scores will depend on the selected model and endpoint.

## Deployment to Hugging Face Spaces

This repository is packaged as a Docker Space and already includes `openenv.yaml` plus a working `server/Dockerfile`.

Typical deploy flow:

```bash
openenv validate
openenv push --repo-id <your-hf-space>
```

After deployment, validate the running Space:

```bash
openenv validate https://your-space.hf.space
```

