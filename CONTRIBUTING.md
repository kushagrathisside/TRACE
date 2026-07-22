# Contributing to TRACE

Thank you for your interest in contributing to **TRACE** (*Trustworthy Retrieval with Automated Continuous Evaluation*). This document covers everything you need to get started.

---

## Table of Contents

1. [Before You Start](#before-you-start)
2. [Development Setup](#development-setup)
3. [Branch & Commit Conventions](#branch--commit-conventions)
4. [Running Tests](#running-tests)
5. [Code Style](#code-style)
6. [Pull Request Checklist](#pull-request-checklist)
7. [Running the Eval Suite](#running-the-eval-suite)
8. [Reporting Bugs](#reporting-bugs)
9. [Proposing Features](#proposing-features)

---

## Before You Start

- Read the [README](README.md) for a project overview and architecture.
- Check [docs/feature-upgrades.md](docs/feature-upgrades.md) to see what's already planned.
- Search existing [issues](../../issues) before opening a new one — duplicates slow everyone down.

---

## Development Setup

### Requirements

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Node.js | 18+ (frontend preview only) |
| Ollama | Latest |
| Docker | Optional (for ChromaDB isolation) |

### Steps

```bash
# 1. Fork and clone
git clone https://github.com/<your-fork>/trace.git
cd trace

# 2. Copy env template and fill in required values
cp .env.example .env
# Edit .env — set at minimum:
#   ADMIN_PASSWORD=<something>
#   CORS_ORIGINS=http://localhost:8000

# 3. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 4. Install backend dependencies
pip install -r backend/requirements.txt
pip install pytest pytest-cov ruff

# 5. Pull the required Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text   # or whatever EMBEDDING_MODEL_NAME is set to

# 6. Start the backend
make dev
```

The API is now at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

---

## Branch & Commit Conventions

### Branch naming

```
feat/<short-description>     # new feature
fix/<short-description>      # bug fix
docs/<short-description>     # documentation only
refactor/<short-description> # code cleanup, no behaviour change
test/<short-description>     # new or updated tests
```

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add streaming support to query endpoint
fix: prevent year=0 from breaking incremental sync
docs: document RAGAS scoring configuration
test: add hallucination guard edge-case coverage
```

One sentence, imperative mood, present tense. No period at the end.

---

## Running Tests

The whole suite is self-contained — it mocks Ollama and ChromaDB, so it needs
neither running:

```bash
make unit
```

With coverage, matching exactly what CI runs:

```bash
cd backend
ADMIN_PASSWORD=test CORS_ORIGINS=* pytest tests/ -v --cov=. --cov-report=term-missing
```

There is one test job, not several. An earlier layout split `tests/` across two
CI jobs — everything except `test_api.py`, then `test_api.py` alone — which
installed identical dependencies twice for no benefit. Please keep new tests
inside `tests/` so they stay in the one suite.

### Test files and what they cover

| File | Covers |
|------|--------|
| `tests/test_config.py` | Config validation, env-var parsing, required fields |
| `tests/test_metrics.py` | Ranking metrics; the title-vs-ID guard that stops metrics silently reading 0.0 |
| `tests/test_hallucination_guard.py` | Paper and person grounding, containment floor, Jaccard threshold |
| `tests/test_hybrid_search.py` | BM25 filtering, tokenization, author indexing, RRF fusion |
| `tests/test_ingestor.py` | Atomic status writes, corrupt file recovery, year tracking |
| `tests/test_api.py` | FastAPI routes, auth, pagination, response shapes |

### Do not add scripts named `test_*` outside `tests/`

`pytest.ini` sets `testpaths = tests`, so CI will not collect them — but anyone
running bare `pytest` will, and a script that hits the real pipeline then runs
as if it were a test. Two such files previously sat at `backend/` root; one
hardcoded an absolute home directory. Put throwaway scripts elsewhere, or make
them real tests.

---

## Code Style

All Python code is formatted with [ruff](https://docs.astral.sh/ruff/).

```bash
# Check
ruff check backend/ --select=E,F,W,I --ignore=E501

# Format
ruff format backend/
```

CI will reject PRs that fail either check. Run both before pushing.

### Conventions

- **No comments** unless the *why* is non-obvious (a hidden constraint, a workaround for a specific external bug).
- **No trailing summaries** in functions — name things so they speak for themselves.
- **No backwards-compat shims** — if something is removed, remove it fully.
- **No catching bare `Exception`** unless it's a top-level handler that logs the error.
- Prefer raising `ValueError` in config validation over silently falling back to a default.

---

## Pull Request Checklist

Before opening a PR, check all of these:

- [ ] `ruff check` and `ruff format --check` both pass
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New behaviour is covered by a test
- [ ] No new required env vars added without updating `.env.example` and `docs/developer-guide.md`
- [ ] No secrets, `.env` files, or personal data committed
- [ ] `ADMIN_PASSWORD` has no default hardcoded anywhere
- [ ] PR description explains *why*, not just *what*

---

## Running the Eval Suite

Integration tests and quality benchmarks require Ollama and a populated ChromaDB:

```bash
make seed          # build a real corpus (~1,100 papers, Semantic Scholar API)
make eval-set      # build the labelled query set; committed labels re-applied
make eval-fast     # retrieval metrics -> MLflow, no LLM judge
make ablate        # stage-wise ablation: what each retrieval stage buys
make test          # DeepEval regression suite
make mlflow-ui     # compare runs
```

See [docs/llmops-evaluation.md](docs/llmops-evaluation.md) for how to interpret
the numbers, and [docs/retrieval-experiments.md](docs/retrieval-experiments.md)
for the measured results and the traps that produce confident but meaningless
ones.

These are **not run in CI** because they need a live Ollama instance and a
populated index. If your PR touches retrieval, run `make ablate` before and
after your change and put both tables in the description. A retrieval change
without a before/after is not reviewable.

---

## Reporting Bugs

Open a [GitHub issue](../../issues/new) and include:

- TRACE version or commit hash
- OS and Python version
- Steps to reproduce
- Expected vs. actual behaviour
- Relevant logs (redact any personal data or API keys)

Do not open issues for questions — use [Discussions](../../discussions) instead.

---

## Proposing Features

Before building something significant:

1. Check [docs/feature-upgrades.md](docs/feature-upgrades.md) — it may already be planned.
2. Open a [Discussion](../../discussions) describing the feature and its motivation.
3. Wait for acknowledgement before investing time in implementation.

Small improvements (tests, docs, typo fixes, minor refactors) can go straight to a PR.

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
