PYTHON  := venv/bin/python3
PIP     := venv/bin/pip
UVICORN := venv/bin/uvicorn

.PHONY: setup run dev eval-self eval-feedback eval-full test mlflow-ui clean help

help:
	@echo ""
	@echo "  make setup          Create venv and install dependencies"
	@echo "  make run            Start server (port 8000)"
	@echo "  make dev            Start server with --reload (dev mode)"
	@echo "  make eval-self      Run self-retrieval sanity check"
	@echo "  make eval-feedback  Analyse feedback.jsonl for quality trends"
	@echo "  make eval-full      Run full eval set and log to MLflow"
	@echo "  make test           Run DeepEval regression test suite (pytest)"
	@echo "  make mlflow-ui      Open MLflow experiment dashboard"
	@echo "  make clean          Remove __pycache__ and .pyc files"
	@echo ""

setup:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Setup complete.  Next:"
	@echo "  1. cp .env.example .env && edit .env"
	@echo "  2. ollama pull llama3.2"
	@echo "  3. make run"

run:
	cd backend && env $$(grep -v '^#' ../.env 2>/dev/null | xargs) ../$(UVICORN) main:app --host 0.0.0.0 --port $${PORT:-8000}

dev:
	cd backend && env $$(grep -v '^#' ../.env 2>/dev/null | xargs) ../$(UVICORN) main:app --reload --host 0.0.0.0 --port $${PORT:-8000}

eval-self:
	cd backend && ../$(PYTHON) eval/self_retrieval.py

eval-feedback:
	cd backend && ../$(PYTHON) eval/analyse_feedback.py

eval-full:
	cd backend && ../$(PYTHON) eval/run_eval.py --run-name "$(shell date +%Y%m%d-%H%M)"

test:
	cd backend && ../$(PYTHON) -m pytest eval/test_rag_regression.py -v --tb=short

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri backend/data/mlruns --port 5050

clean:
	find . -type d -name __pycache__ -not -path '*/venv/*' -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path '*/venv/*' -delete 2>/dev/null || true
