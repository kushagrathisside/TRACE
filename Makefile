PYTHON  := venv/bin/python3
# Load .env safely: `env $(grep .env | xargs)` word-splits values containing
# spaces, so a value like INSTITUTE_NAME=TRACE Institute becomes a command.
LOADENV := set -a; . ../.env 2>/dev/null || true; set +a;
PIP     := venv/bin/pip
UVICORN := venv/bin/uvicorn

.PHONY: setup run dev seed eval-set label-pool label-apply eval-self eval-feedback eval-full eval-fast ablate sweep test unit mlflow-ui clean help

help:
	@echo ""
	@echo "  make setup          Create venv and install dependencies"
	@echo "  make run            Start server (port 8000)"
	@echo "  make dev            Start server with --reload (dev mode)"
	@echo "  make seed           Seed a realistic corpus from real AI/ML authors"
	@echo "  make eval-set       Build the labelled eval set from the index"
	@echo "  make label-pool     Pool candidates for hand-labelling the manual slice"
	@echo "  make label-apply    Write worksheet judgements back into eval_set.json"
	@echo "  make eval-fast      Retrieval metrics only (no LLM judge)"
	@echo "  make eval-full      Full eval set + LLM-judge metrics, logged to MLflow"
	@echo "  make ablate         Stage-wise ablation table (dense/bm25/hybrid/+CE)"
	@echo "  make sweep P=fetch_k V=5,10,20,40  Parameter sweep"
	@echo "  make eval-self      Run self-retrieval integrity check"
	@echo "  make eval-feedback  Analyse traces + feedback for quality trends"
	@echo "  make unit           Run the unit test suite"
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
	cd backend && $(LOADENV) ../$(UVICORN) main:app --host 0.0.0.0 --port $${PORT:-8000}

dev:
	cd backend && $(LOADENV) ../$(UVICORN) main:app --reload --host 0.0.0.0 --port $${PORT:-8000}

seed:
	cd backend && $(LOADENV) ../$(PYTHON) eval/seed_corpus.py --ingest

eval-set:
	cd backend && $(LOADENV) ../$(PYTHON) eval/build_eval_set.py

label-pool:
	cd backend && $(LOADENV) ../$(PYTHON) eval/label_manual.py --pool

label-apply:
	cd backend && $(LOADENV) ../$(PYTHON) eval/label_manual.py --apply

eval-fast:
	cd backend && $(LOADENV) ../$(PYTHON) eval/run_eval.py --no-ragas --run-name "$(shell date +%Y%m%d-%H%M)"

ablate:
	cd backend && $(LOADENV) ../$(PYTHON) eval/ablate.py --markdown

# Usage: make sweep P=fetch_k V=5,10,20,40,80
sweep:
	cd backend && $(LOADENV) ../$(PYTHON) eval/ablate.py --sweep $(P) --values $(V) --markdown

unit:
	cd backend && env ADMIN_PASSWORD=test CORS_ORIGINS='*' ../$(PYTHON) -m pytest tests/ -q

eval-self:
	cd backend && $(LOADENV) ../$(PYTHON) eval/self_retrieval.py

eval-feedback:
	cd backend && $(LOADENV) ../$(PYTHON) eval/analyse_feedback.py

eval-full:
	cd backend && $(LOADENV) ../$(PYTHON) eval/run_eval.py --run-name "$(shell date +%Y%m%d-%H%M)"

test:
	cd backend && ../$(PYTHON) -m pytest eval/test_rag_regression.py -v --tb=short

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri backend/data/mlruns --port 5050

clean:
	find . -type d -name __pycache__ -not -path '*/venv/*' -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path '*/venv/*' -delete 2>/dev/null || true
