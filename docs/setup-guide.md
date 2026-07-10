# TRACE Setup Guide

This guide covers how to set up and run the TRACE server locally using WSL on Windows, optimized for the lowest possible resource usage.

## 1. Prerequisites

1. **WSL (Windows Subsystem for Linux)**: Ensure you have Ubuntu installed in WSL.
2. **Python 3.10+**: TRACE requires Python 3.10 or newer (tested with 3.13).
3. **Ollama**: Download and install [Ollama for Windows](https://ollama.ai/download/windows).

## 2. Pulling the AI Models

Open a PowerShell or WSL terminal and pull the required generative model:
```bash
ollama pull llama3.2:3b
```
*(Note: TRACE uses `all-minilm` for embeddings, which it will automatically download via HuggingFace when the server starts for the first time).*

## 3. Configuration

1. Copy `.env.example` to `.env` in the root of the TRACE directory.
2. Open `.env` and configure your settings.
3. **Crucially**, you must set the `ADMIN_PASSWORD` variable. The server will refuse to start without it.

```env
# ── Models ──
LLM_MODEL_NAME=llama3.2:3b
EMBEDDING_MODEL_NAME=all-minilm
RERANKER_MODEL_NAME=

# ── Admin ──
ADMIN_PASSWORD=your_secure_password
```

*(Note: We leave `RERANKER_MODEL_NAME` empty for local WSL environments to avoid heavy cross-encoder memory requirements and WSL network timeouts).*

## 4. First-Time Python Setup

Open your WSL terminal and navigate to the TRACE directory:
```bash
cd /home/yourusername/TRACE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 5. Starting the Server (The Easy Way)

To start the server with minimal overhead, avoid packaging it with Docker or PyInstaller. We have provided a lightweight Windows batch script (`start.bat`) that boots the server directly inside WSL.

1. Ensure the **Ollama** app is running in your Windows system tray.
2. Double-click the `start.bat` file in the TRACE folder from Windows Explorer.
3. A command prompt will open, and the server will start at `http://0.0.0.0:8000`.

*Note: The script automatically checks if Ollama is accessible. If you see a warning about Ollama being unreachable, it usually means the background daemon is still starting up.*

## 6. Initial Data Ingestion

1. Navigate to **http://localhost:8000/admin** in your browser.
2. Enter your `ADMIN_PASSWORD`.
3. Add researchers (faculty/students) using their Semantic Scholar IDs or names.
4. Click **Sync Now** to pull their papers and build the semantic index.
5. Once the sync says `done`, navigate back to the main student page at `http://localhost:8000/` and try your first search query!
