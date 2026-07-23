# syntax=docker/dockerfile:1
#
# TRACE application image.
#
# Ollama is NOT bundled here. It is a separate service in docker-compose.yml,
# for two reasons: model weights are gigabytes and belong in a volume rather
# than an image layer, and GPU passthrough is configured per host. Keeping them
# apart means this image stays small and rebuilds in seconds when code changes.
#
# No `apt-get` anywhere. Every dependency resolves to a manylinux wheel, and the
# health check uses the Python already in the image instead of curl. That drops
# a package-manager layer, shrinks the image, removes a class of CVEs, and — the
# reason it came up here — builds behind restrictive networks that intercept
# Debian mirrors.
#
# Behind a proxy:
#   docker build \
#     --build-arg HTTP_PROXY=$http_proxy \
#     --build-arg HTTPS_PROXY=$https_proxy -t trace .

FROM python:3.11-slim

# Proxy settings are build-time only: ARG values are not persisted into the
# image, so a proxy baked in at build time will not follow the container to
# another network.
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY=""

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer stays cached until requirements.txt changes,
# so ordinary code edits rebuild in seconds rather than reinstalling torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Run unprivileged. The data directory is a mount point owned by this user so
# the container can write the Chroma index, trace log and registry.
RUN useradd --create-home --uid 10001 trace \
 && mkdir -p /app/backend/data /home/trace/.cache \
 && chown -R trace:trace /app /home/trace
USER trace

# Cache HuggingFace downloads (the cross-encoder, ~90 MB) on a volume so a
# restart does not re-download it.
ENV HF_HOME=/home/trace/.cache/huggingface

WORKDIR /app/backend
EXPOSE 8000

# /health returns 200 even when a subsystem is degraded, so that monitors can
# read the body. The check therefore asserts on content: an unreachable Ollama
# or an inactive reranker fails here rather than surfacing later as bad answers.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "\
import json,sys,urllib.request; \
d=json.load(urllib.request.urlopen('http://localhost:8000/health',timeout=5)); \
sys.exit(0 if d.get('status')=='ok' else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
