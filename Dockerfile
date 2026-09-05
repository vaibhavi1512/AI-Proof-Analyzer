# MAYA backend — Dockerfile (CPU-first, Windows 11 compatible Linux build).
# Targets: reproducible CPU deployment; uses Flask dev server for simplicity.
# For production, switch to gunicorn/waitress behind a reverse proxy.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: Pillow/SHAP image codecs + OpenCV shared libs (headless)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libjpeg-dev zlib1g-dev libpng-dev libfreetype6-dev \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install base deps first, then CPU torch (smaller + works without NVIDIA)
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN mkdir -p /app/backend/instance /app/uploads /app/reports /app/logs /app/artifacts/investigations /app/artifacts/checkpoints

# SECRET_KEY is deliberately NOT defaulted here: production boot fails fast if
# it is unset or a placeholder (see backend/app/config validate_production_config).
ENV FLASK_ENV=production \
    PYTHONPATH=/app \
    ROOT_DIR=/app \
    DATABASE_URL=sqlite:////app/backend/instance/maya.db \
    LOG_LEVEL=INFO \
    ALLOW_PUBLIC_REGISTRATION=false

EXPOSE 5000

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3).status==200 else 1)"

CMD ["python", "backend/run.py", "--host", "0.0.0.0", "--port", "5000"]
