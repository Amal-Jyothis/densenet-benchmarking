# ---------- Stage 1: Python base for dependencies ----------
FROM python:3.10-slim AS python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

# Install build tools (only needed for compiling wheels in this stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to leverage Docker cache)
WORKDIR /tmp
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir -r requirements.txt -w /wheels

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

# Install Python 3.10 and minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Copy prebuilt wheels from builder stage and install
COPY --from=python-base /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /benchmark
# COPY pyproject.toml uv.lock /benchmark/
# RUN pip install uv
# RUN uv sync
COPY . /benchmark
CMD ["python3", "main.py"]
EXPOSE 6006
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:6006/ || exit 1
