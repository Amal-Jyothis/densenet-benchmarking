FROM python:3.10-slim AS python-base

ENV PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir -r requirements.txt -w /wheels

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1


COPY --from=python-base /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

WORKDIR /benchmark
COPY . /benchmark
CMD ["python3", "main.py"]
EXPOSE 6006
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:6006/ || exit 1
