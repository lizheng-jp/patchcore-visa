# Live demo image. Weights, memory banks and samples are not baked in: mount them at /app/models,
# /app/samples and /app/runtime/weights (see README "Deploy").
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BIND_HOST=0.0.0.0 TORCH_THREADS=2
WORKDIR /app
# Pick up Debian security updates at every build (CI rebuilds on each demo change).
RUN apt-get update && apt-get -y upgrade && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
# The base image ships setuptools 78.1.0 (PYSEC-2025-49, GHSA-h35f-9h28-mq5c).
RUN pip install --no-cache-dir --upgrade "setuptools>=83"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py index.html app.js i18n.js style.css favicon.svg showcase.json sam-study.json recorded-experiments.json ./
COPY cases ./cases
COPY runtime/efficient_sam ./runtime/efficient_sam
RUN useradd --system --uid 10001 lab && mkdir -p .runtime-cache && chown lab .runtime-cache
USER lab
EXPOSE 8766
CMD ["python", "-W", "ignore", "server.py", "--port", "8766"]
