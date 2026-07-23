# Quinnferno web application and durable model-evaluation worker.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QUINNFERNO_RUNS=/data/runs \
    QUINNFERNO_WORKERS=12 \
    QUINNFERNO_PER_MODEL_WORKERS=3 \
    QUINNFERNO_JUDGE_MODEL=openai/gpt-5.6-luna-pro \
    QUINNFERNO_JUDGE_WORKERS=2 \
    QUINNFERNO_MAX_AUTO_RECOVERIES=3 \
    QUINNFERNO_RECOVERY_WINDOW_SECONDS=3600 \
    QUINNFERNO_REPORT_INTERVAL_SECONDS=30 \
    QUINNFERNO_MAX_REVIEW_ATTEMPTS=2 \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken

RUN apt-get update \
    && apt-get install -y --no-install-recommends imagemagick librsvg2-bin ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 quinnferno \
    && useradd --uid 1001 --gid 1001 --create-home quinnferno

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY benchmarks ./benchmarks
RUN pip install --no-cache-dir . \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base')" \
    && chmod -R a+rX /app \
    && install -d -o 1001 -g 1001 /data/runs

USER 1001:1001
EXPOSE 8765
VOLUME ["/data"]

CMD ["gunicorn", "--bind=0.0.0.0:8765", "--workers=1", "--threads=8", "--timeout=360", "--graceful-timeout=120", "--access-logfile=-", "--error-logfile=-", "corey_bench.wsgi:app"]
