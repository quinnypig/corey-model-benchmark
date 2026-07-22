# Quinnferno web application and durable model-evaluation worker.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QUINNFERNO_RUNS=/data/runs \
    QUINNFERNO_WORKERS=3

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
    && chmod -R a+rX /app \
    && install -d -o 1001 -g 1001 /data/runs

USER 1001:1001
EXPOSE 8765
VOLUME ["/data"]

CMD ["gunicorn", "--bind=0.0.0.0:8765", "--workers=1", "--threads=8", "--timeout=360", "--graceful-timeout=120", "--access-logfile=-", "--error-logfile=-", "corey_bench.wsgi:app"]
