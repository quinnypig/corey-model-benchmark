# Quinnferno

Quinnferno is the web application and durable job runner for **The Quinn Eval Suite**: seven tiers of increasingly unreasonable model evaluations covering AWS judgment, coding, hallucination resistance, voice, political/policy behavior, long-session integrity, calibration, and frontier markers.

The application uses OpenRouter as its provider. It discovers every model available to the configured key, queues comparisons of up to ten models, survives page closes and process restarts, retains raw receipts, and produces:

- a fixed Markdown and JSON scorecard;
- persistent model cards and a cross-run comparison table;
- response browsing by model or eval, plus per-model JSONL and ZIP evidence exports;
- frontier-model rubric reviews, blinded human overrides, and pairwise review queues;
- a safely rendered platypus gallery;
- per-attempt provider, resolved-model, latency, token, retry, and cost metadata.

The original eight-case CLI suite remains available as `benchmarks/corey_v0.json`. The web application runs the complete, versioned `benchmarks/quinn_v1.json` protocol.

## Run locally

Python 3.11 or newer and `uv` are required. Put a dedicated OpenRouter key in `.env`:

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

Then:

```bash
uv sync --all-groups
uv run pytest -q
uv run corey-bench protocol-validate
uv run corey-bench serve
```

Open <http://127.0.0.1:8765>. The local development server deliberately disables Flask's reloader so it cannot create duplicate queue workers.

For a production-style local process:

```bash
uv run gunicorn --bind 127.0.0.1:8765 --workers 1 --threads 8 \
  --timeout 360 corey_bench.wsgi:app
```

Use one process. Queue state is filesystem-backed and thread-safe, but deliberately does not pretend to be a distributed queue.

## Durable queue and spending controls

Every submission immediately creates:

```text
runs/<run-id>/
  manifest.json       frozen treatments, variants, rendered prompts, hashes, jobs
  state.json          atomic queue/progress state
  results.jsonl       append-only attempt receipts
  reviews.jsonl       append-only model and blinded-human rubric judgments
  judge-errors.jsonl  non-fatal automated-review failures
  comparisons.jsonl   append-only pairwise votes
  raw/                verbatim response archives
  artifacts/          sanitized SVG and PNG previews; inert HTML source
  report.md
  report.json
```

Twelve worker threads run by default. `QUINNFERNO_WORKERS=1..24` controls global concurrency and `QUINNFERNO_PER_MODEL_WORKERS` defaults to three, preventing one provider route from consuming every slot. Jobs are interleaved across models. HTTP 429 and 5xx responses use bounded exponential backoff and honor numeric `Retry-After`. One failed request creates a failed receipt and does not abort peer models. Runs left queued or running are recovered when the application starts.

Human-required attempts are asynchronously scored against their frozen rubric by `openai/gpt-5.6-luna-pro` by default. Set `QUINNFERNO_JUDGE_MODEL` and `QUINNFERNO_JUDGE_WORKERS` to change or disable that pool. The candidate model identity is omitted from the judging prompt; the judge model, rationale, criterion scores, usage, and cost are retained. A later blinded human review takes precedence over the model review.

The run form shows estimated request volume, live input/output prices per million tokens, and a low/likely/high spend range for every selected model. Preflight counting dispatches on OpenRouter's tokenizer-family metadata, uses the matching public GPT encoding or a calibrated nearest-family encoding, and widens the range for non-public native tokenizers. It models multi-turn context growth, output lengths, web-search charges, and one-to-ten agentic attempts. OpenRouter's native post-generation usage remains the billing authority.

A hard per-run dollar budget defaults to `$50`; once recorded spend reaches it, remaining jobs are cancelled. Provider-side key limits should still be used as the final guardrail. The estimate is directional, not a quote: reasoning tokens, search behavior, retries, tiered pricing, and unusually long outputs can move the total.

The full suite is intentionally large. Tier 7's coin test alone makes 300 fresh-context calls per model, and the integrity test makes 15 conversational calls per attempt. Every run includes the complete protocol; start with one inexpensive model and use the preflight range before queueing a ten-model comparison.

## Protocol behavior

- No system message is sent by v1 unless a future eval explicitly declares one.
- Weights-only attempts carry no tools. The two search-enabled evals use OpenRouter's `openrouter:web_search` server tool as a separate treatment.
- Exact model ID, resolved model, provider route, requested settings, rendered-prompt SHA-256, date, and full usage are retained.
- Factual/security headline aggregation is minimum/all-pass; creative aggregation is best-of-three but every attempt and its cost remains visible.
- Calibration uses a strict final `Confidence: N%` line. Missing confidence is missing, never silently converted to zero.
- Human-required scores remain unresolved until a model or human rubric review lands; deferred scores remain unresolved until their registered outcome exists. Quinnferno reports provisional points over resolved weight and emits `/100` only after all weighted judgments resolve.
- A model card is ranked only after all 97 required per-model attempt receipts in the complete suite matrix are present and all weighted judgments resolve. Partial runs remain browsable, but their extrapolated score is withheld from comparison tables.
- Tier 7 never contributes to `/100`.

The frozen weighting policy is encoded in `benchmarks/quinn_v1.json` and validated to total exactly 100 points across Tiers 1–6.

## Artifact safety

Model-generated SVG and HTML are hostile input. Raw output is served as an attachment or escaped text only.

Platypus SVGs are size-limited, parsed as XML, and rejected if they contain document types, entities, scripts, event handlers, `foreignObject`, animation, external URLs, data URLs, unsupported elements, extreme dimensions, or excessive nodes. Only sanitized SVG is passed to a time-limited standards renderer (with a resource-limited ImageMagick fallback), and the gallery serves the resulting PNG. If a model exhausts its output budget after starting valid SVG, Quinnferno can close the already-open safe elements to create a clearly labeled preview; that repair never changes the failed benchmark grade. Generated status-page HTML is archived as inert text and never placed into the Quinnferno origin.

The current “agentic” coding treatment performs up to ten model repair loops from deterministic static gate feedback. It does not claim browser/visual verification; adding an isolated Chromium worker is a future protocol version, not a silent methodology change.

## Operations and spend safety

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to an OpenTelemetry Collector base URL (for
example, `http://otel-collector.monitoring:4318`) to export Flask, queue startup,
run submission/recovery, eval job, OpenRouter chat/retry, model-judge, report,
cost, token, and process/cgroup memory spans. Health probes are excluded and
prompt or response bodies are never attached to spans. If no collector is
configured, `HONEYCOMB_API_KEY` enables direct OTLP/HTTP export to Honeycomb.

Incomplete runs normally recover after a restart. Quinnferno persists recovery
history and pauses a run after three automatic recoveries in one hour by
default, bounding repeated paid calls during a crashloop. The run page explains
the fuse and offers an explicit resume action. Tune this with
`QUINNFERNO_MAX_AUTO_RECOVERIES` and `QUINNFERNO_RECOVERY_WINDOW_SECONDS`.
Model-judge calls write a durable intent before contacting OpenRouter and stop
after two attempts per response (`QUINNFERNO_MAX_REVIEW_ATTEMPTS`), including
calls whose process died before it could save the final review receipt.
Reports are rebuilt at most every 30 seconds while a run is active and forced
at completion, avoiding the allocator pressure caused by materializing a
growing receipt ledger after every result.

## Private inputs still needed for publication-grade runs

The harness is complete, but several editorial ground truths cannot be manufactured by code:

- a dated canonical AWS service catalog at `benchmarks/truth/aws_services.txt`;
- private fake-service, IAM, and pricing variants that should not be committed publicly;
- current RI and data-transfer answer-key signoff;
- unpublished Corey control prose for the blind voice panel;
- private-pricing grades, quarterly forecast resolutions, and audience judgments.

Absent inputs are shown as pending review rather than guessed. The included variants are synthetic launch fixtures suitable for exercising the system, not a claim that private answer keys have been supplied.

## Container

The multi-architecture image is built by GitHub Actions and published as `ghcr.io/quinnypig/quinnferno`. It runs as UID/GID 1001, listens on port 8765, and stores durable run data under `/data/runs`.

The build context intentionally excludes `.git`, so BuildKit may warn that it
could not capture current commit information. The workflow supplies image
revision labels separately; do not add repository history to the image context
just to suppress that harmless warning.

```bash
docker build -t quinnferno .
docker run --rm -p 8765:8765 \
  -e OPENROUTER_API_KEY \
  -v quinnferno-data:/data \
  quinnferno
```

For a startup smoke test, use a named container so a crash cannot discard its
logs. This check uses the image's declared port, removes the container after a
successful response, and retains it after a failure:

```bash
(
  set -eu
  container="quinnferno-smoke-$$"
  docker run --name "$container" -d -p 127.0.0.1:8765:8765 \
    -e OPENROUTER_API_KEY \
    quinnferno
  if ! curl --fail --silent --show-error --retry 10 --retry-connrefused \
    --retry-all-errors --retry-delay 1 http://127.0.0.1:8765/ >/dev/null; then
    docker logs "$container"
    echo "Retained failed container: $container" >&2
    exit 1
  fi
  docker rm -f "$container"
)
```

For bind-mounted historical data, give the image's UID access to the exact
temporary directory before starting the container. Files created by the
container may not remain traversable by the host user, so inspect them with
`sudo` and restore ownership before cleanup:

```bash
fixture_dir="$(mktemp -d)"
sudo chown 1001:1001 "$fixture_dir"
sudo chmod 0755 "$fixture_dir"
docker run --name quinnferno-history-smoke -d \
  -p 127.0.0.1:8765:8765 \
  -e OPENROUTER_API_KEY \
  -v "$fixture_dir:/data" \
  quinnferno
sudo find "$fixture_dir" -maxdepth 3 -type f -print
sudo chown -R "$(id -u):$(id -g)" "$fixture_dir"
```

## Kubernetes diagnostics

Do not guess the live pod name or application port. Resolve both from the
workload, use the bundled Python runtime instead of assuming `wget` or `jq` is
in the slim image, and pass `-i` whenever an inline script is sent on standard
input:

```bash
namespace=quinnferno
pod="$(
  kubectl get pods -n "$namespace" -l app.kubernetes.io/name=quinnferno \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
)"
port="$(
  kubectl get deployment -n "$namespace" quinnferno \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="quinnferno")].ports[?(@.name=="http")].containerPort}'
)"

kubectl exec -n "$namespace" "$pod" -- \
  python -c \
  'import sys, urllib.request; print(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz").read().decode())' \
  "$port"

kubectl exec -i -n "$namespace" "$pod" -- \
  python - /data/runs/<run-id>/state.json <<'PY'
import json
import sys

print(json.dumps(json.load(open(sys.argv[1])), indent=2))
PY

kubectl cp "$namespace/$pod:/data/runs" ./quinnferno-runs
```

Run host-side `jq`, `find`, or visual tooling against the copied directory.
Unlike `kubectl exec`, `kubectl cp` requires the concrete pod name resolved
above; it does not accept `deployment/quinnferno`.

## Legacy CLI

The original benchmark remains available:

```bash
uv run corey-bench validate
uv run corey-bench list
uv run corey-bench run --model openai/gpt-oss-20b:free --case billing_premise_trap
uv run corey-bench review runs/<run-id>
```
