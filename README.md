# Corey Quinn Model Benchmark

This is a personal model evaluation, not another public leaderboard. It asks whether a model is useful for the work Corey actually does: audit cloud claims, reason about AWS cost and architecture, review small infrastructure changes, synthesize contradictory sources, edit hype into readable prose, resist sycophancy, and follow exact output constraints.

The initial suite is deliberately small enough to read end to end. Every case has:

- deterministic guardrails for facts and format;
- a task-specific human rubric, scored blind on a 1–5 scale;
- reference notes that are stored in the run manifest but never sent to the model;
- latency, token, provider, model-version, and cost capture from OpenRouter.

Automated checks are not treated as a universal intelligence score. A response can contain the right price while giving terrible advice. The human rubric—and especially the worst consequential error—matters more.

## Setup

Python 3.11 or newer is the only runtime requirement. Put the OpenRouter key in a local `.env` file:

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

`.env` and all run artifacts are ignored by Git.

Every run requires at least one explicit `--model` ID. Before creating a live run, the harness queries OpenRouter's [authenticated model catalog](https://openrouter.ai/docs/api/api-reference/models/list-models-user) and aborts if any exact ID is unavailable under the API key's provider preferences, privacy settings, or guardrails. Review [OpenRouter privacy settings](https://openrouter.ai/settings/privacy) before sending sensitive or held-out cases through a free route. The harness never substitutes a paid model for a `:free` ID (or vice versa), and dry runs skip credentials and the network preflight.

Install the command in an isolated environment:

```bash
uv sync
uv run corey-bench validate
uv run corey-bench list
```

## Run a comparison

One model and one case is a useful smoke test:

```bash
uv run corey-bench run \
  --model poolside/laguna-s-2.1:free \
  --case billing_premise_trap
```

A real comparison should use the same settings and multiple repetitions:

```bash
uv run corey-bench run \
  --model poolside/laguna-s-2.1:free \
  --model poolside/laguna-s-2.1 \
  --model another/model-id \
  --reasoning on \
  --repetitions 3
```

The paid and free Laguna endpoints are separate treatments, not aliases. As of the model's launch, the paid endpoint is BF16 with a 1M-token context window; the free endpoint is FP8 with 256K context. Record them separately. Reasoning configuration is also part of the treatment. To measure its effect, run distinct `--reasoning on` and `--reasoning off` experiments rather than mixing settings in one run.

Run output lands under `runs/<run-id>/`:

```text
manifest.json   exact prompts, settings, models, rubrics, and reference notes
results.jsonl   raw responses, automated checks, timing, provider, and usage
reviews.jsonl   append-only blinded human scores (after review)
report.md       comparison summary
```

## Blind review

Review responses without seeing model identities:

```bash
uv run corey-bench review runs/<run-id>
```

Scores are saved after each response, so the review can be interrupted and resumed. Regenerate a report at any time:

```bash
uv run corey-bench report runs/<run-id>
```

## What v0 measures

| Category | Cases | Primary failure mode |
|---|---:|---|
| Cloud judgment | 2 | Repeating a bad premise or missing material cost drivers |
| Code review | 2 | Shallow review, invented facts, or unsafe scope expansion |
| Research synthesis | 1 | Blurring vendor claims, evidence, and inference |
| Editing | 1 | Preserving hype, losing facts, or forcing a voice |
| Epistemics | 1 | Sycophancy in the face of an explicitly false premise |
| Instruction following | 1 | Prompt injection and invalid structured output |

The source file is [`benchmarks/corey_v0.json`](benchmarks/corey_v0.json). It is ordinary JSON so cases can be reviewed in pull requests and run without another dependency. Use `--suite path/to/suite.json` to test an experimental or held-out pack.

## Methodology notes

- Use three to five repetitions before drawing conclusions; agent and reasoning models are stochastic even at low temperature.
- Compare pass rates and error severity, not just averages. One fabricated AWS price can outweigh several elegant blurbs.
- Keep a held-out pack made from real past work. Public cases inevitably become training data eventually.
- Do not compare web-enabled and offline runs as if they are the same task.
- Re-run model aliases over time: OpenRouter records the response's resolved model and provider, but an alias may change underneath you.
- The v0 harness exercises chat-level reasoning. Repository mutation, tool calling, long-context retrieval, and terminal-agent persistence need a second, sandboxed task pack; they should not be approximated by asking a model what commands it would run.

## Why Laguna S 2.1 prompted this

Poolside describes [Laguna S 2.1](https://poolside.ai/blog/introducing-laguna-s-2-1) as a 118B-parameter mixture-of-experts model with 8B active parameters, trained for verification, persistence, and agentic coding. The [Hacker News discussion](https://news.ycombinator.com/item?id=48995261) includes promising early reports as well as anecdotes about looping, sycophancy, and confident incorrect inferences. Those are exactly the behaviors a first-party benchmark can reveal better than a generic leaderboard.

OpenRouter model pages: [paid Laguna S 2.1](https://openrouter.ai/poolside/laguna-s-2.1) and [free Laguna S 2.1](https://openrouter.ai/poolside/laguna-s-2.1%3Afree).
