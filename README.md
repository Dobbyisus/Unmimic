# Unmimic

Unmimic sits in front of an AI model served over an OpenAI-compatible API (vLLM,
Ollama, TGI, llama.cpp server) and protects it from distillation attacks: people
harvesting the model's step-by-step reasoning to train a cheaper copy.

It does four things:

1. **Input check** — blocks requests that try to make the model reveal its reasoning.
2. **Output check** — finds reasoning that leaks into an answer, even translated or
   encoded, and removes or rewrites it.
3. **Account watch** — spots accounts that behave like copycats and slows or bans them.
4. **Adaptation** — a short setup that learns each model's format and normal traffic,
   attacks the model to find what works, and calibrates thresholds to a target
   false-positive rate.

> **Status:** early development. The proxy runs as a passthrough with slots for guards;
> the guards themselves do not exist yet.

## Run the proxy

First, let Unmimic learn where the model puts its reasoning. This asks the model 20
ordinary questions and saves the result to `data/models/<model>.json`:

```bash
unmimic init --endpoint http://localhost:11434/v1 --model qwen3:8b
```

Then start the proxy in front of it:

```bash
unmimic serve --upstream http://localhost:11434/v1
```

Then point any OpenAI-compatible client at `http://127.0.0.1:9000/v1` instead of the
model. Settings can also come from environment variables: `UNMIMIC_UPSTREAM_URL`,
`UNMIMIC_UPSTREAM_API_KEY`, `UNMIMIC_HOST`, `UNMIMIC_PORT`, and
`UNMIMIC_OUTPUT_GUARDS` (JSON list, default `["leak_detector", "strip_reasoning"]`).

## Reasoning stays hidden

By default users receive only the answer: Unmimic removes reasoning fields
(`reasoning_content`, `reasoning`), `<think>` blocks, and text before a "Final Answer:"
marker. To let users see the model's reasoning, turn the strip transform off:

```bash
UNMIMIC_OUTPUT_GUARDS='["leak_detector"]' unmimic serve
```

If you swap the model behind an endpoint, re-run `unmimic init`: a stale format config can
strip the wrong part of a reply.

## Leak detector

On by default. If an answer contains the model's hidden reasoning, whether copied word
for word, base64- or ROT13-encoded, reversed, wrapped in tags, or translated, the answer
is replaced with a notice. Install the `ml` extras to turn on the translation check (it
downloads a ~470 MB multilingual embedding model on first use):

```bash
pip install -e ".[ml]"
```

## Install (development)

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"
unmimic --help
```

## Planned usage

```bash
unmimic init --endpoint http://localhost:8000/v1
unmimic redteam --rounds 3
unmimic calibrate --benign my_prompts.jsonl --fpr 0.001
unmimic serve --port 9000
```

## Safety

Unmimic's red-team step attacks the model it protects. Only point it at models you
host yourself. Never run it against a commercial or hosted API.
