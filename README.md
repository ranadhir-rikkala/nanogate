# NanoGate

**Every AI request takes the cheapest safe route, and leaves a receipt that proves it.**

NanoGate is a local-first gateway that sits between your applications and your models. Applications keep their OpenAI SDK
and change only `base_url` and `api_key`. For each request the gateway answers, in order:

1. Who is calling? (API key bound to a tenant and department)
2. What data is in the request? (DLP classification)
3. Which routes does policy permit for that data? (decided before any model or cache is touched)
4. Can a previous answer be reused safely? (semantic cache with a verifier)
5. Is the small local model's answer likely good enough? (calibrated risk router)
6. If not, is a larger local tier or a remote tier allowed, and affordable?
7. Is the answer safe to return? (output screening)
8. What happened, and can that be proven later? (hash-chained, sealed receipt)

```
request ─▶ identity ─▶ rate limit ─▶ DLP ─▶ policy ─▶ budget hold ─▶ verified cache ─▶ local small
        ─▶ risk router ─▶ local large ─▶ remote (only if policy already allowed it) ─▶ output scan
        ─▶ budget settle ─▶ cache write ─▶ sealed receipt ─▶ OpenAI-shaped response + x-nanogate-* headers
```

Two rules hold everywhere:

- **Policy runs before the cache and the model.** A blocked request reaches neither.
- **The router suggests, policy decides.** Nothing downstream of the policy decision can add a route.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/nanogate init      # creates var/nanogate.db and var/bootstrap_keys.json (mode 0600)
.venv/bin/nanogate serve     # http://127.0.0.1:8080  (API + dashboard)
```

With no configuration the gateway uses the **offline backend**: every stage runs for real, but the "model" returns a
labelled placeholder. Point it at any OpenAI-compatible server (vLLM, llama.cpp, Ollama) to get answers:

```bash
cp .env.example .env
# NANOGATE_BACKEND=openai
# LOCAL_MODEL_BASE_URL=http://127.0.0.1:8000/v1
# LOCAL_MODEL_NAME=<small model>      LOCAL_LARGE_MODEL_NAME=<escalation model>
```

### Call it from the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="<app key from var/bootstrap_keys.json>")
reply = client.chat.completions.with_raw_response.create(
    model="nanogate-auto",
    messages=[{"role": "user", "content": "How do I reset the VPN client?"}],
)
print(reply.headers["x-nanogate-route"], reply.headers["x-nanogate-reason"])
```

Models: `nanogate-auto` (policy and router decide), `nanogate-local`, `nanogate-local-large`, `nanogate-remote`.
A pinned model is still checked against policy and returns `ROUTE_NOT_PERMITTED` when it is not allowed.

## What is in the box

| Area | Module | What it does |
|---|---|---|
| Identity | `nanogate/auth.py` | API keys stored as HMAC hashes, roles (`app`, `auditor`, `admin`), header-spoof rejection, token-bucket rate limit |
| DLP | `nanogate/dlp.py` | Checksum-validated detectors (Luhn, SSN rules, IPv4), credential patterns, Unicode and "at/dot" de-obfuscation, redaction. Reports entity types and counts only |
| Policy | `nanogate/policy.py`, `config/policy.yaml` | Per-department routes, data-class ceilings for remote and cache, PII action (`allow`, `local_only`, `redact`, `block`), content-addressed version |
| Budget | `nanogate/budget.py`, `config/rates.yaml` | Reserve, settle, release under `BEGIN IMMEDIATE`; every rate carries a `basis` label |
| Cache | `nanogate/cache/` | Namespace = hash(tenant, department, policy version, model, system prompt, embedder). Retrieval, then a slot verifier that rejects number, negation, antonym, name and term mismatches |
| Router | `nanogate/router/` | One feature function for training and serving, logistic regression, isotonic calibration, threshold chosen for a target accepted-error rate. Pure Python |
| Inference | `nanogate/inference/` | OpenAI-compatible local tiers with logprobs, offline backend, remote **simulator** behind an egress guard |
| Receipts | `nanogate/receipts.py` | `digest = sha256(prev_digest + canonical_json(body))`, HMAC seal per receipt, verification names the first broken receipt |
| API | `nanogate/api/` | `/v1/chat/completions` (JSON and SSE), `/v1/models`, `/healthz`, `/readyz`, `/metrics`, tenant-scoped `/api/*` |
| Dashboard | `nanogate/web/` | Single static page: live decisions, receipt drill-down, chain check, policy dry run |
| Attack suite | `bench/attacks.json`, `nanogate/attack.py` | Secrets, obfuscation, egress, cache isolation, hard negatives, output leaks |

## Commands

```bash
nanogate init                      # database + bootstrap keys
nanogate serve [--host --port]     # run the gateway
nanogate keys create acme it --role app --label crm
nanogate keys list | revoke <key_id>
nanogate verify [--head head.json] # verify the receipt chain, optionally against an anchored head
nanogate train-router graded.jsonl # fit the router from graded rows
nanogate attack                    # run the attack suite in-process
```

## Training the router

The router ships **untrained**, and an untrained router never claims confidence: answers are flagged
`LOCAL_UNVERIFIED`, nothing is cached, and the receipt records `DEGRADED_router`. To train it, grade real answers from
your own serving stack and write one JSON object per line:

```json
{"features": {"logprob_mean": -0.21, "...": 0.0}, "error": 0, "group": "question-17"}
```

`features` comes from `nanogate.router.build_features`, `error` is 1 when the local answer was wrong, and `group` keeps
paraphrases of one question in the same split. `nanogate train-router` reports held-out AUC, Brier score, coverage and
accepted-error rate, and writes the artifact to `var/router.json`. Retrain when the model or runtime changes: logprob
features are only meaningful for the stack that produced them.

## Tests

```bash
.venv/bin/python -m pytest      # unit, pipeline, API, CLI and the full attack suite
```

## Limitations

- The remote tier is a simulator. No request leaves the machine in this build; `egress` on a receipt is `none` or `simulated`.
- The default embedder is lexical (hashed word and character n-grams). It needs no model download, but recall on loose
  paraphrases is low. `SentenceTransformerEmbedder` is included for a real embedding model.
- The slot verifier favours precision: when in doubt it rejects the cached answer and asks the model.
- DLP is pattern based. It does not detect names or free-text addresses.
- Streaming sends the answer as SSE chunks after output screening has finished, so time to first token equals total time.
- Truncating the newest receipts is only detectable against an externally anchored head (`nanogate verify --head`).
- Single node, SQLite. Cache purge from the API is global rather than per tenant.
- No benchmark numbers are published here. Anything you measure depends on your model and hardware.

## Background

NanoGate started as a 24-hour hackathon concept for an on-device enterprise AI gateway. This repository is a
from-scratch implementation of that idea. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
