# Architecture

NanoGate is one FastAPI process backed by one SQLite file in WAL mode. Every component is a plain class that receives
its dependencies through its constructor; `nanogate/api/app.py:build_services` is the only place they are wired.

## Request lifecycle (`nanogate/pipeline.py`)

| # | Stage | Fails with | Fail mode |
|---|---|---|---|
| 1 | Authenticate, reject spoofed identity headers | `AUTH_INVALID`, `AUTH_REVOKED`, `SCOPE_DENIED`, `IDENTITY_SPOOF_REJECTED` | closed |
| 2 | Rate limit, input size | `RATE_LIMITED`, `INPUT_TOO_LARGE` | closed |
| 3 | Classify (DLP over every message) | `DLP_FAIL_CLOSED` | closed |
| 4 | Policy, optional input redaction and re-classification | `SECRET_BLOCKED`, `POLICY_BLOCK`, `ROUTE_NOT_PERMITTED` | closed |
| 5 | Budget hold for the most expensive permitted route | `BUDGET_DENY` | closed |
| 6 | Verified cache (single-turn requests only) | `CACHE_HARD_NEGATIVE` falls through to the model | open (miss) |
| 7 | Local small, then risk score against `min(learned threshold, policy cap)` | `DEGRADED_inference` | next tier |
| 8 | Local large when the small answer is not confident | | next tier |
| 9 | Remote, only with an egress ticket | `REMOTE_DISABLED` | stay local |
| 10 | Low confidence and nowhere to go | `ROUTER_ABSTAIN` or `LOCAL_UNVERIFIED`, per department | |
| 11 | Output scan: credentials withhold the answer, new PII is redacted | `OUTPUT_BLOCKED`, `OUTPUT_REDACTED` | closed |
| 12 | Settle budget, write cache, seal receipt | a receipt failure fails the request | closed |

Each request collects every reason code that applied. The one reported in `x-nanogate-reason` is chosen by a fixed
precedence list (`Reason.PRECEDENCE`), so the same situation always reports the same primary reason.

## Why the pieces are shaped this way

**Policy version is content-addressed.** The version string is the declared version plus a hash of the whole document.
It is part of every cache namespace and every receipt, so changing any rule invalidates cached answers and is visible
in the chain without anyone remembering to bump a number.

**Cache isolation is structural.** Tenant, department, policy version, model, system prompt and embedder are hashed
into the namespace. There is no cross-tenant check to forget, because a cross-tenant lookup has a different key.

**The cache is written conservatively.** Only answers the router accepted, from a local tier, for a single-turn
request, with a cacheable data class, with no redaction on either side. A wrong cache hit is worse than a miss.

**Egress needs a ticket.** `RemoteSimulator.complete` refuses to run without a single-use `EgressTicket`, and
`EgressGuard.issue` re-checks the policy decision, the remote mode and the secret flag before writing the egress log.
Reaching the remote tier by accident requires bypassing two independent checks.

**An untrained router is pessimistic.** Without an artifact the score is `p_error = 0.5` and `trained = false`. The
pipeline treats that as "not confident", never as a number to compare against a threshold.

**Receipts never contain values.** They hold entity types and counts, the decision path with timings, token counts and
cost. Prompts and answers are not stored, and nothing sensitive is hashed either: a hash of a low-entropy value such
as an SSN is reversible by enumeration.

**Receipt verification checks five things per row**: sequence continuity, the previous-hash link, the body digest, the
HMAC seal, and that the indexed SQL columns still agree with the sealed body.

## Data model

`api_keys` (hash only), `budget_ledger`, `receipts` (the chain), `cache_entries`, `egress_log`. Migrations live in
`nanogate/db.py` and are tracked with `PRAGMA user_version`.
