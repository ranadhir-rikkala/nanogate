# Threat model

Scope: a single gateway on hardware the organisation controls, serving several departments and tenants.

| Threat | Control | Where | Test |
|---|---|---|---|
| Credential pasted into a prompt reaches a model or a log | Secret detectors run first; request blocked before cache or model; receipt stores entity type only | `dlp.py`, `policy.py` | `tests/test_pipeline.py::test_secret_is_blocked_before_any_model_or_cache`, attack `secret-*` |
| Obfuscated PII or secrets slip past patterns | NFKC folding, zero-width stripping, "at/dot" and spaced-digit joining before a second scan | `dlp.normalize` | `tests/test_dlp.py::test_obfuscated_values_are_still_found` |
| Sensitive data sent to a remote provider | Policy removes the remote route above the department's data-class ceiling; egress ticket re-checks | `policy.py`, `inference/remote.py` | attack `pii-stays-local-when-unsure`, `pinned-remote-with-pii` |
| Caller pins a route to bypass policy | Pinned route must already be in the permitted set | `PolicyEngine.evaluate` | `tests/test_policy.py::test_pinned_route_must_already_be_permitted` |
| Caller claims another tenant or department in headers | Identity comes from the key; conflicting headers are refused with 403 and receipted | `auth.reject_spoofing` | `tests/test_api.py::test_spoofed_department_header_is_refused` |
| Stolen database reveals API keys | Only HMAC hashes are stored, keyed by a pepper outside the database | `auth.KeyStore` | `tests/test_auth.py::test_key_round_trip_and_hash_only_storage` |
| One tenant reads another's cached answers | Namespace hashes tenant and department | `cache/store.py` | attack `cache-cross-tenant`, `cache-cross-department` |
| Near-duplicate question served the wrong cached answer | Slot verifier rejects number, negation, antonym, name and term mismatches | `cache/verifier.py` | attack `cache-negation-flip`, `cache-number-swap`, `cache-antonym-swap` |
| Model output leaks a credential or invents PII | Output scan withholds credentials and redacts entities the input did not contain | `Gateway._screen` | attack `output-leaks-credential`, `output-invents-pii` |
| Prompt injection asks the gateway to change route or disable checks | Routing is decided by code from identity and classification, never by model output | `pipeline.py` | attack `prompt-injection-cannot-widen-routes` |
| Concurrent requests overspend a budget | Reservation inside `BEGIN IMMEDIATE` | `budget.py` | `tests/test_budget.py::test_concurrent_reservations_cannot_overspend` |
| Audit record edited, deleted or rebuilt | Hash chain, per-receipt HMAC seal, column cross-check, anchored head | `receipts.py` | `tests/test_receipts.py` |
| Tenant admin reads another tenant's audit data or keys | Every `/api/*` query is scoped to the caller's tenant | `api/app.py` | `tests/test_api.py::test_tenants_cannot_read_each_others_receipts_or_keys` |
| A dependency fails mid-request | DLP, policy, budget and receipts fail closed; cache and router fail to the safe side and record `DEGRADED_*` | `pipeline.py` | `tests/test_pipeline.py::test_broken_cache_fails_open_and_broken_dlp_fails_closed` |

## Out of scope

- An attacker with the receipt signing key and write access to the database can forge a consistent chain. Anchor the
  head (`nanogate verify --head`) somewhere they cannot write.
- Network-level egress control. The remote tier is simulated here; a live connector would also need firewall rules.
- Detection of personal names, addresses, or secrets with no recognisable structure.
- Denial of service beyond the per-key rate limit.
