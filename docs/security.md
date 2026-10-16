# Security and threat model

## Assets

Provider credentials, candidate approvals, audit history, paper state, imported/licensed data, model artifacts, and the operator’s confidence in the evidence.

## Trust boundaries

- Browser ↔ loopback FastAPI application.
- Dashboard ↔ durable command queue ↔ isolated worker.
- Application ↔ Keychain.
- Providers/web sources ↔ normalization and scraper quarantine.
- Research artifacts ↔ approval capability.

## Threats and controls

| Threat | Control |
| --- | --- |
| Remote dashboard access | Bind loopback and reject non-loopback client addresses. |
| Cross-site request forgery | Strict same-site session cookie and per-session CSRF token on all writes. |
| Secret leakage | Keychain storage, reference-only environment variables, redacted diagnostics, ignored local state. |
| Duplicate commands | Unique idempotency key and durable command status. |
| Approval substitution | Candidate-specific CSRF-protected button and backend-bound lineage hashes. |
| Future-data leakage | Explicit availability timestamps, chronological joins/splits, leak sentinel tests. |
| Malicious scraped content | Treat as data, not instructions; sanitize/render escaped; raw-cache and parser isolation. |
| Path traversal | CLI `Path` inputs are explicit; dashboard exposes no arbitrary file operation. |
| Unsafe model artifact | Only locally produced, hash-bound artifacts are loadable; never unpickle a downloaded model. |
| Worker uncertainty | Fail closed, block entries, reconcile, and require audit review. |
| Accidental live execution | No real venue dependency, port, endpoint, or registry entry. |

## Security acceptance

Static search must find no MT5/broker order-send import. HTTP writes without CSRF fail. Non-loopback requests fail. Secret values never appear in status, settings, audit, reports, or exceptions. Kill and loss halts remain effective if model output is adversarial.
