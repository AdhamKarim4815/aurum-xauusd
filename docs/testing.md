# Test strategy and acceptance

## Layers

- **Unit:** domain validation, indicators, features, quality, sizing, risk halts, fill ordering, and metrics.
- **Property:** generated intents never exceed caps; timestamps remain UTC; future rows cannot change an earlier feature vector.
- **Research:** chronological folds, purge/embargo, holdout lock, deterministic candidate hashes, stress costs, and gate truth tables.
- **Frequency research:** common-source session filtering, maximum-holding exits, annual activity
  floors, monthly final activity, and research/paper execution parity.
- **Generation 6:** persistent-signal behavior, deterministic hypothesis sampling, learning-report
  tamper detection, broad-coverage futility, isolated paper routing, and dashboard controls.
- **Four-strategy paper lab:** exact-four selection, activity-only override policy, shared
  one-fetch-per-cycle quotes, four namespaced account states, independent risk/position state,
  CSRF controls, and stop/kill-all behavior.
- **Integration:** SQLite WAL, idempotent commands, approval invalidation, Keychain boundary, provider failures, worker start/stop/kill, and restart reconciliation.
- **HTTP/UI:** loopback rejection, CSRF, eligible-only one-click approval, strategy-library ranking, missing-precondition errors, status/audit JSON, and responsive accessible forms.
- **Static safety:** no real execution dependency or order endpoint.

## Golden replay

A fixed fixture and strategy must produce the same trade direction, entry index, adverse fill policy, exit reason, and rounded lot size. Numeric metrics use explicit tolerances to avoid claiming byte equality across BLAS/platform implementations.

## Leakage sentinels

- Appending or mutating bars after an `as_of` cutoff cannot change its feature vector.
- A synthetic perfect future-return column must be rejected from the allowed feature schema.
- Revised macro values cannot be joined before their vintage availability.
- Fold labels and overlapping horizons cannot cross purge/embargo boundaries.
- A meta model cannot create a signal absent from its deterministic parent.
- Changing bars after the registered label horizon cannot change an earlier meta-label.
- A model artifact outside the trusted local model directory, with the wrong checksum, or with
  a mismatched learned-parameter hash is rejected before replay.

## Mac acceptance

1. `uv sync --locked` installs Python 3.12 dependencies.
2. `gold-trader doctor` passes base checks and reports optional ML availability.
3. Synthetic data imports and completes a research smoke run without network.
4. Dashboard binds only to `127.0.0.1` and displays current durable state.
5. Approved replay start acquires a visible sleep lease; stop/kill releases it.
6. Stale/unconfigured live mode never allows entries.
7. Restart does not duplicate commands or fills.
8. No real venue is selectable or importable.
9. The live lab can expose four running account cards from one provider request per poll, while each
   account retains its own balance, risk state, position, and event correlation ID.

## CI policy

Linux verifies portable research/core behavior; macOS verifies the supported operator environment. Required checks are Ruff, strict mypy, pytest, and at least 75% statement coverage for the foundation. Expanded ML jobs may run separately because PyTorch/transformer artifacts are intentionally optional in the base environment.

LightGBM, TCN, and MLflow smoke tests run as separate pytest invocations. This preserves process
isolation between native OpenMP/Metal runtimes and prevents an accelerator-library interaction
from hanging the safety-core suite.
