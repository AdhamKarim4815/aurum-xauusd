# Roadmap

## Foundation — implemented in v0.1

Standalone Python 3.12 project, domain contracts, SQLite audit/commands, data import and quality, manifests, four rule strategies, point-in-time features, risk engine, deterministic replay, cost stress, eligibility report, manual approval service, local dashboard, Keychain, sleep lease, notifications, worker controls, tests, and operating documentation.

## Research depth

- Acquire sufficient bid/ask history and run all four preregistered families.
- Implement the complete expanding walk-forward orchestrator and final-test lineage lock persistence.
- Add LightGBM and TCN experiment runners, calibration reports, SHAP stability, MLflow model cards, and feature-family ablations.
- Add FRED/ALFRED, CFTC, official calendar, cross-market, and governed headline adapters.

## Paper observation

- Add the continuous quote-driven paper loop, persistent fills/positions/equity, freshness monitor, reconnection backfill, native alert matrix, trade/equity charts, and observed-versus-modeled costs.
- Track trading-day and closed-trade progress. Keep future-live readiness false until 30 days and 50 trades plus operator review.

## Separate future project

Only after the paper gate: broker selection, symbol specification capture, jurisdiction/compliance review, MT5 connectivity decision, demo execution, reconciliation, deployment reliability, credential rotation, and a new live safety/approval process.

