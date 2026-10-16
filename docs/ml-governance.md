# Machine-learning governance

## Allowed role

ML is a meta-labeler: it estimates whether a deterministic rule signal is worth accepting. It may reject a signal or reduce participation. It may not originate an order, move a stop, raise the 0.5% cap, bypass a halt, or control the kill switch.

## Approved model families

1. Elastic-net logistic regression as the interpretable calibrated baseline.
2. LightGBM for nonlinear tabular interactions.
3. A compact temporal convolutional network for bounded sequence experiments on Apple MPS with deterministic CPU fallback.
4. Frozen FinBERT for headline feature extraction; it is not fine-tuned into an end-to-end trading policy.

Reinforcement learning, unrestricted Transformer searches, online self-modification, and opaque third-party signal models are prohibited in v1.

## Features and availability

Feature families are evaluated in ablations: XAUUSD price/volatility, cross-market, macro/positioning, and timestamped news. Every value has an event time and an `available_at` time. Joins use the last value actually available as of the decision; revised macro series use archival vintages. Backward filling, centered rolling windows, random train/test splits, and post-release timestamps attached to pre-release observations are forbidden.

## Labels

Triple-barrier meta-labels use a strategy-specific stop, target, and vertical horizon. Labels are constructed only after split boundaries are defined, with purge/embargo for overlapping outcomes. Rule signal generation and labels remain separate artifacts.

## Training and evaluation

- Deterministic seeds, bounded local search budgets, early stopping, and time-aware folds are mandatory.
- Calibration data is separate from training data. Report Brier score, calibration curve, threshold, coverage, and confusion metrics.
- Compare every model with its rule-only parent and a no-skill probability baseline.
- Use feature-family ablations and SHAP summaries; flag unstable importance signs/ranks across folds.
- Store environment, code commit, dependencies, data hash, feature schema, parameters, metrics, and artifacts in local MLflow when the ML extra is installed.
- Never select a final threshold on the untouched test.

## Model card and promotion

Each candidate documents intended use, prohibited use, data period, feature availability, label horizon, folds, calibration, base/stressed trading metrics, failure regimes, missing data policy, compute device, and lineage hashes. A model/config/data change invalidates approval. Models do not retrain automatically while paper trading.

## Alternative and scraped data

Only allowlisted sources with recorded terms, robots outcome, parser version, raw snapshot hash, publication time, retrieval time, and kill switch may enter research. If historical availability cannot be reconstructed defensibly, the feature is exploratory and cannot support eligibility.

