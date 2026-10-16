# Known limitations

- The default historical path imports a Dukascopy/broker export; it does not automate the website export flow.
- Twelve Data entitlement, latency, depth, and rate limits depend on the operator’s current plan. The UI must label rather than infer them.
- Bar data cannot reveal intrabar order. The simulator deliberately resolves ambiguous stop/target collisions against the strategy.
- Mid-only data uses a configured spread model and cannot reproduce broker-specific bid/ask, rejection, margin, swap, or execution behavior exactly.
- Continuous paper depends on Twelve Data one-minute-series availability and entitlement. It polls rather than consuming an exchange tick stream, constructs candles locally, and can miss intraminute extremes; all fills remain simulations.
- The four-strategy lab deliberately admits final candidates that failed only trade-count or monthly-activity gates. This is an observation override, not evidence that the strategies passed validation. Its four accounts can hold four simultaneous positions and therefore represent four separate risk books, not one deployable portfolio.
- Generation 3 tests a large but finite sealed categorical space. Selection-bias statistics and an independent dataset reduce, but cannot remove, multiple-testing risk or regime dependence.
- Alternative-data adapters beyond the documented interfaces and governance rules are not yet enabled by default.
- LightGBM, PyTorch, transformers, SHAP, Optuna, and MLflow are optional local dependencies; the base workflow implements the calibrated logistic reference path and backend diagnostics.
- Research criteria reduce but cannot eliminate overfitting, nonstationarity, selection bias, or data errors.
- `caffeinate` prevents normal sleep while the worker holds its lease; power loss, forced shutdown, kernel failure, or network loss remain possible.
- There is no real broker reconciliation, margin engine, legal/compliance review, or live-order path.
