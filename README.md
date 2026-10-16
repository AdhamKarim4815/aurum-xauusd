# Aurum — XAUUSD Research and Paper Trading

Aurum is a local, safety-first foundation for researching XAUUSD strategies and running approved candidates in paper mode on a MacBook. It contains no real-money execution venue and makes no claim that a strategy will be profitable.

## Disclaimer

This project is for educational and research purposes only. It is not investment advice, and nothing in it places orders: there is no broker connection or live execution path of any kind. Simulated results do not predict real trading outcomes, and no strategy here is claimed to be profitable. Use it at your own risk.

## Safety boundary

- Simulation and paper state only; there is no `order_send`, MT5 SDK, broker credential, or live execution adapter.
- A deterministic risk engine caps each trade at 0.5% of current paper equity, allows one net XAUUSD position per paper account, requires a stop, and enforces daily, five-day, and peak drawdown locks.
- Models may filter a rule-generated signal. They cannot create an order, increase a size, or override a risk rejection.
- Research evidence never starts paper trading. The exact candidate hash must pass every gate before the dashboard enables its one-click approval button.

## Quick start on macOS

```bash
brew install uv
uv sync
uv run gold-trader init
uv run gold-trader doctor
uv run gold-trader serve
```

Open [http://127.0.0.1:8775](http://127.0.0.1:8775). The dashboard refuses non-loopback clients.

To prove the offline workflow with synthetic data:

```bash
uv run gold-trader demo-data --rows 5000
uv run gold-trader research run \
  --dataset var/data/demo_xauusd_5m.csv \
  --strategy trend-pullback
```

Synthetic data is intentionally ineligible for a trading claim. Use it only to exercise imports, reports, the simulator, and the dashboard.

## Real data

Export XAUUSD history from Dukascopy, MT5, or another permitted source, then import it:

```bash
uv run gold-trader data import history.csv \
  --source-name dukascopy-export \
  --destination var/data/xauusd_5m.parquet
```

The importer normalizes timestamps to UTC, validates OHLC and quote consistency, removes duplicate timestamps, reports large gaps, writes Parquet, and emits a content-addressed manifest. Do not use a dataset whose manifest fails quality.

Provider keys belong in macOS Keychain:

```bash
uv run gold-trader secret set twelve-data
uv run gold-trader secret check twelve-data
```

### Live quote providers and failover

Continuous live paper mode can draw quotes from more than one API and automatically fail over if
the current source errors, rate-limits, or goes stale. Supported providers:

| Provider | Keychain secret name | Notes |
| --- | --- | --- |
| Twelve Data | `twelve-data` | Default. Latest timestamped one-minute close (bid = ask = close); plan controls latency. |
| Alpha Vantage | `alpha-vantage` | Free tier. Returns a real two-sided bid/ask via `CURRENCY_EXCHANGE_RATE`. |
| metals.dev | `metals-dev` | Dedicated precious-metals API; bid/ask when the plan returns them, otherwise a single price. |
| Finnhub | `finnhub` | Free tier returns last price only (`OANDA:XAU_USD`); bid and ask are the same value. |

Store a key for each provider you want in the chain, then set the priority order (first-listed is
tried first) via `.env` or the `GOLD_TRADER_LIVE_PAPER__PROVIDER_PRIORITY` environment variable:

```bash
uv run gold-trader secret set alpha-vantage
uv run gold-trader secret set metals-dev
uv run gold-trader secret set finnhub

echo 'GOLD_TRADER_LIVE_PAPER__PROVIDER_PRIORITY=["twelve-data","alpha-vantage","metals-dev","finnhub"]' >> .env
```

Only providers with a stored secret are used; the worker skips the rest. Smoke-test any single
provider before relying on it in the chain:

```bash
uv run gold-trader data quote --provider alpha-vantage
```

The dashboard's "Feed provider" row shows which source answered the most recent poll, and switches
automatically (with a macOS notification) if the active one fails and a lower-priority one takes
over. Whichever provider is used, this never becomes a real broker connection — it only feeds the
paper simulator, per the safety boundary above. Where a provider only returns a single last price
(Twelve Data, Finnhub, and metals.dev without a bid/ask plan), the quote's bid and ask are set to
that same price rather than a fabricated spread — treat it as a mid price, not a two-sided market
quote.

Install the expanded local ML stack when needed:

```bash
uv sync --extra ml
uv run pytest -m optional_ml tests/test_ml.py::test_lightgbm_backend_smoke
uv run pytest -m optional_ml tests/test_ml.py::test_tcn_backend_smoke
uv run pytest -m optional_ml tests/test_ml.py::test_mlflow_backend_smoke
```

The optional backend smokes run in separate processes intentionally: loading LightGBM and
PyTorch together in one macOS process can cause an OpenMP runtime stall. The application keeps
their experiment workers isolated for the same reason.

## Running the tests

```bash
uv sync
uv run pytest
```

On a headless Linux machine with no OS keychain, run the tests with `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` set.

## Main commands

| Command | Purpose |
| --- | --- |
| `gold-trader doctor` | Verify Python, database, loopback binding, sleep integration, and optional ML backends. |
| `gold-trader data import` | Normalize CSV/Parquet bars and create a manifest. |
| `gold-trader research list-strategies` | Show preregistered rule families and hashes. |
| `gold-trader research run` | Create an immutable evidence package and eligibility decision. |
| `gold-trader research run-meta` | Run the fixed, leakage-safe Generation-2 signal filter. |
| `gold-trader research summarize` | Compare all rule and ML lineages for one exact dataset. |
| `gold-trader research generation ...` | Create, seal, run, pause, resume, inspect, and finalize Generation 3. |
| `gold-trader research generation4 ...` | Learn from Generation 3, then create, seal, run, inspect, and finalize Generation 4. |
| `gold-trader research generation5 ...` | Run frequency-first, session-normalized Generation 5 research. |
| `gold-trader research generation6 ...` | Run the sealed higher-activity Generation 6 hypotheses. |
| `gold-trader serve` | Start the dashboard plus isolated paper and research workers. |
| `gold-trader paper start/stop/kill` | Queue idempotent paper controls. |
| `gold-trader report` | Generate the current future-live-readiness report. |

## Generation 3: local research coordinator

Generation 3 uses only this Mac. The default manifest schedules 384 deterministic trials for
each of six transparent strategy families (2,304 total) across six CPU processes. It excludes
the latest 12 complete months from all search work, checkpoints every batch, pauses if the Mac
is unplugged or a resource gate fails, and stops for futility when no viable result improves.

```bash
uv run gold-trader research generation create \
  --dataset var/data/xauusd_5m.parquet \
  --manifest var/generations/generation-3.json \
  --workers 6 \
  --trials-per-family 384 \
  --batch-size 24

uv run gold-trader research generation seal \
  --manifest var/generations/generation-3.json

# With `gold-trader serve` running, queue it for the isolated research worker:
uv run gold-trader research generation start \
  --manifest var/generations/generation-3.json

uv run gold-trader research generation status
uv run gold-trader research generation pause
```

Use `run-now` instead of `start` when you want the coordinator attached to the current terminal.
Starting again resumes from durable completed-trial checkpoints. A shortlist is not eligible and
cannot trade. Opening the untouched holdout is an explicit operation, and the strict Generation 3
gate also requires a second broker/source dataset:

```bash
uv run gold-trader research generation finalize \
  --manifest var/generations/generation-3.json \
  --trial-hash FULL_SHORTLIST_TRIAL_HASH \
  --external-dataset /absolute/path/to/independent_xauusd_history.parquet
```

Even a successful finalization still requires pressing **Approve strategy** on the local
**Strategies** page. Search never starts paper trading automatically.

## Generation 4: adaptive local research

Generation 4 learns a narrower, sealed search space only from Generation 3 discovery shortlists.
It does not inspect the held-out final 12 months. It adds long-only and trend-regime variants and
requires an annualized-return floor plus a risk-adjusted comparison with unlevered XAUUSD buy and
hold. Create and start it after Generation 3 completes:

```bash
uv run gold-trader research generation4 create \
  --parent-generation-id generation-3-r4 \
  --learning-report var/reports/generation-3-learning.json \
  --manifest var/generations/generation-4.json \
  --workers 6 \
  --trials-per-family 512 \
  --batch-size 24

uv run gold-trader research generation4 seal \
  --manifest var/generations/generation-4.json

uv run gold-trader research generation4 start \
  --manifest var/generations/generation-4.json
```

The **Strategies** page combines completed Generation 3 through 6 shortlists, sorted by discovery
return. Discovery entries expose final validation only after their generation completes and an
independently sourced broker file has been imported into `var/data`.
After **Run final validation** is pressed, the same card shows durable queued/running progress and
refreshes automatically when the worker finishes. Results remain under **Finalized Candidates**,
including holdout return, independent-source return, holdout trades, and failed gates. The complete
JSON evidence is stored in `var/reports/<candidate-hash>.json`. Eligible final reports expose a
single **Approve strategy** button.

## Generation 5: frequency-first research

Generation 5 targets repeatable activity without relaxing risk or cost gates. It uses one common
Sunday-22:00 through Friday-22:00 UTC XAU calendar for primary and independent sources, tests 15m
and 1h compression, momentum, pullback, and mean-reversion rules, and enforces pre-registered
16/32/48-bar maximum holding times in research, replay, and continuous paper.

```bash
uv run gold-trader research generation5 create \
  --parent-generation-id generation-4 \
  --trials-per-family 384

uv run gold-trader research generation5 seal
uv run gold-trader research generation5 start
```

The default run contains 1,536 deterministic trials. Discovery requires at least 40 trades in each
of the five most recent full years and a median of at least 48. Final validation requires at least
48 holdout trades plus activity across at least nine months. See
[Generation 5 frequency-first research](docs/generation-5.md) for the full protocol and adaptive
period-reuse limitation.

## Generation 6: new higher-activity hypotheses

Generation 6 is a separate adaptive lineage built after Generation 5 found no shortlist. It tests
persistent trend state, multi-horizon momentum consensus, RSI recovery, and liquid-session trend
rules. It keeps the same deterministic risk limits and strict cost, stability, benchmark, final
holdout, independent-source, and manual-approval gates.

```bash
uv run gold-trader research generation6 create
uv run gold-trader research generation6 seal
uv run gold-trader research generation6 start
uv run gold-trader research generation6 status
```

The default is 1,536 deterministic trials across four families. See
[Generation 6 higher-activity research](docs/generation-6.md) for the exact hypotheses, activity
floors, adaptive-reuse disclosure, and early-stop policy.

## Continuous live paper mode

The continuous loop uses quotes from the configured provider chain (see "Live quote providers and
failover" above) as observations and only creates simulated orders and fills. Configure at least
one provider's key in macOS Keychain, verify one quote, then start live paper after an eligible
candidate has been manually approved:

```bash
uv run gold-trader secret set twelve-data
uv run gold-trader data quote
uv run gold-trader serve

# In another terminal, after dashboard approval:
uv run gold-trader paper start --mode live
uv run gold-trader paper status
```

The loop polls once every two minutes by default, constructs durable five-minute bars, evaluates only
newly completed strategy bars, and executes a signal on the next fresh quote. Network errors,
stale/future timestamps, approval invalidation, or reconciliation uncertainty block entries. Stops,
targets, spread, slippage, commission, swap, one-position limits, 0.5% sizing, daily/five-day loss
halts, and the 10% account lock remain authoritative. `paper stop` and the kill switch cancel pending
signals and flatten an open simulated position at the last known paper mark. No broker order API is
present or selectable.

## Four-strategy experimental live paper lab

The **Live lab** dashboard can run exactly four finalized strategies at the same time. It is a
deliberate paper-only observation mode for candidates whose only failed final gates are trade count
or monthly activity. Return, transaction-cost, drawdown, data-quality, or external-source failures
remain non-overridable.

Each strategy receives its own simulated $10,000 account, 0.5% risk sizing, loss halts, ledger, and
one-position constraint. This means the lab may hold up to four simultaneous XAUUSD positions in
aggregate—one per strategy. It does not pyramid a strategy, net the four accounts, or treat their
returns as one portfolio. One shared quote is fetched per coordinator cycle and cloned into four
independently traceable decisions.

The default two-minute polling cadence is 720 shared requests per day, leaving some headroom under
Twelve Data Basic's current 800-request daily limit. Manual quote tests and other API calls consume
the same quota, so provider exhaustion still fails closed and blocks new entries.

Open [http://127.0.0.1:8775/paper-lab](http://127.0.0.1:8775/paper-lab), select exactly four
qualified finalized candidates, acknowledge the experimental boundary, and press **Start all four
live paper strategies**. The page refreshes status, feed health, equity, return, trades, positions,
quote age, risk state, and last decision bar every three seconds. **Stop all safely** flattens each
simulated account at its last mark; **Kill all four** cancels pending entries and halts the entire
lab. See [Four-strategy live paper lab](docs/four-strategy-paper-lab.md).

## Project map

- `src/gold_trader/` — domain records, providers, features, strategies, simulator, research, risk, persistence, worker, CLI, and dashboard.
- `docs/` — product requirements, architecture, research and ML rules, data governance, risk policy, security, testing, runbooks, and roadmap.
- `var/` — ignored local databases, imported data, manifests, and evidence reports.
- `artifacts/` — ignored local model and experiment artifacts.

Read the [documentation index](docs/README.md) before changing research, risk, or approval behavior.

## License

MIT. See [LICENSE](LICENSE).
