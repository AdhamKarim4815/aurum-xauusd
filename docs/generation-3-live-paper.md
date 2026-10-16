# Generation 3 and continuous live paper

## Safety contract

Generation 3 is a local research system, not a profit finder or an automatic promotion mechanism.
It cannot approve a candidate. Continuous live paper consumes market observations but can construct
only domain `SimulatedOrder` and `Fill` records. There is no real venue registration, manual trade
route, MT5 import, or broker order call.

## Sealed generation lifecycle

1. `create` validates and hashes the exact normalized dataset, records the final-test boundary,
   snapshots non-secret settings, and writes a draft manifest.
2. `seal` verifies the dataset has not changed and hashes the full manifest, family spaces, runtime
   settings, and implementation version. A sealed manifest cannot be replaced in SQLite.
3. `start` places one `research.generation.run` command in the research-only queue. `run-now` performs
   the same operation in the current process.
4. The coordinator owns `caffeinate`, checks AC power, free disk, and system load, dispatches bounded
   batches to up to six process workers, and is the only database writer.
5. Every completed trial is independently durable. Re-running the same manifest skips completed
   hashes. Pause and stop take effect after the active batch, preventing half-written results.
6. Futility stops a run only after the configured minimum batches and consecutive batches with no
   improvement when no strategy has reached the discovery shortlist.
7. `finalize` is the only operation that opens the primary final 12-month holdout. It also requires
   an independent external dataset for Generation 3 eligibility. The evidence report remains
   unapproved until the operator presses the approval button for that exact immutable candidate.

The deterministic queue uses a coprime mixed-radix traversal of each finite categorical space. This
gives repeatable, non-duplicated coverage without claiming that a random search is exhaustive.

## Search families and stages

The sealed spaces cover session breakout, volatility breakout, trend pullback, time-series momentum,
mean reversion, and compression breakout. Parameters include timeframe, lookback, ATR stop distance,
reward/risk, and small family-specific values. The strategies are rule based; Generation 3 does not
allow a model to invent trades or override risk.

Stage 1 cheaply rejects candidates with inadequate activity, excessive drawdown, or strongly
negative recent discovery performance. Stage 2 evaluates the complete pre-holdout discovery period,
doubled costs, six chronological subperiods, profit concentration, a deflated-Sharpe probability,
and a deterministic stationary-bootstrap p-value. The coordinator report also estimates probability
of backtest overfit from cross-trial fold rankings. These statistics are diagnostics and do not
guarantee future performance.

## Continuous paper lifecycle

The paper worker restores only an exact valid approval plus its local evidence, dataset, and Keychain
credential. It warms the approved strategy from historical five-minute bars but never creates trades
for the warmup or a recovered gap. Each provider quote is audited with provider availability time.

For every fresh quote the worker:

1. verifies the approved candidate hash still matches;
2. rejects stale timestamps and future timestamps beyond tolerance;
3. reconciles a prior network gap through provider backfill without retroactive trades;
4. marks or closes the one open simulated position, checking stop before target;
5. applies daily, rolling-five-day, and peak drawdown entry halts;
6. executes a previously generated signal only on this next fresh quote through `RiskEngine`;
7. updates the active five-minute candle and evaluates a newly completed strategy bar once; and
8. persists account, position, risk, fill, and closed-paper-trade audit records.

Quotes from a mid-only provider still use the configured adverse spread and slippage model. Balance
is realized only on close; equity includes the current mark. Commission and time-proportional swap
are included. Stop/kill cancels a pending signal and, if needed, closes an open simulated position at
the last known paper mark with that fact in the exit reason.

## Failure and restart semantics

- A provider exception uses exponential retry up to the configured maximum and blocks entries.
- A stale quote is recorded but cannot mark, enter, or close a position.
- Restart restores pending signal, position, last decision bar, balance, loss baselines, and trade
  count. It does not re-evaluate old bars or duplicate fills.
- Missing approval, evidence, dataset, or Keychain secret moves the session to `recovery-required`.
- An invalidated approval moves the session to `approval-invalidated` and clears the pending signal.
- Risk-halted sessions may keep polling and marking but cannot enter another trade.

## Operator boundary

An ineligible candidate cannot start replay or live paper. Current Generation 2 failures therefore
remain stopped until Generation 3 produces a strict eligible report and the operator separately
approves it from the Strategies page. This is intentional; “continuous” describes the worker lifecycle, not
permission to bypass research evidence.
