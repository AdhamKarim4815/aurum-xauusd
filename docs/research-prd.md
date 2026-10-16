# Research and strategy PRD

## Research question

Can a transparent XAUUSD rule retain positive, risk-adjusted performance after realistic and doubled costs across chronological regimes, and can a calibrated ML scorer improve that rule without increasing risk or using future information?

The correct result may be “no eligible candidate.” Research must never be tuned to force approval.

## Preregistered families

| Family | Horizon | Hypothesis | Initial fixed parameters |
| --- | --- | --- | --- |
| Session breakout | 5m/15m | London/New York range expansion persists when price clears a trailing channel. | 20-bar channel, 14 ATR, 1.5 ATR stop, 2R target. |
| Volatility mean reversion | 15m | Exhaustion outside two standard deviations reverts when the slow trend is weak. | 20-bar band, RSI 14, trend veto, 1.25 ATR stop, 1.5R target. |
| Trend pullback | 1h | A pullback reclaim/rejection of the fast EMA continues in the slow-trend direction. | EMA 20/50, 14 ATR, 1.5 ATR stop, 2R target. |
| Time-series momentum | 4h | A 55-bar channel break persists at a lower turnover. | 55-bar channel, 14 ATR, 2 ATR stop, 2.5R target. |

Changing these values creates a new strategy hash and lineage. Broad, retrospective parameter searches are prohibited.

## Generation-2 preregistration

Registered on 2026-07-22 after the four baseline reports were opened and before any
Generation-2 report was executed. This is a bounded hypothesis check, not an optimizer:

| Variant | Fixed change from baseline | Rationale |
| --- | --- | --- |
| `session-breakout-l12` | channel 12 | Test a shorter session-memory horizon. |
| `session-breakout-l36` | channel 36 | Test a slower session-memory horizon. |
| `volatility-mean-reversion-l12` | band lookback 12 | Test faster exhaustion normalization. |
| `volatility-mean-reversion-l36` | band lookback 36 | Test slower exhaustion normalization. |
| `trend-pullback-ema12-36` | EMA 12/36 | Test faster trend and reclaim recognition. |
| `trend-pullback-ema30-80` | EMA 30/80 | Test slower regime confirmation. |
| `time-series-momentum-l20` | channel 20 | Test whether higher turnover retains the momentum edge. |
| `time-series-momentum-l35` | channel 35 | Test an intermediate turnover horizon. |

ATR periods, stops, targets, RSI thresholds, cost assumptions, split rules, and every
eligibility threshold remain unchanged. All eight variants are evaluated and reported;
no post-result parameter additions or deletions are allowed in this generation. Because
the baseline market regimes have already been inspected, multiplicity and regime-selection
risk must be disclosed in every Generation-2 promotion review and tested again during the
mandatory paper-observation period.

## Generation-2 ML-filter preregistration

Registered on 2026-07-22 after all eight Generation-2 rule reports were opened and before
any Generation-2 ML report was executed. Every registered Generation-2 rule variant is
subjected to the same fixed signal-filter protocol; a failed or data-insufficient model is
reported as ineligible and is not replaced post-result.

- Model: elastic-net logistic regression with scaling, balanced class weights, sigmoid
  probability calibration, random seed 42, and a fixed eligibility threshold of 0.55.
- Scope: the model can only suppress a signal proposed by its deterministic parent rule. It
  cannot invent a signal, choose size, move stops or targets, or bypass the risk engine.
- Examples: only parent-rule signal timestamps are labeled. Features contain price-derived
  values available at the signal bar close; no future, cross-market, macro, or news fields are
  used in this generation.
- Label: side-aware stop/target triple barrier using the next bar open, stop-first ordering for
  unresolved same-bar crossings, and a loss label when neither barrier is reached. The fixed
  horizon is 24 bars for 15-minute and 1-hour variants and 12 bars for 4-hour variants.
- Development split: chronological 70% model fit, 15% calibration, and 15% internal evaluation.
  At least 300 complete rule-signal examples and both outcome classes in each slice are
  required.
- Walk-forward: the model is refitted only on information strictly before each validation
  boundary. Purge and embargo are the greater of the parent lookback and label horizon.
- Untouched evidence: the latest 12 complete months remain the final holdout. Its probability
  skill must be positive by Brier skill score, and the filtered simulation must improve parent
  Sharpe or maximum drawdown.
- Existing return, doubled-cost, profit factor, Sharpe, drawdown, trade count, fold stability,
  concentration, data-quality, and sufficiency gates remain unchanged.

The learned coefficients, calibration parameters, dataset/configuration/strategy/model hashes,
model card, and a checksum-protected local artifact are retained for exact replay. Inspecting
these results cannot justify a threshold, feature, label, horizon, or parameter change within
this generation; such a change is a newly preregistered lineage.

## Data sufficiency and splits

- Intraday eligibility requires at least four years of quality 5-minute data; swing eligibility requires at least eight years of quality hourly data.
- The most recent 12 complete months are locked as the final test.
- Intraday development uses expanding 24-month train and 3-month validation windows rolled by 3 months.
- Swing development uses expanding 5-year train and 12-month validation windows rolled by 6 months.
- Purge and embargo equal the maximum label/holding horizon. Overlapping examples cannot cross fold boundaries.
- The final test may be opened once per candidate lineage. A material post-test change creates a new lineage; it may not reuse the old test claim.

## Fill and cost contract

- A bar-close signal enters at the next tradable bar open, never the signal close.
- Entry and exit pay half-spread plus adverse slippage. Commission is round-trip and swap accrues by time held.
- A gap through a stop fills at the adverse available opening price.
- If a stop and target are both possible inside a bar without tick ordering, the stop is assumed first.
- Reports contain a base-cost run and a run with spread, slippage, and commission doubled.
- Mid-only history requires an explicit spread model; it cannot silently be treated as bid/ask.

## Eligibility gate

An evidence package is eligible only when all conditions are true:

- Base and doubled-cost untouched returns are positive.
- Profit factor is at least 1.15 and annualized Sharpe is at least 0.75.
- Maximum drawdown is no more than 10%.
- There are at least 100 intraday or 40 swing holdout trades.
- At least 60% of walk-forward folds are profitable.
- No one profitable fold contributes more than half of total positive fold P&L.
- Data, leakage, reproducibility, calibration, and risk tests pass.
- An ML-filtered version also demonstrates positive probability skill and improves risk-adjusted return or drawdown versus its rule-only parent.

Eligibility permits review; it never grants execution authority.

## Required report contents

Candidate lineage hashes, source manifests, quality results, time coverage, split boundaries, assumptions, trade ledger, base/stressed metrics, fold results, gate decisions, model card when applicable, and the simulation disclaimer.
