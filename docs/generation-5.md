# Generation 5 frequency-first research

## Purpose and boundary

Generation 5 addresses a specific Generation 4 failure: promising strategies produced too few
closed trades in the final 12-month period. It is a separate immutable lineage. Generation 3 and 4
manifests, trials, final reports, and approvals are not rewritten.

This is not a mandate to maximize turnover. More trades increase spread, slippage, commission, and
model-selection risk. A candidate survives only when its higher activity remains profitable under
doubled costs and the unchanged account risk controls.

Generation 5 is adaptive: the July 2025-July 2026 period was viewed for earlier lineages. Every
manifest and final report records this reuse. Results remain simulation evidence and require the
normal 30-trading-day and 50-closed-paper-trade observation gate.

## Data-session policy

All primary and independent-source bars use the same conservative UTC XAU session before
aggregation or simulation:

- Monday through Thursday: included.
- Friday: included before 22:00 UTC.
- Saturday: excluded.
- Sunday: included from 22:00 UTC.

The report records input rows, output rows, removed rows, and removed fraction for each source.
This prevents a nominally independent 24/7 feed from manufacturing weekend signals that the
broker-style source could not trade. The fixed UTC boundary is an explicit approximation; a future
broker-symbol calendar can replace it only in a new lineage.

## Fixed hypotheses

Generation 5 samples four transparent families on 15-minute and 1-hour bars:

- compression breakout;
- time-series momentum;
- trend pullback; and
- volatility-conditioned mean reversion.

The fixed space tests 8-55 bar lookbacks, 10/14 ATR periods, 1.0-2.0 ATR stops, 1.25-2.0 reward/risk,
both-direction and long-only variants, no/50/100-bar regime filters, and 16/32/48-bar maximum
holding times. Family-specific compression, EMA, band, and RSI parameters remain small and bounded.

Maximum holding time is part of the strategy hash. Research replay and continuous paper mode both
enforce it; it never widens a stop or bypasses risk.

## Discovery gates

The latest 24 discovery months must first produce at least 72 trades, lose less than 2%, and remain
inside 10% drawdown. Full discovery then requires:

- positive base and doubled-cost returns;
- profit factor at least 1.10, Sharpe at least 0.50, and drawdown no greater than 10%;
- at least 360 closed discovery trades;
- at least 40 trades in each of the five most recent complete discovery years and a median of at
  least 48;
- at least 60% profitable chronological folds and no fold above half of positive fold profit;
- at least 2% annualized return and Calmar no worse than price-only buy-and-hold; and
- selection-bias and stationary-bootstrap diagnostics retained in the evidence package.

The score contains only a small capped activity term. Return, stress performance, drawdown,
stability, and benchmark quality remain authoritative.

## Final validation

A shortlist must produce at least 48 closed primary holdout trades and at least nine months with two
or more trades. It must also pass the existing return, doubled-cost, profit-factor, Sharpe, drawdown,
stability, concentration, selection-bias, independent-source, and risk-adjusted benchmark gates.
The external source must produce at least 75% of the primary activity target after common-session
normalization.

Finalization remains queued, one-time per candidate lineage, and separate from approval. A passing
candidate does not start paper trading automatically.

## Commands

```bash
uv run gold-trader research generation5 create
uv run gold-trader research generation5 seal
uv run gold-trader research generation5 start
uv run gold-trader research generation5 status
uv run gold-trader research generation5 pause
uv run gold-trader research generation5 stop
```

Progress appears under **Generations**. Completed shortlists appear under **Strategies**, where
final-validation progress and results remain durable.
