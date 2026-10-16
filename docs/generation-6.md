# Generation 6 higher-activity research

## Purpose and boundary

Generation 6 is a new sealed lineage created after Generation 5 ended without a shortlist. It
does not weaken the account-risk, doubled-cost, drawdown, stability, independent-source, or manual
approval gates. Instead, it tests genuinely different rules intended to produce more observations
and closed trades.

This is adaptive research. Generation 6 was designed after inspecting earlier discovery results,
and the same historical period has been used across lineages. Its final 12-month test is locked
within the Generation 6 lineage, but it is not a globally untouched test. Results are simulations;
profitability is neither assumed nor promised.

## Fixed hypotheses

- **Trend state:** participate while fast/slow trend, price location, and normalized slope agree.
  A strategy may re-enter only after its previous simulated position has closed.
- **Momentum consensus:** require short- and longer-horizon price moves to agree and require the
  short move to clear an ATR-normalized threshold.
- **RSI recovery:** enter when RSI recovers out of an extreme, optionally only when price agrees
  with a fixed trailing trend regime.
- **Session trend:** participate in a pre-registered London/New York UTC window when fast trend,
  slow trend, and price location agree.

All families are transparent rule strategies. The search covers 15-minute and 1-hour bars, fixed
ATR stops and targets, both-direction and long-only policies, and explicit 4-24 bar maximum holding
times. It does not add reinforcement learning, unrestricted model search, martingale, grid trading,
averaging down, or real-order execution.

## Activity and robustness gates

Stage one examines only the last 24 discovery months and requires at least 80 closed trades, return
above -2%, and drawdown no greater than 10%. A full discovery candidate must then have:

- positive base and doubled-cost return;
- profit factor at least 1.10 and Sharpe at least 0.50;
- maximum drawdown no greater than 10%;
- at least 400 discovery trades;
- at least 45 trades in every recent full year and a five-year median of at least 55;
- at least 60% profitable folds and no fold contributing more than half of positive fold profit;
- annualized return of at least 2%; and
- Calmar ratio no worse than unlevered XAUUSD buy and hold.

Final validation retains the stricter configured profit-factor and Sharpe requirements. It also
requires at least 50 holdout trades, at least three trades in ten active months, positive doubled-
cost performance, selection-bias evidence, bootstrap significance, an independent source with
positive results and comparable activity, and a Calmar ratio no worse than buy and hold.

## Source and runtime parity

Primary, independent validation, replay, and continuous live paper all use the same UTC
Sunday-22:00 through Friday-22:00 XAU session. Maximum holding time is enforced by the same
deterministic simulator semantics used by Generation 5. Generation 6 has isolated research and
paper command workers so adding it does not alter Generation 5's sealed source hash.

No candidate is automatically approved. A shortlist must finish one-time final validation, pass
every gate, and be explicitly approved on the Strategies page. The runtime still creates only
simulated orders and fills.

## Operator commands

```bash
uv run gold-trader research generation6 create
uv run gold-trader research generation6 seal
uv run gold-trader research generation6 start
uv run gold-trader research generation6 status
uv run gold-trader research generation6 pause
uv run gold-trader research generation6 stop
```

The default search contains 384 deterministic trials for each of four families, or 1,536 trials.
It checkpoints every batch and can stop early only after broad four-family coverage, at least 24
batches, and 12 consecutive batches without improvement when no shortlist exists. A futile result
is evidence that the fixed hypothesis set failed; it is not permission to relax safety gates.
