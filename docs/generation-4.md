# Generation 4 adaptive research

## Purpose

Generation 4 is a new immutable lineage derived from completed Generation 3 discovery evidence. It
is not a continuation that can rewrite Generation 3. The learning report records the exact parent
manifest, parent shortlist distribution, pre-holdout buy-and-hold benchmark, learned parameter
spaces, and its own hash. The final 12-month holdout remains unopened during learning and search.

## Search policy

Only Generation 3 families that produced robust shortlists are included. For the completed
`generation-3-r4` run these are compression breakout and time-series momentum. Generation 4:

- narrows each original parameter to values represented in the robust half of its parent shortlist;
- tests deterministic long-only and both-direction variants;
- tests no regime filter plus 100- and 200-period EMA regime filters;
- runs a recent-period screen before a complete pre-holdout discovery simulation;
- applies base and doubled transaction costs, chronological fold stability, profit concentration,
  selection-bias diagnostics, a 2% minimum CAGR, and a Calmar comparison with unlevered XAUUSD buy
  and hold; and
- checkpoints every batch through a single SQLite writer and resumes by exact trial hash.

The buy-and-hold comparison is an opportunity-cost diagnostic based on the XAUUSD price series. It
does not model broker financing and does not turn the bot into a passive investment product.

## Dashboard workflow

The **Strategies** page lists completed Generation 3 and current Generation 4 shortlists together,
highest discovery return first. A shortlist is not eligible for paper trading.

1. Wait until the generation is complete.
2. Import a second, independently sourced broker CSV or Parquet file into `var/data`.
3. Press **Run final validation** on one carefully reviewed shortlist. This opens that lineage's
   untouched holdout once and checks the same period against the independent source.
4. Keep the **Strategies** page open to see queued/running progress, or leave and return later; the
   job state is stored in SQLite and resumes after a dashboard restart. Completed and failed results
   remain on that page. Complete evidence is also written to
   `var/reports/<candidate-hash>.json`.
5. If every final gate passes, press **Approve strategy**. No hash typing is required.
6. Start replay paper first; then use continuous live paper only after inspecting replay evidence.

The approval button remains CSRF-protected and the service rechecks exact eligibility. Changing the
dataset, strategy, configuration, model, or implementation creates a different hash and prevents the
old approval from authorizing it.

## Commands

```bash
gold-trader research generation4 create
gold-trader research generation4 seal
gold-trader research generation4 start
gold-trader research generation4 status
gold-trader research generation4 pause
gold-trader research generation4 stop
gold-trader research generation4 finalize --trial-hash HASH --external-dataset FILE
```

Generation 4 remains simulation research. It cannot register or call a live execution venue, and
profitability is neither assumed nor promised.
