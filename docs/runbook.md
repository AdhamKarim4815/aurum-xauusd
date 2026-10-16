# Operations and recovery runbook

## Start of session

1. Run `gold-trader doctor`; do not continue past a failed safety check.
2. Start `gold-trader serve` and open the loopback dashboard.
3. Confirm exact approval, paper state, kill state, feed entitlement/freshness, and data/source health.
4. Start replay paper first after every material change. Live/delayed paper remains blocked until its provider passes freshness and reconciliation.
5. Confirm the dashboard shows the sleep lease while active.

## Normal stop

Use dashboard or `gold-trader paper stop`. Wait for `stopped`, `entries_allowed=false`, pending simulated orders resolved/cancelled, and sleep lease released. Review the last audit events.

In continuous live paper, normal stop closes an open simulated position at the last known paper mark
and records `operator-stop-last-known-quote`. This is not a claim that a broker fill was available.

For the four-strategy lab, use **Stop all safely** on **Live lab**. Confirm all four account cards
show `stopped`, open positions are zero, entries are blocked, and the shared sleep lease is released.
Do not stop only one account: the exact four-strategy observation set is the unit of operation.

## Kill switch

Use kill when quotes, positions, process state, or model lineage is uncertain. Kill blocks entries immediately, records the reason, cancels simulated pending orders, releases `caffeinate`, and notifies the operator. Investigate before `paper reset-kill`; reset is not resume.

## Stale feed or disconnect

- Expected state: entries blocked and feed marked stale/unconfigured.
- Verify local network and provider entitlement without exposing the key.
- Backfill the missing interval, rerun quality, and compare the last pre-gap and first post-gap quote.
- Reconcile paper orders and positions. Never create trades that would have happened during the gap.
- Resume only from the next genuinely new quote.

## Worker crash or Mac restart

The process exit releases its own sleep lease. On restart, the live worker restores only when the exact approval, evidence, dataset, and Keychain key still exist. It backfills gaps without retroactive trades and retains the last decision-bar marker to prevent duplicates. A failed restore moves to `recovery-required`; inspect and stop/kill before a new start.

The live-lab worker restores all four candidate hashes together from `paper_lab` and
`paper_lab:account:<candidate-hash>` state. If any exact report, dataset hash, strategy hash, or
provider requirement fails, it enters `recovery-required` instead of continuing a partial set.

## Four-strategy lab start checklist

1. Stop the normal approved paper account and verify the global kill switch is reset.
2. Confirm at least one Keychain-backed provider returns a fresh XAUUSD quote.
3. Open **Live lab** and review the failed gates on all four candidates.
4. Select exactly four, accept the paper-only acknowledgement, and start.
5. Wait for four `running` account cards and four healthy feeds. `0 / 4` positions is normal until
   one of the 1h or 4h strategies completes a bar and emits a signal.
6. Review each return independently. Do not add the four percentages or interpret them as one
   portfolio.

## Generation 3 through 6 operations

- Keep the Mac connected to AC power and leave at least the configured free disk reserve.
- A network outage pauses the four-strategy lab automatically. Leave `gold-trader serve` running;
  the dashboard shows `paused-network`, the next retry time, and returns to `running` after a
  successful quote and gap reconciliation. Do not press Start again while it is paused.
- `research generation pause` and `stop` act after the current batch; completed trials remain durable.
- Resume only with the same sealed manifest. Changing data, settings, family space, or implementation
  requires a new manifest/hash and invalidates comparison with the old lineage.
- Do not finalize every top result. Review discovery concentration, PBO, deflated-Sharpe, and bootstrap
  evidence first; opening the holdout is intentionally irreversible for that lineage.
- A shortlist or positive holdout is not approval. Strict finalization requires the independent dataset,
  then pressing **Approve strategy** for the eligible immutable candidate on the Strategies page.
- Final-validation progress and its durable result are shown on **Strategies**. Queued work survives a
  page close or dashboard restart; the full evidence report is `var/reports/<candidate-hash>.json`.
- Generation 4 must name a completed Generation 3 parent. Preserve its learning report beside the
  sealed manifest; changing either requires a new Generation 4 lineage.
- Generation 5 must name a completed Generation 4 parent. Its learning report explicitly discloses
  adaptive period reuse. Do not describe a Generation 5 result as a globally untouched test.
- Generation 5 normalizes every source to the same UTC Sunday-22 through Friday-22 session and
  enforces the strategy's hashed maximum holding time in research, replay, and continuous paper.
- Generation 6 may name a terminal `completed`, `futile`, or operator-`stopped` Generation 5 parent. Its new hypotheses,
  activity policy, and learning report are sealed before execution. A futile outcome must remain
  recorded; do not turn it into a passing strategy by weakening risk or validation gates.
- Generation 6 uses isolated `generation6.` and `paper6.` queues. In live paper, the stored mode is
  `live-g6`; this prevents the older paper worker from restoring the same session concurrently.

## Database recovery

Stop the worker before copying the SQLite database and its WAL/SHM files. Preserve evidence reports and dataset manifests. Test restoration into a separate path. Never edit approval or event rows to recover; invalidate and create a new approval after verification.

## Source/parser failure

Disable the affected source, retain the last raw snapshot and error, run parser fixtures, review terms/robots changes, and create a new parser version. A reparsed dataset receives a new content hash and invalidates dependent lineages.

## Loss halt

Do not reset a daily/five-day/peak halt to continue trading. Stop, review the trade ledger, assumptions, fills, and model context. A peak account lock requires a new research and approval decision.
