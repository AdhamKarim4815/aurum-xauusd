# Requirements traceability

| Requirement | Primary implementation | Verification |
| --- | --- | --- |
| PRD-001 Local boundary | `web.LocalOnlyMiddleware`, CLI loopback check, Keychain module | HTTP and doctor tests |
| PRD-002 No live orders | `ExecutionVenue` port plus simulation-only runtime | Static import/route test |
| PRD-003 Reproducible data | normalization, SHA-256 manifests, Parquet import | data quality tests |
| PRD-004 Research evidence | `ResearchRunner`, JSON report, SQLite run record | deterministic research tests |
| PRD-005 Manual promotion | `ApplicationService.approve_candidate`, eligible-only dashboard button | CSRF and one-click approval tests |
| PRD-006 Invalidation | bound approval hashes and invalidation repository method | service/database tests |
| PRD-007 Deterministic risk | `RiskEngine` | unit and property tests |
| PRD-008 Safe worker | SQLite command queue, `PaperWorker`, fail-closed state | idempotency/worker tests |
| PRD-009 Emergency control | kill service, event, worker cancellation and lease release | service/worker tests |
| PRD-010 Evidence dashboard | local templates and read APIs, no trade form | HTTP/UI tests |
| PRD-011 Bounded Generation-2 rules | fixed registry tuple and strategy-hash parameters | registry/hash tests and complete comparison report |
| PRD-012 Leakage-safe ML filter | signal-only meta labels, chronological calibration, purged folds, fixed threshold | horizon, stop-first, chronology, suppression-only tests |
| PRD-013 Exact ML replay | checksum-validated local artifact plus learned-parameter model hash | worker artifact and tamper-rejection tests |
| PRD-014 Sealed Generation 3 | immutable manifest hash, deterministic trial queue, single-writer checkpoints | manifest tamper, deterministic hash, checkpoint tests |
| PRD-015 Isolated local coordinator | research command prefix, process pool, AC/disk/load and futility gates | command routing and coordinator tests |
| PRD-016 Untouched Generation 3 finalization | explicit finalize operation, holdout boundary, external-source gate | finalize gate and manifest tests |
| PRD-017 Continuous live paper | quote polling, five-minute aggregation, next-quote simulated execution | live paper lifecycle test |
| PRD-018 Live fail-closed recovery | approval/freshness/network checks and no-retroactive reconciliation | stale quote and worker recovery tests |
| PRD-019 Durable paper risk ledger | intents, risk decisions, simulated orders, fills, positions, account and close events | event-ledger and risk invariant tests |
| PRD-020 Adaptive Generation 4 | parent learning report, sealed G4 manifest, benchmark gates, isolated command worker | learning hash, deterministic trial, tamper, coordinator tests |
| PRD-021 Strategy choice library | combined G3-G6 shortlist view, independent-data finalization, eligible-only approval button | ranking and approval HTTP tests |
| PRD-022 Durable validation progress | isolated validation queue/worker, SQLite state, result report endpoint | validation queue, retry, API, and worker tests |
| PRD-023 Frequency-first Generation 5 | common XAU session, bounded fast-rule space, time exits, per-year/month activity gates | session, simulator, manifest, coordinator, finalization, and paper-parity tests |
| PRD-024 Higher-activity Generation 6 | new state/recovery hypotheses, sealed adaptive lineage, activity floors, isolated research/paper workers | strategy, manifest, coordinator, command-routing, finalization, and dashboard tests |
| PRD-025 Experimental four-strategy lab | `PaperLabWorker`, namespaced account state, shared-cycle quote provider, activity-only override validator, dashboard/API | shared quote, override policy, exact-four queue, isolated-account worker, CSRF/dashboard tests |

Any new requirement must be added here with an implementation owner and executable verification before it is considered complete.
