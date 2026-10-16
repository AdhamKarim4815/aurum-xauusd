# ADR-0002: SQLite, Parquet, and local artifacts

**Status:** Accepted

Use SQLite WAL for commands, audit, approvals, research indexes, and application state; Parquet for columnar time series; and local ignored directories for reports/models. This matches a personal single-operator Mac application, provides durable relational behavior without cloud infrastructure, and keeps large data out of Git.

