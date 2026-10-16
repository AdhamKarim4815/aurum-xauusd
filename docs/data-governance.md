# Data and source governance

## Source order

1. Dukascopy historical export for free cached bid/ask history.
2. User-provided broker exports for broker-specific validation.
3. Twelve Data XAU/USD for optional live/delayed observations with a Keychain-stored key.
4. FRED/ALFRED, CFTC, and official release calendars for macro and positioning.
5. Structured news APIs/RSS, then explicitly allowlisted HTML scrapers.

An adapter is replaceable; source identity and semantics remain in every manifest.

## Manifest requirements

Dataset ID, original source and URL, retrieval timestamp, raw/normalized content hashes, parser version, symbol mapping, original timezone, UTC policy, row count, coverage, terms/licence URL, expected interval, entitlement/latency, and quality results.

## Quality rules

- Required columns: UTC timestamp and OHLC; bid/ask and volume are retained when present.
- Reject invalid or missing timestamps/prices, inconsistent OHLC, nonmonotonic results, and crossed quotes.
- Deduplicate deterministically, keeping the last identical timestamp and recording the count.
- Report gaps over six expected intervals; distinguish market closures from unexplained gaps.
- Detect source jumps/outliers and compare overlapping providers before eligibility.
- Never backward fill price, macro, or headline values.

## Point-in-time joins

Store `event_time`, `published_at`, `available_at`, and `retrieved_at` separately. A feature at decision time may use only records whose `available_at` is not later. ALFRED vintages prevent revised macro history from appearing early. CFTC data must use Friday publication time, not the preceding Tuesday position date.

## Scraper registry

Each scraper has an owner, purpose, URL patterns, terms URL, robots decision, request rate, user agent, cache TTL, raw snapshot location, parser version, fixture tests, untrusted-content sanitation, and enable/disable switch. Stop collection on a terms/robots change, repeated parser failure, or ambiguous timestamps.

## Retention and deletion

Raw data is immutable and content-addressed. Derived datasets can be regenerated from a manifest. Credentials and session material never enter datasets. The operator may delete local data/artifacts after backing up manifests and reports; deletion does not rewrite old evidence claims.

