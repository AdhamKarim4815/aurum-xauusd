# Future MT5 integration — blocked design note

This is not an implementation authorization. V1 must contain no MetaTrader Python package, `order_send`, Expert Advisor bridge, cloud execution credential, or live venue registration.

## Preconditions

- Approved candidate completes at least 30 trading days and 50 closed paper trades.
- Base versus observed spread, slippage, swap, downtime, and drawdown are reviewed.
- Broker, account jurisdiction, XAUUSD symbol/contract specification, margin, trading sessions, fill policy, and demo environment are selected.
- A separate legal/compliance and live-risk PRD is approved.

## Connectivity decision to make later

The official MetaTrader Python package communicates with a terminal process, while MetaTrader’s macOS package runs through Wine. Later work must compare:

1. Windows VPS/PC with official terminal-coupled Python.
2. Mac MetaTrader/Wine with a deliberately supported bridge.
3. A paid cross-platform MetaTrader cloud API.

The decision must be based on demo measurements, failure recovery, credential custody, latency, reconciliation, and operational support—not developer convenience.

References: [MetaTrader macOS installation](https://www.metatrader5.com/en/terminal/help/start_advanced/install_mac), [official Python integration](https://www.mql5.com/en/docs/python_metatrader5).

## Required future controls

Read-only shadow mode, demo-only order allowlist, broker-side stop confirmation, order-check before send, client/broker idempotency IDs, position/order/deal reconciliation, maximum notional and margin controls, session/rollover awareness, independent kill path, credential rotation, clock synchronization, durable telemetry, and staged canary rollout. Live approval must be distinct from research and paper approval.

