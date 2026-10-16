# ADR-0001: Mac-native core and deferred MT5

**Status:** Accepted

Use native Python 3.12 for research, replay, dashboard, and paper state. Define an execution port but register simulation only. Defer MT5 until the paper gate because the official Python API is terminal-coupled and the macOS terminal uses Wine. This keeps v1 reproducible on the operator’s Mac and prevents accidental real orders.

