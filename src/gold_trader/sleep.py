from __future__ import annotations

import shutil
import subprocess
import sys


class SleepLease:
    """Owns a macOS sleep-prevention process for exactly one paper session."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def active(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def acquire(self) -> bool:
        if self.active:
            return True
        executable = shutil.which("caffeinate")
        if sys.platform != "darwin" or executable is None:
            return False
        self._process = subprocess.Popen(
            [executable, "-dimsu"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self.active

    def release(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
