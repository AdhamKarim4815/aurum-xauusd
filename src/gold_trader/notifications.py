from __future__ import annotations

import shutil
import subprocess
import sys


class MacNotifier:
    def notify(self, title: str, message: str) -> bool:
        executable = shutil.which("osascript")
        if sys.platform != "darwin" or executable is None:
            return False
        safe_title = title.replace('"', "'")
        safe_message = message.replace('"', "'")
        script = f'display notification "{safe_message}" with title "{safe_title}"'
        subprocess.run(
            [executable, "-e", script],
            check=False,
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
