"""
PatchContext – application entry point.

Usage:
    # Launch the Streamlit UI
    streamlit run patchcontext/app.py

    # Or run directly (delegates to Streamlit)
    python patchcontext/app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    ui_path = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_path)]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
