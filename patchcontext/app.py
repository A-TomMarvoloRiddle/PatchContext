"""
PatchContext – application entry point.

Usage:
    streamlit run patchcontext/app.py
    # or
    streamlit run patchcontext/ui/streamlit_app.py
"""

from __future__ import annotations

from patchcontext.ui.streamlit_app import main

if __name__ == "__main__":
    main()
