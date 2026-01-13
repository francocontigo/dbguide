#!/usr/bin/env python3
"""
DBGuide CLI - Entry point for running the application.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main():
    """Run the DBGuide Streamlit application."""
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # Import and run Streamlit
    from streamlit.web import cli as stcli

    app_path = project_root / "dbguide" / "app" / "streamlit_app_refactored.py"

    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
