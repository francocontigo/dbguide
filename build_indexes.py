#!/usr/bin/env python3
"""
Build indexes for DBGuide - CLI entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main():
    """Build vector and BM25 indexes."""
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # Import and run the build script
    from dbguide.ingest.build_index_refactored import main as build_main

    build_main()


if __name__ == "__main__":
    main()
