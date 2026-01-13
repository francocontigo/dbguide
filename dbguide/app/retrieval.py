"""
Retrieval utilities for DBGuide: reading markdown docs, building vector and BM25 indexes, and hybrid search.
All functions are type-annotated and documented.

DEPRECATED: This module is kept for backward compatibility.
Please use dbguide.services modules for new code.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional

import chromadb
from rank_bm25 import BM25Okapi

# Import from compatibility layer
from dbguide.app.retrieval_compat import (
    Doc,
    read_markdown_docs,
    tokenize,
    build_vector_index,
    build_bm25,
    save_bm25,
    load_bm25,
    get_all_metadata_keys_and_values,
    hybrid_search
)

__all__ = [
    'Doc',
    'read_markdown_docs',
    'tokenize',
    'build_vector_index',
    'build_bm25',
    'save_bm25',
    'load_bm25',
    'get_all_metadata_keys_and_values',
    'hybrid_search'
]
