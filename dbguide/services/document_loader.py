"""
Document indexing and loading utilities for DBGuide.
Handles reading markdown documents and parsing metadata.
"""
from __future__ import annotations

import glob
import os
from typing import List

import yaml

from dbguide.models.document import Document


class DocumentLoader:
    """Loads documents from markdown files with YAML frontmatter."""

    def __init__(self, corpus_dir: str = "corpus"):
        """
        Initialize document loader.

        Args:
            corpus_dir: Directory containing markdown files.
        """
        self.corpus_dir = corpus_dir

    def load_documents(self) -> List[Document]:
        """
        Load all markdown files from the corpus directory.

        Returns:
            List of Document objects with parsed metadata.
        """
        paths = glob.glob(
            os.path.join(self.corpus_dir, "**/*.md"),
            recursive=True
        )

        documents: List[Document] = []

        for path in paths:
            doc = self._load_single_document(path)
            if doc:
                documents.append(doc)

        return documents

    def _load_single_document(self, path: str) -> Document | None:
        """
        Load a single markdown document.

        Args:
            path: Path to the markdown file.

        Returns:
            Document object or None if loading fails.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            metadata, body = self._parse_frontmatter(raw)

            return Document(
                id=path,
                text=body.strip(),
                metadata=metadata
            )
        except Exception as e:
            print(f"Warning: Failed to load document {path}: {e}")
            return None

    def _parse_frontmatter(self, raw: str) -> tuple[dict, str]:
        """
        Parse YAML frontmatter from markdown content.

        Args:
            raw: Raw markdown content.

        Returns:
            Tuple of (metadata dict, body text).
        """
        metadata = {}
        body = raw

        # Parse simple YAML frontmatter (--- \n ... \n ---)
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                _, meta_str, body = parts
                try:
                    loaded = yaml.safe_load(meta_str) or {}
                    if isinstance(loaded, dict):
                        metadata = loaded
                except Exception:
                    metadata = {}

        return metadata, body


def get_all_metadata_keys_and_values(documents: List[Document]) -> dict[str, list]:
    """
    Extract all unique metadata keys and their possible values.

    Args:
        documents: List of documents to analyze.

    Returns:
        Dictionary mapping metadata keys to lists of unique values.
    """
    meta_dict: dict[str, set] = {}

    for doc in documents:
        for key, value in doc.metadata.items():
            if key not in meta_dict:
                meta_dict[key] = set()
            meta_dict[key].add(value)

    # Convert sets to sorted lists
    return {k: sorted(list(vs)) for k, vs in meta_dict.items()}
