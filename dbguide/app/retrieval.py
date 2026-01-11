
"""
Retrieval utilities for DBGuide: reading markdown docs, building vector and BM25 indexes, and hybrid search.
All functions are type-annotated and documented.
"""
from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import chromadb
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class Doc:
    """
    Represents a single markdown document/card with optional metadata.
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def read_markdown_docs(corpus_dir: str = "corpus") -> List[Doc]:
    """
    Reads all markdown files in a directory (recursively), parses YAML frontmatter as metadata.
    Returns a list of Doc objects.
    """
    paths = glob.glob(os.path.join(corpus_dir, "**/*.md"), recursive=True)
    docs: List[Doc] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        meta: Dict[str, Any] = {}
        body = raw

        # Parse simple YAML frontmatter (--- \n ... \n ---)
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                _, meta_str, body = parts
                try:
                    loaded = yaml.safe_load(meta_str) or {}
                    if isinstance(loaded, dict):
                        meta = loaded
                except Exception:
                    meta = {}

        docs.append(Doc(id=path, text=body.strip(), metadata=meta))
    return docs


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text for BM25: lowercases, splits on whitespace, removes short tokens.
    """
    text = text.replace("```", " ").replace("\n", " ")
    parts = [p.strip().lower() for p in text.split()]
    return [p for p in parts if len(p) >= 3]


def build_vector_index(
    docs: List[Doc],
    chroma_dir: str = "data/chroma",
    collection_name: str = "sql_cards",
) -> chromadb.api.models.Collection.Collection:
    """
    Builds a vector index using SentenceTransformer and stores in ChromaDB.
    Returns the ChromaDB collection.
    """
    os.makedirs(chroma_dir, exist_ok=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode([d.text for d in docs], normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.get_or_create_collection(name=collection_name)

    col.upsert(
        ids=[d.id for d in docs],
        documents=[d.text for d in docs],
        embeddings=embeddings,
        metadatas=[{**{"path": d.id}, **d.metadata} for d in docs],
    )
    return col


def build_bm25(docs: List[Doc]) -> BM25Okapi:
    """
    Builds a BM25 index from a list of Doc objects.
    """
    tokenized = [tokenize(d.text) for d in docs]
    return BM25Okapi(tokenized)


def save_bm25(bm25: BM25Okapi, docs: List[Doc], path: str = "data/bm25.pkl") -> None:
    """
    Saves the BM25 index and docs to a pickle file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)


def load_bm25(path: str = "data/bm25.pkl") -> Tuple[BM25Okapi, List[Doc]]:
    """
    Loads the BM25 index and docs from a pickle file.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["docs"]


def get_all_metadata_keys_and_values(docs: List[Doc]) -> Dict[str, List[Any]]:
    """
    Returns a dictionary with all unique metadata keys and their possible values across all docs.
    Useful for building filter UIs or agents.
    """
    meta_dict: Dict[str, set] = {}
    for d in docs:
        for k, v in d.metadata.items():
            if k not in meta_dict:
                meta_dict[k] = set()
            meta_dict[k].add(v)
    # Convert sets to sorted lists
    return {k: sorted(list(vs)) for k, vs in meta_dict.items()}


def hybrid_search(
    query: str,
    col: chromadb.api.models.Collection.Collection,
    bm25: BM25Okapi,
    docs: List[Doc],
    top_k: int = 6,
    alpha: float = 0.55,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a hybrid vector + BM25 search, optionally filtering by metadata.

    Args:
        query: The user query string.
        col: ChromaDB collection.
        bm25: BM25Okapi index.
        docs: List of Doc objects.
        top_k: Number of results to return.
        alpha: Weight for vector score (0..1). (1-alpha) is weight for BM25.
        metadata_filter: Dict of metadata to filter on (e.g. {"dialect": "mysql"}).

    Returns:
        List of dicts: {"id", "score", "text"}
    """
    # 1) Vector-based search (Chroma) with metadata filter
    chroma_kwargs = {"query_texts": [query], "n_results": top_k}
    if metadata_filter:
        print(f"[hybrid_search] Usando filtro de metadata: {metadata_filter}")
        chroma_kwargs["where"] = metadata_filter
    v = col.query(**chroma_kwargs)
    vec_ids = v["ids"][0]
    vec_dist = v["distances"][0]  # distance (smaller is better)

    # Convert distance to a score (larger is better)
    vec_score = {i: 1.0 / (1e-6 + float(d)) for i, d in zip(vec_ids, vec_dist)}

    # 2) BM25 keyword-based score, filter docs by metadata if provided
    qtok = tokenize(query)
    bm25_scores = bm25.get_scores(qtok)
    # Only consider BM25 scores for docs matching the filter
    if metadata_filter:
        filtered_docs = [d for d in docs if all(k in d.metadata and d.metadata[k] == v for k, v in metadata_filter.items())]
        print(f"[hybrid_search] Docs filtrados por metadata: {[d.id for d in filtered_docs]}")
        bm_score = {
            d.id: float(bm25_scores[i])
            for i, d in enumerate(docs)
            if all(k in d.metadata and d.metadata[k] == v for k, v in metadata_filter.items())
        }
    else:
        bm_score = {d.id: float(bm25_scores[i]) for i, d in enumerate(docs)}

    # 3) Combine scores from both methods
    all_ids = set(vec_score.keys()) | set(bm_score.keys())
    doc_map = {d.id: d.text for d in docs}
    merged = []
    for _id in all_ids:
        score = alpha * vec_score.get(_id, 0.0) + (1 - alpha) * bm_score.get(_id, 0.0)
        merged.append((_id, score))
    merged.sort(key=lambda x: x[1], reverse=True)

    print(f"[hybrid_search] Top resultados combinados:")
    for _id, sc in merged[:top_k]:
        print(f"  - {_id} | score={sc:.4f}")

    out = []
    for _id, sc in merged[:top_k]:
        out.append({"id": _id, "score": sc, "text": doc_map.get(_id, "")})
    return out
