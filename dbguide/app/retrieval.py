# app/retrieval.py
from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import chromadb
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class Doc:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def read_markdown_docs(corpus_dir: str = "corpus") -> List[Doc]:
    paths = glob.glob(os.path.join(corpus_dir, "**/*.md"), recursive=True)
    docs: List[Doc] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        meta: Dict[str, Any] = {}
        body = raw

        # Parse frontmatter YAML simples (--- \n ... \n ---)
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
    # tokenização simples (ok pro MVP)
    text = text.replace("```", " ").replace("\n", " ")
    parts = [p.strip().lower() for p in text.split()]
    return [p for p in parts if len(p) >= 3]


def build_vector_index(
    docs: List[Doc],
    chroma_dir: str = "data/chroma",
    collection_name: str = "sql_cards",
) -> chromadb.api.models.Collection.Collection:
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
    tokenized = [tokenize(d.text) for d in docs]
    return BM25Okapi(tokenized)


def save_bm25(bm25: BM25Okapi, docs: List[Doc], path: str = "data/bm25.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)


def load_bm25(path: str = "data/bm25.pkl") -> Tuple[BM25Okapi, List[Doc]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["docs"]


def hybrid_search(
    query: str,
    col,
    bm25: BM25Okapi,
    docs: List[Doc],
    top_k: int = 6,
    alpha: float = 0.55,
) -> List[Dict]:
    """
    alpha = peso do vetor (0..1). (1-alpha) = peso do bm25.
    """
    # 1) Vetorial (Chroma)
    v = col.query(query_texts=[query], n_results=top_k)
    vec_ids = v["ids"][0]
    vec_dist = v["distances"][0]  # distância (menor = melhor)

    # transforma distância em score (maior = melhor)
    vec_score = {i: 1.0 / (1e-6 + float(d)) for i, d in zip(vec_ids, vec_dist)}

    # 2) BM25
    qtok = tokenize(query)
    bm25_scores = bm25.get_scores(qtok)
    bm_score = {docs[i].id: float(bm25_scores[i]) for i in range(len(docs))}

    # 3) mistura
    all_ids = set(vec_score.keys()) | set(bm_score.keys())
    merged = []
    for _id in all_ids:
        score = alpha * vec_score.get(_id, 0.0) + (1 - alpha) * bm_score.get(_id, 0.0)
        merged.append((_id, score))
    merged.sort(key=lambda x: x[1], reverse=True)

    doc_map = {d.id: d.text for d in docs}
    out = []
    for _id, sc in merged[:top_k]:
        out.append({"id": _id, "score": sc, "text": doc_map.get(_id, "")})
    return out
