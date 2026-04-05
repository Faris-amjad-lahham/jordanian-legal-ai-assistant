"""
embeddings.py
-------------
OpenAI text-embedding-3-small with persistent disk caching.

Embeddings are computed once per language set, saved to:
  data/embeddings_{lang}.npy       — numpy float32 matrix
  data/embeddings_{lang}_ids.json  — ordered list of article IDs

On subsequent runs, cached embeddings are loaded from disk (~instant).
Cost: ~$0.002 total for the full Jordanian Labour Law dataset.
"""

import os
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────
# OpenAI client (optional import)
# ─────────────────────────────────────────────
try:
    from openai import OpenAI
    _client = OpenAI()
    _OPENAI_AVAILABLE = True
except Exception:
    _client = None
    _OPENAI_AVAILABLE = False

EMBEDDING_MODEL = "text-embedding-3-small"
DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 100          # OpenAI embeddings batch limit
MAX_TEXT_CHARS = 8000     # Truncate very long articles


# ─────────────────────────────────────────────
# Cache path helpers
# ─────────────────────────────────────────────
def _cache_paths(lang: str) -> Tuple[Path, Path]:
    return (
        DATA_DIR / f"embeddings_{lang}.npy",
        DATA_DIR / f"embeddings_{lang}_ids.json",
    )


def _ids_match(articles: List[Dict[str, Any]], cached_ids: List[str]) -> bool:
    if len(cached_ids) != len(articles):
        return False
    for a, cid in zip(articles, cached_ids):
        if str(a["article_id"]) != str(cid):
            return False
    return True


# ─────────────────────────────────────────────
# Embedding computation
# ─────────────────────────────────────────────
def _call_openai_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    """Batch-call OpenAI embeddings API. Returns float32 ndarray or None."""
    if not _OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = [t[:MAX_TEXT_CHARS] for t in texts[i: i + BATCH_SIZE]]
            resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)  # type: ignore
            all_vecs.extend(d.embedding for d in resp.data)
        return np.array(all_vecs, dtype=np.float32)
    except Exception as exc:
        print(f"[embeddings] API error: {exc}")
        return None


def compute_and_cache_embeddings(
    articles: List[Dict[str, Any]],
    lang: str,
) -> Optional[np.ndarray]:
    """
    Compute (or load from cache) embeddings for a list of normalized articles.

    Returns a float32 numpy array of shape (n, embedding_dim), or None if
    the OpenAI API is unavailable.
    """
    if not articles:
        return None

    npy_path, ids_path = _cache_paths(lang)

    # ── Try loading from disk cache ──────────────────────────────────────────
    if npy_path.exists() and ids_path.exists():
        try:
            cached_ids: List[str] = json.loads(ids_path.read_text(encoding="utf-8"))
            if _ids_match(articles, cached_ids):
                matrix = np.load(str(npy_path))
                if matrix.shape[0] == len(articles):
                    return matrix
        except Exception:
            pass  # corrupted cache → recompute

    # ── Compute from API ─────────────────────────────────────────────────────
    texts = [a["text"] for a in articles]
    matrix = _call_openai_embeddings(texts)
    if matrix is None:
        return None

    # ── Persist to disk ──────────────────────────────────────────────────────
    try:
        DATA_DIR.mkdir(exist_ok=True)
        np.save(str(npy_path), matrix)
        ids_path.write_text(
            json.dumps([str(a["article_id"]) for a in articles]),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[embeddings] Cache write warning: {exc}")

    return matrix


# ─────────────────────────────────────────────
# Query embedding (single call, not cached)
# ─────────────────────────────────────────────
def embed_query(query: str) -> Optional[np.ndarray]:
    """Embed a single query string. Returns 1-D float32 array or None."""
    if not _OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        resp = _client.embeddings.create(  # type: ignore
            model=EMBEDDING_MODEL,
            input=[query[:MAX_TEXT_CHARS]],
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as exc:
        print(f"[embeddings] Query embed error: {exc}")
        return None


# ─────────────────────────────────────────────
# Cosine similarity
# ─────────────────────────────────────────────
def cosine_similarity_scores(
    query_vec: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between:
      query_vec : shape (d,)
      matrix    : shape (n, d)

    Returns shape (n,) array of scores in [-1, 1].
    """
    if matrix.shape[0] == 0:
        return np.array([], dtype=np.float32)

    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    q_unit = query_vec / q_norm
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    m_norms = np.where(m_norms == 0, 1.0, m_norms)
    m_unit = matrix / m_norms
    return (m_unit @ q_unit).astype(np.float32)
