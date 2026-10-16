"""Shared multilingual sentence-embedding model (needs the ``ml`` extras).

One model instance serves every guard. ``embed_cached`` keeps embeddings of fixed corpora
on disk, so ``unmimic serve`` does not re-embed thousands of prompts at every start.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=2)
def load_embedder(name: str = DEFAULT_MODEL) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: list[str], name: str = DEFAULT_MODEL) -> Any:
    """Unit-length embeddings, one row per text (a numpy array)."""
    import numpy as np

    model = load_embedder(name)
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return model.encode(texts, normalize_embeddings=True, batch_size=64)


def embed_cached(texts: list[str], cache_dir: Path, name: str = DEFAULT_MODEL) -> Any:
    import numpy as np

    digest = hashlib.sha256("\x00".join([name, *texts]).encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"embeddings-{digest}.npy"
    if path.exists():
        return np.load(path)
    vectors = embed(texts, name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors
