from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers import SentenceTransformer


def _fingerprint(paths: Iterable[Path], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype(np.float32)


def maybe_load_cache(cache_dir: Path, cache_key: str, matrix_name: str) -> np.ndarray | None:
    meta_path = cache_dir / "cache_meta.json"
    matrix_path = cache_dir / f"{matrix_name}.npy"
    if not meta_path.exists() or not matrix_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("cache_key") != cache_key:
        return None
    return np.load(matrix_path)


def save_cache(cache_dir: Path, cache_key: str, matrix_name: str, matrix: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / f"{matrix_name}.npy"
    meta_path = cache_dir / "cache_meta.json"
    np.save(matrix_path, matrix)

    meta = {"cache_key": cache_key}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def get_or_create_embeddings(
    cache_dir: Path,
    matrix_name: str,
    texts: list[str],
    source_files: list[Path],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    cache_key = _fingerprint(source_files, model_name) + f":{matrix_name}:{len(texts)}"
    cached = maybe_load_cache(cache_dir, cache_key, matrix_name)
    if cached is not None and len(cached) == len(texts):
        return cached

    model = load_model(model_name)
    matrix = encode_texts(model, texts)
    save_cache(cache_dir, cache_key, matrix_name, matrix)
    return matrix
