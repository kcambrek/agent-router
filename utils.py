import numpy as np

from models import embedder


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / max(norm, 1e-12)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed(query: str) -> np.ndarray:
    vector = np.asarray(embedder.embed_query(query), dtype=np.float64)
    return l2_normalize(vector)


def embed_documents(texts: list[str]) -> np.ndarray:
    matrix = np.asarray(embedder.embed_documents(texts), dtype=np.float64)
    return l2_normalize(matrix)
