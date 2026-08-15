"""ColBERT (late-interaction) + BM25 sparse embeddings via FastEmbed.

Dense branch: ``colbert-ir/colbertv2.0`` produces one 128-d vector per token
(multi-vector). Qdrant stores these as dense multi-vectors and scores them with
MAX_SIM late interaction (see ``vector_db.qdrant.collection_params``). The
sparse branch stays FastEmbed BM25 for lexical matching.
"""

from functools import lru_cache

import numpy as np
from fastembed import (
    LateInteractionTextEmbedding,
    SparseEmbedding,
    SparseTextEmbedding,
)

from job_seeker.config import settings
from job_seeker.logging_setup import get_logger

logger = get_logger(__name__)

__all__ = [
    "get_colbert_embedder",
    "get_sparse_embedder",
    "embed_colbert_docs",
    "embed_colbert_query",
    "embed_sparse",
    "embedding_dim",
    "warm_up",
]


@lru_cache(maxsize=1)
def get_colbert_embedder() -> LateInteractionTextEmbedding:
    logger.info(
        "Loading ColBERT model %s (first load downloads weights)", settings.embedding_model
    )
    return LateInteractionTextEmbedding(model_name=settings.embedding_model)


@lru_cache(maxsize=1)
def get_sparse_embedder() -> SparseTextEmbedding:
    logger.info(
        "Loading sparse (BM25) model %s (first load downloads weights)",
        settings.sparse_embedding_model,
    )
    return SparseTextEmbedding(model_name=settings.sparse_embedding_model)


def embed_colbert_docs(texts: list[str]) -> list[list[list[float]]]:
    """Embed documents into ColBERT token matrices.

    Returns one entry per text: a ``(num_tokens, dim)`` matrix of 128-d token
    vectors (a dense multi-vector) ready to store in Qdrant.

    A small ``batch_size`` is used deliberately: ColBERT's attention MatMul is
    O(batch x heads x seq^2), so the fastembed default of 256 would request a
    ~3GB tensor for a full batch (see settings.colbert_batch_size).
    """
    return [
        [vec.tolist() for vec in token_matrix]
        for token_matrix in get_colbert_embedder().embed(
            list(texts), batch_size=settings.colbert_batch_size
        )
    ]


def embed_colbert_query(query: str) -> list[list[float]]:
    """Embed a search query into a ColBERT token matrix (padded to 32 tokens)."""
    token_matrix = next(iter(get_colbert_embedder().query_embed([query])))
    return [vec.tolist() for vec in token_matrix]


def embed_sparse(texts: list[str] | str) -> list[SparseEmbedding]:
    return list(get_sparse_embedder().embed(texts))


def embedding_dim() -> int:
    """Return the ColBERT token-vector dimension for the configured model."""
    return int(get_colbert_embedder().embedding_size)


def warm_up() -> None:
    """Pre-load both embedders (first call downloads the models)."""
    logger.info("Warming up embedding models…")
    embed_colbert_docs(["warmup"])
    embed_sparse(["warmup"])
    logger.info("Embedding models ready")
