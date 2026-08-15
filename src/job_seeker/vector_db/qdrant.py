"""Qdrant collection management for hybrid (ColBERT multi-vector + BM25) retrieval.

Runs either against a Qdrant server (QDRANT_URL) or fully embedded in-process
(QDRANT_PATH) with the collection persisted as files on disk.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Datatype,
    Distance,
    HnswConfigDiff,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    OptimizersConfigDiff,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from job_seeker.config import settings
from job_seeker.vector_db.embeddings import embedding_dim

PAYLOAD_INDEX_FIELDS = ("job_id", "company", "working_location", "section")

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return a shared client. Prefers embedded mode (QDRANT_PATH) over server mode."""
    global _client
    if _client is None:
        if settings.qdrant_path:
            settings.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=str(settings.qdrant_path))
        else:
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_key or None,
            )
    return _client


def close_client() -> None:
    """Explicitly release the client so its destructor is a no-op at exit.

    In embedded mode the local backend lazily imports ``portalocker`` on close,
    which fails during interpreter shutdown (``sys.meta_path is None``). Closing
    while the import system is alive avoids that harmless ``__del__`` warning.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None


def is_embedded() -> bool:
    return settings.qdrant_path is not None


def _hnsw_config() -> HnswConfigDiff:
    return HnswConfigDiff(
        m=settings.qdrant_hnsw_m,
        ef_construct=settings.qdrant_hnsw_ef_construct,
        on_disk=True,
    )


def collection_params() -> dict:
    """Build kwargs for create/recreate_collection:
    ColBERT dense multi-vectors (MAX_SIM late interaction) + BM25 sparse vectors.
    """
    return {
        "vectors_config": {
            settings.qdrant_dense_vector_name: VectorParams(
                size=embedding_dim(),
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=_hnsw_config(),
            )
        },
        "sparse_vectors_config": {
            settings.qdrant_sparse_vector_name: SparseVectorParams(
                modifier=Modifier.IDF,
            )
        },
        "optimizers_config": OptimizersConfigDiff(memmap_threshold=20000),
    }


def _create_payload_indexes(client: QdrantClient, collection: str) -> None:
    if is_embedded():
        return
    for field in PAYLOAD_INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            continue


def ensure_collection(collection: str | None = None, recreate: bool = False) -> str:
    """Create the hybrid collection if missing (or recreate it), then index payload fields."""
    client = get_client()
    collection = collection or settings.qdrant_collection
    exists = client.collection_exists(collection)
    if exists and not recreate:
        return collection
    params = collection_params()
    if exists:
        client.recreate_collection(collection_name=collection, **params)
    else:
        client.create_collection(collection_name=collection, **params)
    _create_payload_indexes(client, collection)
    return collection


def delete_collection(collection: str | None = None) -> None:
    client = get_client()
    collection = collection or settings.qdrant_collection
    if client.collection_exists(collection):
        client.delete_collection(collection_name=collection)


def collection_info(collection: str | None = None) -> dict:
    """Return a diagnostic summary of the collection, including BM25/HNSW status."""
    client = get_client()
    collection = collection or settings.qdrant_collection
    if not client.collection_exists(collection):
        return {"collection": collection, "exists": False}
    info = client.get_collection(collection)
    params = info.config.params
    vectors = params.vectors or {}
    dense = vectors.get(settings.qdrant_dense_vector_name)
    return {
        "collection": collection,
        "exists": True,
        "status": info.status,
        "points_count": info.points_count,
        "bm25_sparse_index_active": bool(params.sparse_vectors),
        "sparse_vector_name": settings.qdrant_sparse_vector_name,
        "sparse_model": settings.sparse_embedding_model,
        "dense_vector_name": settings.qdrant_dense_vector_name,
        "dense_vector_size": dense.size if dense else None,
        "dense_distance": str(dense.distance) if dense else None,
        "dense_multi_vector": bool(getattr(dense, "multivector_config", None)),
        "hnsw": {
            "m": dense.hnsw_config.m if dense and dense.hnsw_config else None,
            "ef_construct": dense.hnsw_config.ef_construct if dense and dense.hnsw_config else None,
        },
        "embedded_mode": is_embedded(),
        "storage": str(settings.qdrant_path) if settings.qdrant_path else settings.qdrant_url,
    }
