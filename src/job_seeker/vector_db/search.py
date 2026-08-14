"""Hybrid retrieval: ColBERT multi-vector (MaxSim) + BM25 lexical search fused with RRF."""

from qdrant_client import models

from job_seeker.config import settings
from job_seeker.vector_db.embeddings import embed_colbert_query, embed_sparse
from job_seeker.vector_db.qdrant import get_client


def _colbert_query_vector(query: str) -> list[list[float]]:
    """Embed a query into a ColBERT token matrix (multi-query vector)."""
    return embed_colbert_query(query)


def _sparse_vector(query: str) -> models.SparseVector:
    sparse = next(iter(embed_sparse([query])))
    return models.SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )


def hybrid_search(
    query: str,
    top_k: int | None = None,
    prefetch_limit: int | None = None,
    collection: str | None = None,
    filter_: models.Filter | None = None,
    dense_query: str | None = None,
    sparse_query: str | None = None,
) -> list[models.ScoredPoint]:
    """Run ColBERT + BM25 prefetch queries and fuse the results with RRF.

    The ColBERT branch uses Qdrant's MAX_SIM comparator over token-level
    multi-vectors (late interaction). ``dense_query``/``sparse_query`` optionally
    override the text embedded for each branch (e.g. an LLM-expanded keyword
    query for the sparse BM25 branch).

    Returns raw scored points (one per matched chunk)."""
    client = get_client()
    prefetch_limit = prefetch_limit or settings.prefetch_limit
    prefetch = [
        models.Prefetch(
            query=_colbert_query_vector(dense_query or query),
            using=settings.qdrant_dense_vector_name,
            limit=prefetch_limit,
            filter=filter_,
        ),
        models.Prefetch(
            query=_sparse_vector(sparse_query or query),
            using=settings.qdrant_sparse_vector_name,
            limit=prefetch_limit,
            filter=filter_,
        ),
    ]
    response = client.query_points(
        collection_name=collection or settings.qdrant_collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k or settings.search_limit,
        with_payload=True,
    )
    return response.points


def _point_to_job(point: models.ScoredPoint) -> dict:
    payload = point.payload or {}
    return {
        "score": point.score,
        "job_id": payload.get("job_id"),
        "job_url": payload.get("job_url"),
        "company": payload.get("company"),
        "salary": payload.get("salary"),
        "working_location": payload.get("working_location"),
        "matched_section": payload.get("section"),
        "matched_text": payload.get("text"),
    }


def search_jobs(
    query: str,
    top_k: int | None = None,
    prefetch_limit: int | None = None,
    collection: str | None = None,
    filter_: models.Filter | None = None,
    dense_query: str | None = None,
    sparse_query: str | None = None,
) -> list[dict]:
    """Hybrid RRF search deduplicated to one result per job.

    Chunks of the same job can each score via RRF, so we fetch extra candidates,
    keep the highest-scoring chunk per job, and return ``top_k`` distinct jobs.
    """
    target = top_k or settings.search_limit
    points = hybrid_search(
        query,
        top_k=max(target * 3, prefetch_limit or settings.prefetch_limit),
        prefetch_limit=prefetch_limit,
        collection=collection,
        filter_=filter_,
        dense_query=dense_query,
        sparse_query=sparse_query,
    )
    results: list[dict] = []
    seen: set[str] = set()
    for point in points:
        job = _point_to_job(point)
        job_id = job["job_id"]
        if job_id in seen or not job_id:
            continue
        seen.add(job_id)
        results.append(job)
        if len(results) >= target:
            break
    return results


def search_jobs_expanded(
    query: str,
    top_k: int | None = None,
    expand: bool = True,
    collection: str | None = None,
    filter_: models.Filter | None = None,
    model: str | None = None,
) -> tuple[list[dict], str]:
    """Hybrid search where the BM25 branch uses an LLM-expanded keyword query.

    Returns ``(results, expanded_query)``. ``query`` stays as the dense (semantic)
    query; the expanded text drives the sparse BM25 branch for better lexical recall.
    """
    expanded = query
    if expand:
        from job_seeker.query_expansion import expand_query

        expanded = expand_query(query, model=model)
    results = search_jobs(
        query,
        top_k=top_k,
        collection=collection,
        filter_=filter_,
        sparse_query=expanded,
    )
    return results, expanded
