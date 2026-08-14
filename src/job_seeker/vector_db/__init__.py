"""Vector database package (Qdrant).

Deliberately empty: re-exporting public functions here would eagerly import
``fastembed`` (via ``embeddings``) at package import time, slowing CLI startup.
Import from submodules explicitly instead, e.g.
``from job_seeker.vector_db.search import search_jobs``.
"""
