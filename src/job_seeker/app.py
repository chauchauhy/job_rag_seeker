"""Streamlit web demo: hybrid RAG job search (dense HNSW + BM25 + RRF)."""

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Work around a hanging Windows WMI query in platform.system() that freezes
# Streamlit's import (env_util.py) when the WMI service is unhealthy.
if sys.platform == "win32":
    platform._wmi = None

import streamlit as st

from job_seeker.config import settings
from job_seeker.discovery_ui import render_discovery
from job_seeker.logging_setup import setup_logging
from job_seeker.vector_db.qdrant import collection_info
from job_seeker.vector_db.search import search_jobs, search_jobs_expanded

setup_logging()

st.set_page_config(page_title="Job Seeker RAG", page_icon="", layout="wide")

st.title("Job Seeker - Hybrid RAG Search")
st.caption(
    "ColBERT late-interaction (MaxSim multi-vector) + BM25 sparse vectors fused "
    "with RRF. Optionally expand the query with the LLM (opencode) to boost BM25 "
    "keyword recall."
)


def _render_diagnostics() -> None:
    st.sidebar.header("Vector index")
    try:
        info = collection_info()
        if not info["exists"]:
            st.sidebar.error("Collection not found. Run: `job-seeker ingest --recreate`")
            return
        st.sidebar.json(
            {
                "collection": info["collection"],
                "status": info["status"],
                "points (chunks)": info["points_count"],
                "dense model": info["dense_vector_name"] + f" ({info['dense_vector_size']}d)",
                "dense distance": info["dense_distance"],
                "hnsw": info["hnsw"],
                "storage": info["storage"],
                "embedded mode": info["embedded_mode"],
            },
            expanded=False,
        )
        if info["bm25_sparse_index_active"]:
            st.sidebar.success("BM25 sparse index: ACTIVE")
        else:
            st.sidebar.warning("BM25 sparse index: NOT ACTIVE")
        st.sidebar.write(
            f"BM25 vector: `{info['sparse_vector_name']}`\n\n"
            f"BM25 model: `{info['sparse_model']}`"
        )
    except Exception as exc:
        st.sidebar.error(f"Qdrant unavailable: {exc}")


def _render_job(job: dict, rank: int) -> None:
    company = job["company"] or "Unknown company"
    score = job.get("score", 0.0)
    with st.container(border=True):
        col1, col2, col3 = st.columns([4, 1, 1])
        col1.markdown(f"**#{rank} · {company}**")
        col2.metric("RRF score", f"{score:.3f}")
        col3.markdown(
            f"[:arrow_upper_right: Apply]({job['job_url']})" if job.get("job_url") else ""
        )
        meta = " · ".join(
            str(v)
            for v in [job.get("working_location"), job.get("salary"), job.get("matched_section")]
            if v
        )
        st.caption(meta)
        with st.expander("Matched snippet"):
            st.write(job.get("matched_text", ""))


def _warm_up_embeddings() -> None:
    from job_seeker.vector_db.embeddings import warm_up

    warm_up()


def _safe_load_resume() -> dict | None:
    """Load the extracted resume JSON, returning ``None`` when it is missing."""
    try:
        from job_seeker.rag import load_resume

        return load_resume()
    except Exception:
        return None


def _render_search(top_k: int, expand_query: bool, rerank: bool) -> None:
    query = st.text_input(
        "Search query",
        placeholder="e.g. Java Spring Boot backend developer, or Android + machine learning",
    )

    if not query:
        st.info("Enter a query to search. Example: `machine learning model fine-tuning`")
        return

    if st.button("Search", type="primary"):
        try:
            info = collection_info()
            if not info.get("exists"):
                st.error(
                    "No Qdrant collection yet. Run `job-seeker ingest --recreate` "
                    "(or `job-seeker init-vector-db`) to index crawled jobs, then retry."
                )
                return
            if not info.get("points_count"):
                st.warning(
                    "The collection is empty. Run `job-seeker ingest` to index crawled jobs."
                )
                return
        except Exception as exc:
            st.error(f"Qdrant unavailable: {exc}")
            return

        try:
            if expand_query:
                with st.spinner("Running hybrid search…"):
                    results, expanded = search_jobs_expanded(query, top_k=top_k)
                if expanded != query:
                    st.success(f"Expanded BM25 query: `{expanded}`")
            else:
                with st.spinner("Running hybrid search…"):
                    results = search_jobs(query, top_k=top_k)
                expanded = query

            if rerank:
                from job_seeker.rag import load_resume, rerank_with_llm

                resume = load_resume()
                with st.spinner("Reranking with LLM..."):
                    ranked = rerank_with_llm(resume, results, top_n=top_k)
                st.subheader(f"Top {len(ranked)} job matches (LLM reranked)")
                for item in ranked:
                    job_id = item.get("job_id")
                    job = next((r for r in results if str(r["job_id"]) == str(job_id)), None)
                    if job is None:
                        continue
                    with st.container(border=True):
                        c1, c2 = st.columns([4, 1])
                        c1.markdown(f"**Rank {item.get('rank')} · {item.get('company')}**")
                        c2.metric("Fit", f"{item.get('fit_score', 0)}/100")
                        st.write(item.get("reason", ""))
                        parts = [
                            str(v)
                            for v in [
                                job.get("working_location"),
                                job.get("salary"),
                                f"RRF {job.get('score', 0):.3f}",
                            ]
                            if v
                        ]
                        if job.get("job_url"):
                            parts.append(f"[:arrow_upper_right: Apply]({job.get('job_url')})")
                        st.caption(" · ".join(parts))
                return

            st.subheader(f"Top {len(results)} job matches")
            for rank, job in enumerate(results, start=1):
                _render_job(job, rank)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            raise


def main() -> None:
    _render_diagnostics()

    if not st.session_state.get("embeddings_warmed"):
        with st.spinner("Loading embedding models (first time only — downloads models)…"):
            try:
                _warm_up_embeddings()
                st.session_state["embeddings_warmed"] = True
            except Exception as exc:
                st.warning(f"Could not pre-load embedding models: {exc}")

    with st.sidebar:
        st.header("Options")
        top_k = st.slider("Top K", min_value=3, max_value=30, value=10)
        expand_query = st.checkbox("Expand keywords with LLM (BM25)", value=True)
        rerank = st.checkbox("Rerank with LLM (uses resume profile)", value=False)

    tab_search, tab_discovery = st.tabs(["Hybrid Search", "Job Discovery"])
    with tab_search:
        _render_search(top_k, expand_query, rerank)
    with tab_discovery:
        render_discovery(get_cv=_safe_load_resume)


if __name__ == "__main__":
    main()
