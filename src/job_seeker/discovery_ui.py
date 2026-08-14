"""Shared Streamlit widget for interest-based job discovery.

Used by both the web demo (``app.py``) and the dashboard (``dashboard.py``) so
the discovery UI is defined once instead of copy-pasted.
"""

from collections.abc import Callable

import streamlit as st

from job_seeker.recommend import recommend_jobs, recommend_jobs_for_cv

__all__ = ["render_discovery"]


def render_discovery(get_cv: Callable[[], dict | None] | None = None) -> None:
    """Render the interest-based job discovery widget.

    Args:
        get_cv: Optional callable returning the candidate profile dict to use
            in "Interest + CV" mode. Callers that have no resume handy may pass
            ``None`` to skip the CV blend. Exceptions raised by ``get_cv`` are
            caught and downgraded to interest-only matching with a warning.
    """
    st.subheader("Interest-based job discovery")
    st.caption(
        "Describe your interests or goals; we match jobs using each company's "
        "background and focus, and explain why each one fits."
    )
    mode = st.radio(
        "Match using",
        ["Interest only", "Interest + CV"],
        horizontal=True,
        key="discovery_mode",
    )
    interest = st.text_input(
        "What are you interested in?",
        placeholder="e.g. interest in AI, want to learn more about LLMs",
        key="discovery_interest",
    )
    if not interest:
        st.info("Tell us what you're interested in to get job recommendations.")
        return

    if st.button("Find matching jobs", type="primary", key="discovery_btn"):
        with st.spinner("Analyzing company backgrounds & matching your interests…"):
            try:
                if mode == "Interest + CV":
                    cv = None
                    if get_cv is not None:
                        try:
                            cv = get_cv()
                        except Exception:
                            st.warning("No resume found — falling back to interest-only matching.")
                            cv = None
                    result = (
                        recommend_jobs_for_cv(interest, cv) if cv else recommend_jobs(interest)
                    )
                else:
                    result = recommend_jobs(interest)
            except Exception as exc:
                st.error(f"Discovery failed: {exc}")
                return

        recs = result.get("recommendations", [])
        if not recs:
            st.warning(
                "No matching jobs found. Try different wording, or run "
                "`job-seeker ingest` to index crawled jobs."
            )
            return

        st.subheader(f"Top {len(recs)} recommended jobs")
        for i, rec in enumerate(recs, start=1):
            with st.container(border=True):
                company = rec.get("company") or "Unknown company"
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**#{i} · {company}**")
                if rec.get("job_title"):
                    c1.caption(rec.get("job_title"))
                url = rec.get("job_url")
                if url:
                    c2.markdown(f"[:arrow_upper_right: Apply]({url})")
                meta = " · ".join(
                    str(v)
                    for v in [
                        rec.get("location") or rec.get("working_location"),
                        rec.get("salary"),
                        f"Job ID: {rec.get('job_id')}" if rec.get("job_id") else None,
                    ]
                    if v
                )
                st.caption(meta)
                if url:
                    st.caption(f"URL: {url}")
                st.write(f"**Why it fits:** {rec.get('reason', '')}")
                responsibilities = rec.get("Responsibilities")
                requirements = rec.get("Requirements")
                if responsibilities:
                    with st.expander("Responsibilities"):
                        st.write(responsibilities)
                if requirements:
                    with st.expander("Requirements"):
                        st.write(requirements)
