"""Streamlit dashboard: dark-mode analytics UI for the Resume-to-Job matching pipeline.

Run from the project root:
    .venv\\Scripts\\streamlit run src/job_seeker/dashboard.py

Dark theme: this app relies on `.streamlit/config.toml` where
`theme.base = "dark"` is already set. To configure it yourself, create
`.streamlit/config.toml` in the project root with:

    [theme]
    base = "dark"
    primaryColor = "#00D4FF"
    backgroundColor = "#0E1117"

This module only renders the dashboard. Domain models live in
:mod:`job_seeker.models`, advice generation in :mod:`job_seeker.advice`, and
the shared discovery widget in :mod:`job_seeker.discovery_ui`.

The app uses the real extracted CV at `data/raw/cv.json` when that file exists
(it is written by `job-seeker extract-resume` or the in-app CV manager); it
falls back to the demo CV at `data/raw/mock_cv.json`. Potential jobs come from
the real Qdrant collection once ingested (run `job-seeker ingest`).
"""

from __future__ import annotations

import json
import platform
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Work around a hanging Windows WMI query in platform.system() that freezes
# Streamlit's import (env_util.py) when the WMI service is unhealthy.
if sys.platform == "win32":
    platform._wmi = None

import plotly.graph_objects as go
import streamlit as st

from job_seeker.advice import generate_actionable_advice
from job_seeker.config import RAW_DIR, settings
from job_seeker.crawler import load_existing_jobs, upsert_jobs
from job_seeker.discovery_ui import render_discovery
from job_seeker.logging_setup import setup_logging
from job_seeker.market_skills import (
    SALARY_BANDS,
    SKILL_DICT_PATH,
    candidate_gap,
    extract_skills,
    filter_jobs,
    load_jobs,
    top_companies,
)
from job_seeker.models import ActionableAdvice, ExtractedCV
from job_seeker.pipeline import evaluate_job_match, load_cv, retrieve_matching_jobs
from job_seeker.resume import process_resume
from job_seeker.vector_db.ingest import (
    ingest_jobs_dir,
    ingest_jobs_list,
    validate_jobs_data,
)
from job_seeker.vector_db.qdrant import collection_info

setup_logging()

st.set_page_config(
    page_title="Job Match Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_CV_PATH = settings.cv_json_path
MOCK_CV_PATH = RAW_DIR / "mock_cv.json"

__all__ = ["main"]


def _load_cv() -> ExtractedCV:
    """Load the real CV when available, otherwise fall back to the mock CV.

    Prefers the CV updated in this session (``st.session_state["cv"]``), then
    the canonical ``data/raw/cv.json``, then the demo ``data/raw/mock_cv.json``.
    """
    cached = st.session_state.get("cv")
    if cached is not None:
        return cached
    for path in (DEFAULT_CV_PATH, MOCK_CV_PATH):
        if Path(path).is_file():
            try:
                return load_cv(path)
            except Exception:
                continue
    st.warning("No CV found. Upload a resume PDF in the Profile & Jobs Manager tab "
               "(or run `job-seeker extract-resume --pdf <your_cv.pdf>` locally).")
    return ExtractedCV(Candidate_Name="")


def _clear_job_cache() -> None:
    """Drop per-job match-report / advice caches so stale results don't linger."""
    for key in list(st.session_state.keys()):
        if key.startswith("report_") or key.startswith("advice_"):
            del st.session_state[key]


_PROGRESS_SPANS = {
    "embed": (0.0, 0.7),
    "upsert": (0.7, 0.3),
    "convert": (0.0, 0.5),
    "extract": (0.5, 0.45),
    "done": (1.0, 0.0),
}


def _run_with_progress(title: str, fn, done_label: str):
    """Run a long callable while rendering a live progress bar + caption.

    ``fn`` receives an ``on_progress(stage, fraction, message)`` callback and
    its return value is returned to the caller.
    """
    with st.status(title, expanded=True) as status:
        bar = st.progress(0.0)
        label = st.empty()

        def on_progress(stage: str, fraction: float, message: str) -> None:
            base, span = _PROGRESS_SPANS.get(stage, (0.0, 1.0))
            bar.progress(min(1.0, base + span * max(0.0, min(fraction, 1.0))))
            label.caption(message)

        result = fn(on_progress)
        status.update(label=done_label, state="complete")
        return result


def _new_job_id(company: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{company} {title}".lower()).strip("-")[:40]
    slug = slug or "job"
    return f"manual_{slug}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _score_color(score: int) -> str:
    if score > 80:
        return "#2ECC71"  # green
    if score > 50:
        return "#F1C40F"  # yellow
    return "#E74C3C"  # red


def _render_tags(tags: list[str]) -> None:
    if not tags:
        return
    bubbles = "".join(
        f'<span style="background:#1B2430;color:#00D4FF;border:1px solid #00D4FF;'
        f'border-radius:12px;padding:3px 12px;margin:0 6px 6px 0;'
        f'display:inline-block;font-size:0.9rem;">{tag}</span>'
        for tag in tags
    )
    st.markdown(f'<div>{bubbles}</div>', unsafe_allow_html=True)


def _render_radar(score: int, gap_text: str) -> None:
    """Plot a dark-theme radar chart: Candidate skill vs JD requirement level."""
    categories = [
        "Programming",
        "Backend / APIs",
        "Data & ML",
        "Cloud & DevOps",
        "Frontend",
        "Soft Skills",
    ]
    base = max(0, min(score, 95))
    candidate = [
        base + 5,
        base,
        max(10, base - 12 if "ML" not in gap_text or "machine learning" in gap_text.lower() else base + 8),
        max(10, base - 18),
        max(10, base - 25),
        max(15, base - 5),
    ]
    jd = [base + 12, base + 8, base + 6, base + 10, base + 14, base + 4]
    candidate = [min(v, 100) for v in candidate]
    jd = [min(v, 100) for v in jd]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=candidate,
            theta=categories,
            fill="toself",
            name="Candidate Skill Level",
            line=dict(color="#00D4FF", width=2),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=jd,
            theta=categories,
            fill="toself",
            name="JD Requirement Level",
            line=dict(color="#F1C40F", width=2),
            fillcolor="rgba(241,196,15,0.15)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Skill Fit Radar",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EAF2"),
        margin=dict(l=60, r=60, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
        ),
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Potential jobs (real Qdrant jobs, empty when the index is unpopulated)
# ---------------------------------------------------------------------------
def get_potential_jobs(cv: ExtractedCV) -> list[dict]:
    """Return the top potential jobs from the real Qdrant collection.

    Each dict: {label, title, score, evidence, gap, job}.
    Returns an empty list when the collection is missing or empty, or when the
    retrieval fails.
    """
    try:
        info = collection_info()
        if not (info.get("exists") and info.get("points_count", 0) > 0):
            return []
        jobs = retrieve_matching_jobs(cv, top_k=5)
        return [
            {
                "label": f"{j.get('company') or 'Unknown'} — {j.get('matched_section') or 'Job'}",
                "title": j.get("company") or "Unknown company",
                "score": None,
                "evidence": None,
                "gap": None,
                "job": j,
            }
            for j in jobs
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Rendering: job dashboard
# ---------------------------------------------------------------------------
def _render_job_dashboard(
    title: str,
    score: int | None,
    evidence: str | None,
    gap: str | None,
    advice: ActionableAdvice | None,
) -> None:
    score = score or 0
    color = _score_color(score)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"<h2 style='margin-bottom:0;'>{title}</h2>"
            f"<span style='color:#8A94A6;'>Company &amp; details from match report</span>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='text-align:center;background:#1B2430;border-radius:12px;"
            f"padding:12px;border-left:4px solid {color};'>"
            f"<div style='font-size:2.2rem;font-weight:800;color:{color};'>{score}</div>"
            f"<div style='color:#8A94A6;font-size:0.8rem;'>MATCH SCORE</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    radar_col, detail_col = st.columns([3, 2])
    with radar_col:
        _render_radar(score, gap or "")
    with detail_col:
        st.subheader("Evidence")
        st.info(evidence or "")
        st.subheader("Gap Analysis")
        st.warning(gap or "")

    st.divider()
    st.subheader("Actionable Advice")

    if advice is None:
        st.info("No advice generated yet.")
        return

    _render_tags(advice.Tags)

    tab_letter, tab_interview, tab_raise = st.tabs(
        ["Cover Letter", "Interview Prep", "Suggestions to Raise"]
    )
    with tab_letter:
        st.markdown(advice.Cover_Letter_Draft)
        st.download_button(
            "Download cover letter (.md)",
            data=advice.Cover_Letter_Draft,
            file_name="cover_letter.md",
            mime="text/markdown",
        )
    with tab_interview:
        if advice.Interview_Prep:
            for i, item in enumerate(advice.Interview_Prep, start=1):
                with st.expander(f"Q{i}. {item.Question}"):
                    st.markdown(f"**Suggested Answer:**\n\n{item.Suggested_Answer}")
        else:
            st.info("No interview questions generated.")
    with tab_raise:
        if advice.Suggestions_To_Raise:
            for s in advice.Suggestions_To_Raise:
                st.markdown(f"- {s}")
        else:
            st.info("No suggestions generated.")


def _render_real_job(cv: ExtractedCV, job: dict) -> None:
    jid = str(job["job"].get("job_id") or job["title"])
    parts = [
        str(v)
        for v in [job["job"].get("working_location"), job["job"].get("salary")]
        if v
    ]
    if job["job"].get("job_url"):
        parts.append(f"[:arrow_upper_right: Apply]({job['job'].get('job_url')})")
    st.markdown(f"<h2 style='margin-bottom:0;'>{job['title']}</h2>", unsafe_allow_html=True)
    st.caption(" · ".join(parts) or "Real job from Qdrant")
    if job["job"].get("matched_text"):
        with st.expander("Matched snippet"):
            st.write(job["job"].get("matched_text", ""))

    report = st.session_state.get(f"report_{jid}")
    advice = st.session_state.get(f"advice_{jid}")

    if report is None:
        if st.button("Generate match report & interview prep", type="primary", key=f"gen_{jid}"):
            with st.spinner("Evaluating match & preparing interview materials…"):
                try:
                    report = evaluate_job_match(cv, job["job"])
                    st.session_state[f"report_{jid}"] = report
                    st.session_state[f"advice_{jid}"] = generate_actionable_advice(
                        cv.model_dump(), report.model_dump()
                    )
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
                    return
            st.rerun()
        st.info("Click to generate a match report and interview prep for this job.")
        return

    _render_job_dashboard(
        report.Job_Title,
        report.Match_Score,
        report.Evidence,
        report.Gap_Analysis,
        advice,
    )


def _render_match_dashboard(cv: ExtractedCV) -> None:
    jobs = get_potential_jobs(cv)

    with st.sidebar:
        st.title("Candidate")
        st.markdown(f"**{cv.Candidate_Name}**")
        st.caption(
            f"{cv.Total_Years_of_Experience or '?'} yrs experience · "
            f"{len(cv.Hard_Skills)} hard skills"
        )
        with st.expander("Hard Skills"):
            st.write(", ".join(cv.Hard_Skills))
        st.divider()
        st.header("Potential Jobs (Top 5)")
        st.caption("Real jobs from Qdrant")
        labels = [j["label"] for j in jobs]
        selected = st.radio("Select a job to review", labels, index=0)

    st.title("Resume-to-Job Match Dashboard")
    st.caption("Phase 5 · Output Generation & Actionable Advice")

    if not jobs:
        st.warning(
            "No jobs in the vector index yet. Run `job-seeker crawl` then "
            "`job-seeker ingest` to index real jobs, then refresh."
        )
        return

    job = jobs[labels.index(selected)]
    _render_real_job(cv, job)


def _render_cv_manager(cv: ExtractedCV) -> None:
    st.subheader("Update CV")
    st.caption(
        "Upload a new resume PDF. It is stored under `data/raw/`, extracted into "
        "`data/raw/cv.json`, and job suggestions are refreshed against the new profile."
    )

    uploaded = st.file_uploader("Resume (PDF)", type=["pdf"], key="cv_uploader")
    if uploaded is not None:
        st.caption(f"{uploaded.name} · {len(uploaded.getvalue()) / 1024:.1f} KB")
        if st.button("Save & re-suggest jobs", type="primary", key="cv_save_btn"):
            try:
                settings.cv_pdf_storage.parent.mkdir(parents=True, exist_ok=True)
                settings.cv_pdf_storage.write_bytes(uploaded.getvalue())
                extracted = _run_with_progress(
                    "Extracting CV…",
                    lambda cb: process_resume(
                        settings.cv_pdf_storage, on_progress=cb
                    ),
                    "CV extraction complete.",
                )
                new_cv = ExtractedCV.model_validate(extracted)
                st.session_state["cv"] = new_cv
                _clear_job_cache()
                st.success(f"CV updated for {new_cv.Candidate_Name}.")
                st.rerun()
            except Exception as exc:
                st.error(f"CV update failed: {exc}")

    st.divider()
    st.subheader("Current CV")
    st.markdown(
        f"**{cv.Candidate_Name}** · {cv.Total_Years_of_Experience or '?'} yrs "
        f"· {len(cv.Hard_Skills)} hard skills"
    )
    if cv.Hard_Skills:
        st.write(", ".join(cv.Hard_Skills))


def _render_job_manager() -> None:
    st.subheader("Add Job Description")
    st.caption(
        "Appends a job to `data/raw/` and re-embeds it into the vector database, "
        "so it is considered in future suggestions."
    )

    with st.form("jd_form"):
        company = st.text_input("Company", placeholder="e.g. Acme Labs")
        title = st.text_input("Job Title", placeholder="e.g. AI Engineer")
        salary = st.text_input("Salary (optional)", placeholder="e.g. 45K - 55K")
        location = st.text_input("Working location (optional)", placeholder="e.g. Hong Kong Island")
        url = st.text_input("Job URL (optional)")
        responsibilities = st.text_area("Responsibilities", height=120)
        requirements = st.text_area("Requirements", height=120)
        submitted = st.form_submit_button("Save & re-embed", type="primary")

    if submitted:
        if not company and not title:
            st.warning("Please provide at least a company or job title.")
            return
        job = {
            "job_id": _new_job_id(company, title),
            "job_url": url,
            "company": company,
            "salary": salary,
            "working_location": location,
            "Responsibilities": responsibilities,
            "Requirements": requirements,
        }
        try:
            path = settings.jobsdb_output_file
            path.parent.mkdir(parents=True, exist_ok=True)
            merged = upsert_jobs(load_existing_jobs(path), [job])
            path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            count = _run_with_progress(
                "Embedding & upserting job…",
                lambda cb: ingest_jobs_list([job], on_progress=cb),
                "Job embedded.",
            )
            _clear_job_cache()
            st.success(f"Job saved & embedded ({count} vectors).")
            st.rerun()
        except Exception as exc:
            st.error(f"Job add failed: {exc}")

    st.divider()
    st.subheader("Upload Jobs File (JSON)")
    st.caption(
        "Upload a jobs file — a JSON list of job objects like the crawler output. "
        "It is saved under `data/raw/jobs/` and re-embedded into the vector database."
    )
    uploaded = st.file_uploader("Jobs file (.json)", type=["json"], key="jobs_file_uploader")
    mode = st.radio(
        "Index mode",
        ["Append", "Rebuild"],
        horizontal=True,
        key="jobs_upload_mode",
    )
    if mode == "Rebuild":
        st.caption(
            "Rebuild recreates the vector collection from **all** files in "
            "`data/raw/jobs/` (the current index is replaced)."
        )
    if st.button(
        "Upload & re-embed",
        type="primary",
        key="jobs_upload_btn",
        disabled=uploaded is None,
    ):
        if uploaded is None:
            return
        try:
            data = json.loads(uploaded.getvalue().decode("utf-8"))
            jobs, warnings = validate_jobs_data(data)
            target = settings.jobs_dir / Path(uploaded.name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            start = time.time()

            def _ingest(cb):
                if mode == "Rebuild":
                    return ingest_jobs_dir(recreate=True, on_progress=cb)
                return ingest_jobs_list(jobs, on_progress=cb)

            count = _run_with_progress(
                "Rebuilding vector index…" if mode == "Rebuild" else "Embedding & upserting…",
                _ingest,
                "Index updated.",
            )
            elapsed = time.time() - start
            message = (
                f"Saved {len(jobs)} jobs to `{target.name}` and embedded "
                f"{count} vectors in {elapsed:.1f}s."
            )
            if warnings:
                shown = ", ".join(warnings[:5])
                more = f" (+{len(warnings) - 5} more)" if len(warnings) > 5 else ""
                message += f"\n\nSkipped (no job text): {shown}{more}"
            _clear_job_cache()
            st.success(message)
            st.rerun()
        except Exception as exc:
            st.error(f"Upload failed: {exc}")


def _render_skill_chips(skills: list[dict], color: str, label: str) -> None:
    """Render market skills as tag chips with their job counts."""
    if not skills:
        st.caption("None")
        return
    bubbles = "".join(
        f'<span style="background:#1B2430;color:{color};border:1px solid {color};'
        f'border-radius:12px;padding:3px 12px;margin:0 6px 6px 0;'
        f'display:inline-block;font-size:0.9rem;">'
        f"{skill} · {count} jobs</span>"
        for skill, count in skills
    )
    st.markdown(
        f"<div><span style='color:#8A94A6;font-size:0.85rem;'>{label}:</span>"
        f"<div>{bubbles}</div></div>",
        unsafe_allow_html=True,
    )


def _render_market_skills(cv: ExtractedCV) -> None:
    """Render the market-skills tab: what skill is hot in the crawled job market."""
    st.title("Market Skills")
    st.caption(
        "Which skills are hot across crawled Analyst Programmer postings, based on "
        "requirements text. Keywords live in `data/skills.json` — edit that file to "
        "refresh the dictionary without touching code."
    )

    jobs = load_jobs()
    if not jobs:
        st.warning(
            "No crawled jobs found. Run `job-seeker crawl` to fetch postings, "
            "then refresh."
        )
        return

    top_n = st.slider("Top skills to show", min_value=5, max_value=40, value=20, key="market_top_n")
    locations = sorted({str(job.get("working_location") or "") for job in jobs if job.get("working_location")})
    sel_locations = st.multiselect("Filter by working location", locations, key="market_location")
    sel_bands = st.multiselect(
        "Filter by salary band (per month)",
        list(SALARY_BANDS),
        key="market_salary_band",
        help="Many listings omit salary and fall into the '—' band.",
    )
    filtered = filter_jobs(jobs, locations=sel_locations, salary_bands=sel_bands)
    if not filtered:
        st.info("No jobs match the current filters.")
        return

    rows = extract_skills(filtered)
    if not rows:
        st.info(
            "No skills from the dictionary were found in these postings. "
            "Add or refine keywords in `data/skills.json`."
        )
        return

    total_jobs = len(filtered)
    st.caption(f"{total_jobs} job(s) match the filters.")

    top_rows = [r for r in rows if r["count"] > 0][:top_n]

    st.subheader("Top skills by job count")
    fig = go.Figure(
        go.Bar(
            x=[r["count"] for r in top_rows],
            y=[r["skill"] for r in top_rows],
            orientation="h",
            marker_color="#00D4FF",
            text=[str(r["count"]) for r in top_rows],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EAF2"),
        margin=dict(l=40, r=60, t=30, b=30),
        height=max(320, 28 * len(top_rows)),
        xaxis_title="Jobs mentioning skill",
    )
    st.plotly_chart(fig, width="stretch")

    col_table, col_gap = st.columns([3, 2])
    with col_table:
        st.subheader("Skill table")
        search = st.text_input("Filter skills", placeholder="e.g. Java", key="market_skill_search")
        table_rows = [
            {
                "Skill": r["skill"],
                "# Jobs": r["count"],
                "% of jobs": f"{r['count'] / total_jobs * 100:.0f}%",
            }
            for r in rows
            if not search or search.lower() in r["skill"].lower()
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.subheader("Which jobs need a skill?")
        selected_skill = st.selectbox("Select a skill", [r["skill"] for r in rows], key="market_drilldown")
        selected = next(r for r in rows if r["skill"] == selected_skill)
        by_id = {str(job.get("job_id") or ""): job for job in filtered}
        for job_id in selected["job_ids"]:
            job = by_id.get(job_id)
            if not job:
                continue
            meta = " · ".join(
                str(v)
                for v in [
                    job.get("working_location"),
                    job.get("salary"),
                    job.get("company"),
                ]
                if v
            )
            st.markdown(f"- **{job.get('company') or 'Unknown'}** — {meta or 'details unknown'}")
            if job.get("job_url"):
                st.markdown(f"  [:arrow_upper_right: Apply]({job.get('job_url')})")

    with col_gap:
        st.subheader("Your skills vs. the market")
        gap = candidate_gap(cv.Hard_Skills, rows, top_n=top_n)
        st.markdown("**In demand & you already have**")
        _render_skill_chips(
            [(r["skill"], r["count"]) for r in gap["matched"]],
            color="#2ECC71",
            label="matched",
        )
        st.markdown("**Hot skills you may lack**")
        _render_skill_chips(
            [(r["skill"], r["count"]) for r in gap["missing"]],
            color="#F1C40F",
            label="upskill targets",
        )
        st.caption("Comparison uses your CV's hard skills against the dictionary aliases.")

    st.divider()
    st.subheader("Top companies hiring")
    companies = top_companies(filtered, top_n=10)
    fig2 = go.Figure(
        go.Bar(
            x=[c["count"] for c in companies],
            y=[c["company"] for c in companies],
            orientation="h",
            marker_color="#F1C40F",
            text=[str(c["count"]) for c in companies],
            textposition="outside",
        )
    )
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EAF2"),
        margin=dict(l=40, r=60, t=30, b=30),
        xaxis_title="Job postings",
    )
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Update the skill dictionary"):
        st.markdown(
            f"Skills are matched against keyword aliases in `{SKILL_DICT_PATH}`. "
            "Add a canonical name as a key and a list of aliases as its value, "
            "e.g. `\"Kotlin\": [\"kotlin\"]`. Compound terms win over shorter "
            "ones, and each skill counts once per job."
        )


def main() -> None:
    cv = _load_cv()

    tab_dashboard, tab_discovery, tab_market, tab_manager = st.tabs(
        ["Match Dashboard", "Job Discovery", "Market Skills", "Profile & Jobs Manager"]
    )
    with tab_discovery:
        render_discovery(get_cv=lambda: cv.model_dump())
    with tab_dashboard:
        _render_match_dashboard(cv)
    with tab_market:
        _render_market_skills(cv)
    with tab_manager:
        _render_cv_manager(cv)
        _render_job_manager()


if __name__ == "__main__":
    main()
