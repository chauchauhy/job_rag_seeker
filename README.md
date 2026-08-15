# Job Seeker

Pipeline that crawls Analyst Programmer jobs from JobsDB (HK), parses a resume PDF into markdown, and extracts structured JSON from it with an LLM (via the opencode CLI) for downstream vector-search matching.

## Project structure

```
job_seeker/
├── src/job_seeker/
│   ├── __init__.py            # package exports (models + version)
│   ├── __main__.py            # python -m job_seeker
│   ├── cli.py                 # argparse CLI (job-seeker)
│   ├── config.py              # settings loaded from .env
│   ├── models.py              # Pydantic domain models + cv_to_text()
│   ├── pipeline.py            # end-to-end Resume->Job matching pipeline
│   ├── advice.py              # LLM actionable advice (cover letter/interview)
│   ├── discovery_ui.py        # shared interest-based discovery Streamlit widget
│   ├── app.py                 # web demo (Streamlit)
│   ├── dashboard.py           # match dashboard (Streamlit, pure rendering)
│   ├── crawler.py             # JobsDB scraper (Selenium)
│   ├── resume.py              # PDF -> markdown -> LLM extraction
│   ├── rag.py                 # retrieve + LLM rerank
│   ├── recommend.py           # interest-based job recommendations
│   ├── query_expansion.py     # LLM keyword expansion for BM25
│   ├── evaluator.py           # per-requirement match evaluation
│   ├── llm.py                 # LLM CLI wrapper + JSON parsing
│   └── vector_db/
│       ├── qdrant.py          # collection management
│       ├── embeddings.py      # FastEmbed dense + BM25 wrappers
│       ├── schema.py          # job chunking
│       ├── ingest.py          # chunk + embed + upsert
│       └── search.py          # hybrid RRF search
├── tests/                     # pytest suite (models, chunking, JSON extraction)
├── data/raw/                  # crawled jobs + cv.json + mock_cv.json (gitignored)
├── results/                   # extracted resume markdown (gitignored)
├── repo/                      # third-party clones (gitignored)
├── .env                       # local configuration (copy of .env.sample)
├── .env.sample
├── requirements.txt
├── pyproject.toml
└── README.md
```

Design notes:
- **One concept per module** — models, LLM advice, and rendering are separate so each file has a single purpose and `__all__` declares its public API.
- **Single source of truth** — every prompt builder uses `models.cv_to_text()` for the candidate profile (handles both `model_dump()` snake_case and raw resume.json spaced keys).
- **No eager heavy imports** — `vector_db/__init__.py` is intentionally empty so importing the package doesn't pull in FastEmbed.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.sample` to `.env` and adjust values (especially `RESUME_PDF_PATH`). The opencode CLI must be on `PATH` for LLM extraction.

## Usage

```bash
# Scrape JobsDB analyst-programmer jobs
job-seeker crawl

# Convert resume PDF to markdown + extract structured JSON via LLM
job-seeker extract-resume [--pdf path/to/cv.pdf]

# Create the hybrid Qdrant collection (HNSW dense + BM25 sparse)
job-seeker init-vector-db [--recreate]

# Chunk + embed (dense + BM25) crawled jobs and upsert into Qdrant
job-seeker ingest [--path data/raw/jobsdb_<JOB_NAME>_jobs.json] [--recreate]

# Hybrid RRF search over jobs (semantic + lexical); --expand boosts the BM25
# branch with LLM-expanded keywords
job-seeker search "Java Spring Boot machine learning" --expand

# RAG: retrieve jobs, then rerank top-k with the LLM against your resume
job-seeker rag "Java Spring Boot machine learning" [--top-k 10] [--no-rerank]

# Web demo (Streamlit)
streamlit run src/job_seeker/app.py
```

## Web demo

`streamlit run src/job_seeker/app.py` opens a browser UI with:
- **Vector index diagnostics**: confirms whether the **BM25 sparse index is ACTIVE**, plus HNSW config, dense model/dimension, and collection size.
- **Hybrid search**: enter any query, choose `top-k`, optionally expand keywords with the LLM for the BM25 branch and/or rerank results with the LLM against your resume.
- **Job Discovery**: describe your interests/goals (e.g. *"interest in AI, want to learn more…"*) and get the top 5 recommended jobs matched on each company's background, with a one-line "why it fits" — optionally blended with your CV profile.
- Results are shown with RRF scores, job links, locations, and matched snippets.

## Match dashboard

`streamlit run src/job_seeker/dashboard.py` opens a dark-theme dashboard with:

- **Top 5 potential jobs**: real jobs retrieved from Qdrant (run `job-seeker crawl` + `job-seeker ingest` first) — with match scores, a skill-fit radar, evidence, and gap analysis.
- **Interview Prep**: likely interview questions with suggested answers for each job.
- **Suggestions to Raise**: concrete talking points, strengths to emphasise, and questions to ask the interviewer.
- **Job Discovery**: interest-based recommendations (`Interest only` or `Interest + CV`), reusing the same company-background matching as the web demo.

Jobs generate match reports, interview prep, and suggestions on demand with the LLM.

## Query expansion

`job-seeker search "query" --expand` (or the "Expand keywords" toggle in the web demo) sends the query to opencode, which produces a set of concrete related skills/technologies/synonyms. The expanded text is embedded with the BM25 sparse model and used for the lexical branch, so keyword recall covers terms you didn't type (e.g. `Android Firebase` → `Kotlin Jetpack Compose Retrofit Firestore ...`).

## RAG / retrieval

- **No Qdrant server needed**: set `QDRANT_PATH` (default `qdrant_storage/`) to run Qdrant fully embedded in-process; the collection is saved as files in that folder inside the project. To use a server instead, uncomment `QDRANT_URL`.
- **Dense vectors**: ColBERT (`EMBEDDING_MODEL`, default `colbert-ir/colbertv2.0`) via FastEmbed produces one 128-d vector per token. Qdrant stores these as dense **multi-vectors** and scores them with **MAX_SIM late interaction** (tuned via `QDRANT_HNSW_M`, `QDRANT_HNSW_EF_CONSTRUCT`).
- **Sparse vectors**: BM25 (`Qdrant/bm25`) enables lexical keyword matching that vector search misses.
- **Hybrid fusion**: ColBERT + BM25 prefetch queries are combined with **Reciprocal Rank Fusion (RRF)**.
- **Chunking**: job postings are split into overlapping chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP`) so long descriptions retrieve well; results are deduplicated to one result per job.
- **Reranking**: `rag` optionally reranks retrieved jobs with the LLM against your extracted resume profile.

> **Note**: switching embedding models changes the vector schema (ColBERT is a 128-d multi-vector, not a single dense vector). Recreate the collection and re-ingest with `job-seeker ingest --recreate`.

All paths and settings are configurable through `.env`.
