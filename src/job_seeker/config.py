import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

__all__ = ["BASE_DIR", "DATA_DIR", "RAW_DIR", "RESULTS_DIR", "Settings", "settings"]

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = BASE_DIR / "results"

load_dotenv(BASE_DIR / ".env")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def _optional_path(value: str | None) -> Path | None:
    return _resolve(value) if value else None


@dataclass(frozen=True)
class Settings:
    opencode_model: str = os.getenv("OPENCODE_MODEL", "opencode-go/deepseek-v4-flash")

    jobsdb_base_url: str = os.getenv(
        "JOBSDB_BASE_URL",
        "https://hk.jobsdb.com/programmer-jobs/in-Hong-Kong-SAR?sortmode=KeywordRelevance",
    )
    jobsdb_target_count: int = int(os.getenv("JOBSDB_TARGET_COUNT", "90"))
    jobsdb_output_file: Path = _resolve(
        os.getenv("JOBSDB_OUTPUT_FILE", "data/raw/jobsdb_analyst_programmer_jobs.json")
    )

    resume_pdf_path: Path = _resolve(os.getenv("RESUME_PDF_PATH", ""))
    resume_markdown_output: Path = _resolve(
        os.getenv("RESUME_MARKDOWN_OUTPUT", "results/resume.md")
    )
    resume_json_output: Path = _resolve(
        os.getenv("RESUME_JSON_OUTPUT", "results/resume.json")
    )
    cv_json_path: Path = _resolve(os.getenv("CV_JSON_PATH", "data/raw/cv.json"))
    cv_pdf_storage: Path = _resolve(
        os.getenv("CV_PDF_STORAGE", "data/raw/uploaded_resume.pdf")
    )

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_path: Path | None = _optional_path(os.getenv("QDRANT_PATH"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "jobs_collection")
    qdrant_dense_vector_name: str = "colbert"
    qdrant_sparse_vector_name: str = "bm25"

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "colbert-ir/colbertv2.0")
    sparse_embedding_model: str = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
    colbert_batch_size: int = int(os.getenv("COLBERT_BATCH_SIZE", "32"))

    qdrant_hnsw_m: int = int(os.getenv("QDRANT_HNSW_M", "16"))
    qdrant_hnsw_ef_construct: int = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "128"))
    qdrant_hnsw_full_scan_threshold: int = int(
        os.getenv("QDRANT_HNSW_FULL_SCAN_THRESHOLD", "10000")
    )

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    prefetch_limit: int = int(os.getenv("PREFETCH_LIMIT", "50"))
    search_limit: int = int(os.getenv("SEARCH_LIMIT", "10"))


settings = Settings()
