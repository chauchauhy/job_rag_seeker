"""Job Seeker - crawl jobs, parse resumes, and match them with vector search."""

from job_seeker.models import (
    ActionableAdvice,
    ExtractedCV,
    InterviewPrepItem,
    JobMatchReport,
    cv_to_text,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ExtractedCV",
    "JobMatchReport",
    "InterviewPrepItem",
    "ActionableAdvice",
    "cv_to_text",
]
