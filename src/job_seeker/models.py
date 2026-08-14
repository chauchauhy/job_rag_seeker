"""Shared domain models.

Pure Pydantic models with no configuration or I/O dependencies, so they can be
imported anywhere (apps, CLI, tests) without pulling in heavy runtime deps.
"""

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ExtractedCV",
    "JobMatchReport",
    "InterviewPrepItem",
    "ActionableAdvice",
    "cv_to_text",
]


class ExtractedCV(BaseModel):
    """Structured candidate profile extracted from a resume.

    ``model_dump()`` returns snake_case keys (``Candidate_Name``, ``Hard_Skills``),
    while raw extraction output (e.g. ``data/raw/cv.json`` from
    :mod:`job_seeker.resume`) uses spaced keys (``Candidate Name``,
    ``Hard Skills``). :func:`cv_to_text` accepts both.
    """

    model_config = ConfigDict(
        alias_generator=lambda name: name.replace("_", " "),
        populate_by_name=True,
    )

    Candidate_Name: str
    Total_Years_of_Experience: int | None = None
    Hard_Skills: list[str] = Field(default_factory=list)
    Soft_Skills: list[str] = Field(default_factory=list)
    Work_Experience: list[dict] = Field(default_factory=list)
    Education: list[dict] = Field(default_factory=list)

    def to_profile_text(self) -> str:
        """Serialize this CV to a compact text block for LLM prompts."""
        return cv_to_text(self.model_dump())


class JobMatchReport(BaseModel):
    """LLM evaluation of a CV against a single job ad."""

    Job_Title: str
    Match_Score: int = Field(ge=0, le=100)
    Evidence: str
    Gap_Analysis: str


class InterviewPrepItem(BaseModel):
    """A single interview question with a suggested answer."""

    Question: str
    Suggested_Answer: str


class ActionableAdvice(BaseModel):
    """Tailored application advice produced for a matched job."""

    Tags: list[str] = Field(..., min_length=1)
    Cover_Letter_Draft: str
    Interview_Prep: list[InterviewPrepItem] = Field(default_factory=list)
    Suggestions_To_Raise: list[str] = Field(default_factory=list)


def cv_to_text(cv: dict) -> str:
    """Serialize a CV dict to a compact text block for LLM prompts.

    Accepts both snake_case keys (``ExtractedCV.model_dump()``) and the spaced
    keys found in raw extraction output (``data/raw/cv.json``). This is the
    single source of truth used by every prompt-builder in the project.
    """

    def pick(upper: str, spaced: str, default=None):
        return cv.get(upper, cv.get(spaced, default))

    name = pick("Candidate_Name", "Candidate Name", "N/A")
    years = pick("Total_Years_of_Experience", "Total Years of Experience")
    skills = pick("Hard_Skills", "Hard Skills", []) or []
    soft = pick("Soft_Skills", "Soft Skills", []) or []
    experience = pick("Work_Experience", "Work Experience", []) or []

    lines = [
        f"Candidate: {name}",
        f"Total Years of Experience: {years}",
        f"Hard Skills: {', '.join(skills)}",
        f"Soft Skills: {', '.join(soft)}",
        "Work Experience:",
    ]
    for job in experience:
        title = job.get("Job Title") or job.get("Job_Title") or ""
        company = job.get("Company") or ""
        duties = job.get("Responsibilities") or []
        if isinstance(duties, list):
            duties = " | ".join(duties)
        lines.append(f"- {title} @ {company}: {duties}")
    return "\n".join(lines)
