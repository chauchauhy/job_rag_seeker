"""Non-LLM market-skill analysis over crawled job postings.

Answers "what skill is hot in the current market" by counting how many job
postings mention each skill, using a keyword/alias dictionary. The dictionary
lives in ``data/skills.json`` (edit it to keep keywords up to date); when that
file is missing it is re-created from an embedded seed copy, mirroring the
mock-CV fallback pattern used elsewhere in the project.

Matching is case-insensitive and span-aware: longer/compound aliases (e.g.
``spring boot``) win over shorter ones (``spring``) so a phrase is attributed
to the most specific skill. A skill is counted at most once per job, so the
count is "how many distinct postings demand this skill", not raw mentions.

Everything here is pure logic (plain dicts in, plain dicts out) so it can be
unit-tested and reused from the CLI or the Streamlit dashboard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from job_seeker.config import DATA_DIR, settings
from job_seeker.crawler import load_existing_jobs

__all__ = [
    "SKILL_DICT_PATH",
    "SALARY_BANDS",
    "load_skill_dict",
    "load_jobs",
    "normalize_text",
    "extract_skills",
    "top_companies",
    "filter_jobs",
    "salary_band",
    "candidate_gap",
]

SKILL_DICT_PATH = DATA_DIR / "skills.json"

SALARY_BANDS = ("< 40K", "40K\u201360K", "\u2265 60K", "\u2014")

# Aliases are matched in order of descending length, so compound terms like
# "spring boot" always take precedence over "spring" within the same phrase.
SKILL_DICT_FALLBACK: dict[str, list[str]] = {
    "Java": ["java", "java/jee", "java ee", "j2ee", "core java"],
    "JavaScript": ["javascript"],
    "TypeScript": ["typescript"],
    "Python": ["python", "python3"],
    "C#": ["c#", "csharp"],
    "C++": ["c++", "cpp"],
    "PHP": ["php"],
    "Golang": ["golang"],
    "Node.js": ["node.js", "nodejs", "node js", "node"],
    "Spring Boot": ["spring boot"],
    "Spring": ["spring framework", "spring mvc", "spring"],
    "Hibernate": ["hibernate"],
    ".NET": [".net core", ".net framework", ".net"],
    "ASP.NET": ["asp.net", "asp.net mvc", "asp.net core"],
    "Entity Framework": ["entity framework"],
    "React": ["react", "react.js", "reactjs"],
    "React Native": ["react native"],
    "Vue": ["vue", "vue.js", "vuejs"],
    "Angular": ["angular", "angular.js", "angularjs"],
    "jQuery": ["jquery"],
    "AJAX": ["ajax"],
    "Flutter": ["flutter"],
    "Next.js": ["next.js", "nextjs"],
    "Redux": ["redux"],
    "FastAPI": ["fastapi"],
    "Django": ["django"],
    "Flask": ["flask"],
    "Laravel": ["laravel"],
    "Webpack": ["webpack"],
    "HTML5/CSS3": ["html5", "css3", "html", "css"],
    "Bootstrap": ["bootstrap"],
    "PowerBuilder": ["powerbuilder"],
    "JSP": ["jsp"],
    "JSF": ["jsf"],
    "EJB": ["ejb"],
    "Crystal Report": ["crystal report"],
    "Jasper Report": ["jasper report", "jasper"],
    "Visual Basic": ["vb.net", "visual basic", "vb"],
    "COBOL": ["cobol"],
    "SAP": ["sap", "sap s/4hana", "s/4hana"],
    "ABAP": ["abap"],
    "SQL": ["sql", "rdbms"],
    "SQL Server": ["sql server", "mssql", "ms sql server"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "Oracle": ["oracle"],
    "PL/SQL": ["pl/sql"],
    "MongoDB": ["mongodb", "mongo"],
    "Sybase": ["sybase"],
    "NoSQL": ["nosql"],
    "Redis": ["redis"],
    "GraphQL": ["graphql"],
    "ETL": ["etl"],
    "Data Warehouse": ["data warehouse", "data warehousing"],
    "AWS": ["aws"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "OpenShift": ["openshift", "red hat openshift"],
    "CI/CD": ["ci/cd", "ci cd"],
    "Linux": ["linux"],
    "Unix": ["unix"],
    "Shell Scripting": ["shell scripting", "shell script", "bash", "shell"],
    "PowerShell": ["powershell"],
    "Git": ["git", "github", "gitlab"],
    "Jenkins": ["jenkins"],
    "Terraform": ["terraform"],
    "Microservices": ["microservices", "micro-service", "micro service"],
    "REST APIs": ["rest api", "restful api", "restful", "rest"],
    "Message Queue": ["message queue", "message queueing"],
    "Apache Camel": ["camel"],
    "Kafka": ["kafka"],
    "Nginx": ["nginx"],
    "Tomcat": ["tomcat"],
    "WebSphere": ["websphere", "ibm websphere"],
    "WebLogic": ["weblogic", "oracle weblogic"],
    "Machine Learning": ["machine learning"],
    "AI": ["artificial intelligence", "ai"],
    "Prompt Engineering": ["prompt engineering"],
    "LLM": ["llm", "large language model"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "RPA": ["rpa"],
    "UiPath": ["uipath", "ui path"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel"],
    "HL7": ["hl7"],
    "FHIR": ["fhir"],
    "DICOM": ["dicom"],
    "Boomi": ["boomi"],
    "Pega": ["pega"],
}

_ALIAS_EDGE = re.compile(r"[a-z0-9]")
_NON_WORD = re.compile(r"\W+", re.UNICODE)
_SALARY_NUM = re.compile(r"\d[\d,]*")
_WHITESPACE = re.compile(r"\s+")


def _norm_token(token: str) -> str:
    """Lowercase and collapse separators so 'Node.JS' ~ 'node.js' ~ 'nodejs'."""
    return _NON_WORD.sub("", token).lower()


def load_skill_dict(path: str | Path | None = None) -> dict[str, list[str]]:
    """Load the skill dictionary, re-creating it from the embedded seed if missing.

    Returns ``{canonical skill name: [aliases, ...]}`` where each list has the
    canonical name prepended so it always matches its own spelling.
    """
    path = Path(path) if path else SKILL_DICT_PATH
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(SKILL_DICT_FALLBACK, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raw = SKILL_DICT_FALLBACK
    else:
        try:
            with open(path, encoding="utf-8") as json_file:
                raw = json.load(json_file)
        except (OSError, json.JSONDecodeError):
            raw = SKILL_DICT_FALLBACK
    aliases = {}
    for name, value in raw.items():
        value = value if isinstance(value, list) else [value]
        aliases[name] = list(dict.fromkeys([name, *(str(v) for v in value)]))
    return aliases


def load_jobs(path: str | Path | None = None) -> list[dict]:
    """Load the crawled job postings, empty list when the file is missing."""
    return load_existing_jobs(Path(path) if path else settings.jobsdb_output_file)


def normalize_text(text: str) -> str:
    """Lowercase and normalise whitespace for alias matching."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def _match_skills(text: str, skill_aliases: dict[str, list[str]]) -> set[str]:
    """Return the set of skills mentioned in ``text`` (deduped, span-aware)."""
    pairs = [
        (skill, alias)
        for skill, aliases in skill_aliases.items()
        for alias in aliases
        if alias
    ]
    pairs.sort(key=lambda pair: -len(pair[1]))
    consumed: list[tuple[int, int]] = []
    found: set[str] = set()
    for skill, alias in pairs:
        if skill in found:
            continue
        for match in re.finditer(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text):
            start, end = match.span()
            if any(start < tail and end > head for head, tail in consumed):
                continue
            consumed.append((start, end))
            found.add(skill)
            break
    return found


def extract_skills(
    jobs: list[dict],
    fields: tuple[str, ...] = ("Requirements",),
    skill_dict: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Count how many distinct jobs mention each skill.

    Args:
        jobs: List of job dicts (``data/raw/jobsdb_*_jobs.json`` shape).
        fields: Job text fields scanned for skills; defaults to requirements.
        skill_dict: Optional override of the alias dictionary.

    Returns:
        List of ``{"skill", "count", "job_ids"}`` sorted by count descending,
        then alphabetically. Jobs with no skill-bearing text are skipped.
    """
    aliases = skill_dict or load_skill_dict()
    buckets: dict[str, set[str]] = {}
    for job in jobs:
        text = normalize_text(
            " ".join(str(job.get(field) or "") for field in fields)
        )
        if not text.strip():
            continue
        job_id = str(job.get("job_id") or "")
        for skill in _match_skills(text, aliases):
            buckets.setdefault(skill, set()).add(job_id)
    return sorted(
        (
            {"skill": skill, "count": len(job_ids), "job_ids": sorted(job_ids)}
            for skill, job_ids in buckets.items()
        ),
        key=lambda row: (-row["count"], row["skill"]),
    )


def top_companies(jobs: list[dict], top_n: int | None = None) -> list[dict]:
    """Count job postings per company, sorted by count descending."""
    counts: dict[str, int] = {}
    for job in jobs:
        company = str(job.get("company") or "Unknown").strip() or "Unknown"
        counts[company] = counts.get(company, 0) + 1
    rows = sorted(
        ({"company": company, "count": count} for company, count in counts.items()),
        key=lambda row: (-row["count"], row["company"]),
    )
    return rows[:top_n] if top_n else rows


def salary_band(job: dict) -> str:
    """Bucket a job's salary text into a coarse monthly band.

    The lowest number found is used as the anchor. Listings with no parseable
    salary (or an annual figure) fall into a catch-all band rather than being
    dropped, so filtering never silently hides data.
    """
    numbers = [
        int(n.replace(",", "")) for n in _SALARY_NUM.findall(str(job.get("salary") or ""))
    ]
    if not numbers:
        return "\u2014"
    anchor = min(numbers)
    if anchor < 40_000:
        return "< 40K"
    if anchor < 60_000:
        return "40K\u201360K"
    return "\u2265 60K"


def filter_jobs(
    jobs: list[dict],
    locations: list[str] | tuple[str, ...] = (),
    salary_bands: list[str] | tuple[str, ...] = (),
) -> list[dict]:
    """Subset jobs by location substring(s) and/or salary band(s).

    Empty selection means "no filter" for that dimension. Location matching is
    case-insensitive substring containment.
    """
    result = jobs
    if locations:
        wanted = [loc.lower() for loc in locations]
        result = [
            job
            for job in result
            if any(
                loc in str(job.get("working_location") or "").lower()
                for loc in wanted
            )
        ]
    if salary_bands:
        bands = set(salary_bands)
        result = [job for job in result if salary_band(job) in bands]
    return result


def candidate_gap(
    cv_skills: list[str] | None,
    skill_counts: list[dict],
    top_n: int = 10,
    skill_dict: dict[str, list[str]] | None = None,
) -> dict[str, list[dict]]:
    """Split the top market skills into ones the candidate has vs. is missing.

    Args:
        cv_skills: The candidate's ``Hard_Skills`` list.
        skill_counts: Output of :func:`extract_skills` (already sorted).
        top_n: How many of the hottest skills to consider.
        skill_dict: Optional override used to resolve aliases for comparison.

    Returns:
        ``{"missing": [...], "matched": [...]}``, each a sublist of
        ``skill_counts`` preserving its order. ``missing`` are the hot skills
        the candidate does not (per the dictionary) already have.
    """
    aliases = skill_dict or load_skill_dict()
    have: list[str] = [norm for s in (cv_skills or []) if (norm := _norm_token(s))]
    top = skill_counts[:top_n]

    def covered(skill: str) -> bool:
        candidates = [_norm_token(skill)] + [
            _norm_token(alias) for alias in aliases.get(skill, [])
        ]
        for have_norm in have:
            for candidate in candidates:
                if candidate and (candidate in have_norm or have_norm in candidate):
                    return True
        return False

    return {
        "missing": [row for row in top if not covered(row["skill"])],
        "matched": [row for row in top if covered(row["skill"])],
    }
