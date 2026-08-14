"""JobsDB crawler: scrape analyst-programmer job listings into a JSON file."""

import json
import random
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from job_seeker.config import RAW_DIR, settings

__all__ = [
    "build_page_url",
    "extract_job_id",
    "human_delay",
    "safe_find_text",
    "extract_section_from_detail",
    "extract_detail_fields",
    "normalize_job_url",
    "load_existing_jobs",
    "upsert_jobs",
    "crawl_jobs",
]


def build_page_url(page: int) -> str:
    params = {"sortmode": "KeywordRelevance"}
    if page > 1:
        params["page"] = page
    return f"{settings.jobsdb_base_url}?{urlencode(params)}"


def extract_job_id(job_url: str, card_job_id: str) -> str:
    if card_job_id:
        return card_job_id.strip()
    path_match = re.search(r"/job/(\d+)", job_url)
    if path_match:
        return path_match.group(1)
    parsed = urlparse(job_url)
    query_id = parse_qs(parsed.query).get("jobId") or parse_qs(parsed.query).get("adId")
    return query_id[0] if query_id else ""


def human_delay() -> None:
    time.sleep(random.uniform(2, 5))


def safe_find_text(parent, selectors) -> str:
    for selector in selectors:
        try:
            value = parent.find_element(By.CSS_SELECTOR, selector).text.strip()
            if value:
                return value
        except NoSuchElementException:
            continue
    return ""


def extract_section_from_detail(driver, keyword: str) -> str:
    keyword_lower = keyword.lower()
    for _ in range(2):
        labels = driver.find_elements(
            By.XPATH,
            "//*[self::h1 or self::h2 or self::h3 or self::h4 or self::strong or self::b or self::p]",
        )
        for label in labels:
            try:
                text = label.text.strip()
                if not text or keyword_lower not in text.lower():
                    continue
                parent_text = label.find_element(By.XPATH, "..").text.strip()
                if not parent_text:
                    continue
                cleaned = parent_text.replace(text, "", 1).strip()
                if cleaned:
                    return cleaned
                sibling_candidates = label.find_elements(
                    By.XPATH,
                    "following-sibling::*[self::ul or self::ol or self::p or self::div][1]",
                )
                if sibling_candidates:
                    return sibling_candidates[0].text.strip()
            except StaleElementReferenceException:
                continue
    return ""


def extract_detail_fields(driver, job_url: str) -> tuple[str, str]:
    responsibilities = ""
    requirements = ""
    original = driver.current_window_handle
    human_delay()
    driver.execute_script("window.open(arguments[0], '_blank');", job_url)
    detail_handle = driver.window_handles[-1]
    driver.switch_to.window(detail_handle)
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        responsibilities = extract_section_from_detail(driver, "responsibilit")
        requirements = extract_section_from_detail(driver, "requirement")
    except TimeoutException:
        responsibilities = ""
        requirements = ""
    finally:
        driver.close()
        driver.switch_to.window(original)
    return responsibilities, requirements


def normalize_job_url(job_url: str) -> str:
    return urlparse(job_url)._replace(fragment="").geturl()


def load_existing_jobs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as json_file:
            data = json.load(json_file)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def upsert_jobs(existing: list[dict], fresh: list[dict]) -> list[dict]:
    existing = list(existing)
    by_id = {job.get("job_id"): job for job in existing if job.get("job_id")}
    by_url = {
        normalize_job_url(job.get("job_url", "")): job
        for job in existing
        if job.get("job_url")
    }
    for job in fresh:
        job_id = job.get("job_id")
        url_key = normalize_job_url(job.get("job_url", ""))
        target = None
        if job_id and job_id in by_id:
            target = by_id[job_id]
        elif url_key and url_key in by_url:
            target = by_url[url_key]
        if target is not None:
            target.update(job)
        else:
            existing.append(job)
            if job_id:
                by_id[job_id] = job
            if url_key:
                by_url[url_key] = job
    return existing


def crawl_jobs() -> None:
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 20)
    jobs_data = []
    seen_urls = set()
    page = 1

    try:
        while len(jobs_data) < settings.jobsdb_target_count:
            if page > 1:
                human_delay()
            driver.get(build_page_url(page))
            try:
                wait.until(EC.presence_of_all_elements_located((By.XPATH, "//article")))
            except TimeoutException:
                break

            cards = driver.find_elements(By.XPATH, "//article")
            if not cards:
                break

            for card in cards:
                if len(jobs_data) >= settings.jobsdb_target_count:
                    break
                try:
                    job_link = card.find_element(By.CSS_SELECTOR, 'a[data-automation="jobTitle"]')
                except NoSuchElementException:
                    continue

                job_url = job_link.get_attribute("href") or ""
                job_url_key = normalize_job_url(job_url)
                if not job_url_key or job_url_key in seen_urls:
                    continue

                seen_urls.add(job_url_key)
                company = safe_find_text(card, ['a[data-automation="jobCompany"]', 'span[data-automation="jobCompany"]'])
                salary = safe_find_text(card, ['span[data-automation="jobSalary"]', '[data-automation="jobSalary"]'])
                location = safe_find_text(card, ['span[data-automation="jobLocation"]', '[data-automation="jobLocation"]'])
                job_id = extract_job_id(job_url, card.get_attribute("data-job-id") or "")
                responsibilities, requirements = extract_detail_fields(driver, job_url)

                jobs_data.append(
                    {
                        "job_id": job_id,
                        "job_url": job_url,
                        "company": company,
                        "salary": salary,
                        "working_location": location,
                        "Responsibilities": responsibilities,
                        "Requirements": requirements,
                    }
                )
            page += 1
    finally:
        driver.quit()

    settings.jobsdb_output_file.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_jobs(settings.jobsdb_output_file)
    merged = upsert_jobs(existing, jobs_data)
    with open(settings.jobsdb_output_file, "w", encoding="utf-8") as json_file:
        json.dump(merged, json_file, ensure_ascii=False, indent=2)
    print(
        f"Data saved successfully! new={len(jobs_data)} total={len(merged)} "
        f"file={settings.jobsdb_output_file}"
    )
