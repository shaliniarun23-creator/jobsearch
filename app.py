import os
import re
import requests
import pandas as pd
import streamlit as st

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

# Works locally with .env and on Streamlit Cloud with Secrets
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
GOOGLE_CSE_ID = st.secrets.get("GOOGLE_CSE_ID", os.getenv("GOOGLE_CSE_ID"))


TARGET_ROLES = [
    "founders office",
    "founder's office",
    "regional medical officer",
    "medical science liaison",
    "medical liaison",
    "medical officer",
    "hospital coordinator",
    "medical coordinator",
    "chief of staff",
    "consultant",
    "business analyst",
    "analyst",
    "project manager",
    "product manager",
    "strategy associate",
    "strategy manager",
    "strategy analyst",
]

TARGET_LOCATIONS = [
    "India",
    "Singapore",
    "UAE",
    "Dubai",
    "Chennai",
    "Bengaluru",
    "Hyderabad",
    "Mumbai",
    "Gurugram",
    "Remote",
]

TRUSTED_JOB_SITES = [
    "linkedin.com/jobs",
    "indeed.com",
    "naukri.com",
    "foundit.in",
    "iimjobs.com",
    "wellfound.com",
    "greenhouse.io",
    "lever.co",
    "workdayjobs.com",
    "myworkdayjobs.com",
]

PROFILE_KEYWORDS = """
BDS Global MBA healthcare life sciences clinical operations medical affairs
medical science liaison hospital coordination medical coordinator stakeholder management
strategy consulting business analyst project management product management chief of staff
founders office founder office healthcare strategy digital transformation smart hospitals
market research primary research expert interviews Excel Power BI Tableau dashboards
operations revenue growth patient journey pharma medical communication
"""

REJECT_KEYWORDS = [
    "intern",
    "internship",
    "software engineer",
    "developer",
    "nurse",
    "mbbs mandatory",
    "md mandatory",
    "phd mandatory",
    "ca mandatory",
    "10+ years",
    "12+ years",
    "15+ years",
]


def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def lower_text(text):
    return clean_text(text).lower()


def extract_source(url):
    for site in TRUSTED_JOB_SITES:
        if site in url:
            return site
    return "Other"


def extract_company_from_title(title):
    title = clean_text(title)
    separators = [" - ", " | ", " at "]

    for sep in separators:
        if sep in title:
            parts = title.split(sep)
            if len(parts) > 1:
                return clean_text(parts[-1])

    return ""


def google_search_jobs(role, location, results_per_query=10):
    query = (
        f'"{role}" "{location}" jobs '
        f'(site:linkedin.com/jobs OR site:indeed.com OR site:naukri.com OR '
        f'site:greenhouse.io OR site:lever.co OR site:workdayjobs.com OR site:myworkdayjobs.com)'
    )

    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(results_per_query, 10),
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        jobs = []

        for item in data.get("items", []):
            job_url = item.get("link", "")
            title = clean_text(item.get("title", ""))
            snippet = clean_text(item.get("snippet", ""))

            if any(site in job_url for site in TRUSTED_JOB_SITES):
                jobs.append(
                    {
                        "Role Search": role,
                        "Location Search": location,
                        "Job Title": title,
                        "Company": extract_company_from_title(title),
                        "Job URL": job_url,
                        "Source": extract_source(job_url),
                        "Snippet": snippet,
                    }
                )

        return jobs

    except Exception as e:
        st.warning(f"Search failed for {role} - {location}: {e}")
        return []


def fetch_job_page_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=20)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        text = clean_text(soup.get_text(" "))
        return text[:15000]

    except Exception:
        return ""


def similarity_score(a, b):
    a = lower_text(a)
    b = lower_text(b)

    if not a or not b:
        return 0

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([a, b])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return round(score * 100, 2)
    except Exception:
        return 0


def role_fit_score(job_title, jd_text):
    combined = lower_text(f"{job_title} {jd_text}")
    matched_roles = [role for role in TARGET_ROLES if role in combined]

    if not matched_roles:
        return 30

    return min(100, 50 + len(matched_roles) * 12)


def location_score(location_search, jd_text):
    combined = lower_text(f"{location_search} {jd_text}")

    for loc in TARGET_LOCATIONS:
        if loc.lower() in combined:
            return 100

    return 50


def extract_years_required(text):
    text = lower_text(text)

    patterns = [
        r"(\d+)\+?\s*years",
        r"(\d+)\s*-\s*(\d+)\s*years",
        r"minimum\s*(\d+)\s*years",
    ]

    years = []

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for match in matches:
            if isinstance(match, tuple):
                for number in match:
                    if str(number).isdigit():
                        years.append(int(number))
            else:
                if str(match).isdigit():
                    years.append(int(match))

    return min(years) if years else None


def experience_score(years_required):
    if years_required is None:
        return 70

    if years_required <= 3:
        return 85

    if 4 <= years_required <= 7:
        return 100

    if 8 <= years_required <= 9:
        return 60

    return 20


def reject_job(job_title, jd_text):
    combined = lower_text(f"{job_title} {jd_text}")

    for keyword in REJECT_KEYWORDS:
        if keyword in combined:
            return True

    return False


def live_job_score(text):
    text = lower_text(text)

    live_signals = [
        "apply",
        "apply now",
        "submit application",
        "posted",
        "actively hiring",
        "reposted",
        "new",
    ]

    closed_signals = [
        "no longer accepting",
        "job expired",
        "position closed",
        "applications closed",
        "not accepting applications",
    ]

    if any(signal in text for signal in closed_signals):
        return 0

    if any(signal in text for signal in live_signals):
        return 100

    return 60


def calculate_final_score(job):
    title = job.get("Job Title", "")
    snippet = job.get("Snippet", "")
    jd_text = job.get("Job Description", "")
    location = job.get("Location Search", "")

    combined = f"{title} {snippet} {jd_text}"

    relevance = similarity_score(PROFILE_KEYWORDS, combined)
    role_score = role_fit_score(title, jd_text)
    loc_score = location_score(location, jd_text)
    years_required = extract_years_required(combined)
    exp_score = experience_score(years_required)
    live_score = live_job_score(combined)
    rejected = reject_job(title, jd_text)

    final_score = (
        relevance * 0.30
        + role_score * 0.25
        + loc_score * 0.15
        + exp_score * 0.15
        + live_score * 0.15
    )

    if rejected:
        final_score -= 40

    final_score = max(min(round(final_score, 2), 100), 0)

    return {
        "Relevance Score": relevance,
        "Role Fit Score": role_score,
        "Location Score": loc_score,
        "Experience Score": exp_score,
        "Live Job Score": live_score,
        "Years Required": years_required,
        "Reject Flag": rejected,
        "Final Score": final_score,
    }


st.set_page_config(page_title="Job Hunt Live Job Finder", layout="wide")

st.title("Job Hunt Live Job Finder + Ranking Dashboard")

st.warning(
    "This app uses Google Search API to discover public job pages. "
    "It does not directly scrape protected LinkedIn or Indeed search pages."
)

if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    st.error(
        "Missing GOOGLE_API_KEY or GOOGLE_CSE_ID. "
        "For Streamlit Cloud, add them in Manage app → Settings → Secrets."
    )
    st.stop()

with st.sidebar:
    st.header("Search Controls")

    selected_roles = st.multiselect(
        "Select target roles",
        TARGET_ROLES,
        default=[
            "medical science liaison",
            "founders office",
            "chief of staff",
            "consultant",
            "business analyst",
            "project manager",
        ],
    )

    selected_locations = st.multiselect(
        "Select locations",
        TARGET_LOCATIONS,
        default=["India", "Singapore", "UAE", "Dubai"],
    )

    results_per_query = st.slider("Results per role-location search", 1, 10, 5)

    run_search = st.button("Find Live Jobs")


if run_search:
    all_jobs = []

    with st.spinner("Searching live jobs..."):
        for role in selected_roles:
            for location in selected_locations:
                jobs = google_search_jobs(role, location, results_per_query)
                all_jobs.extend(jobs)

    if not all_jobs:
        st.error("No jobs found. Try broader roles or locations.")
        st.stop()

    unique_jobs = {job["Job URL"]: job for job in all_jobs}
    jobs = list(unique_jobs.values())

    st.info(f"Found {len(jobs)} unique job links. Checking job pages...")

    scored_jobs = []
    progress = st.progress(0)

    for index, job in enumerate(jobs):
        jd_text = fetch_job_page_text(job["Job URL"])
        job["Job Description"] = jd_text

        scores = calculate_final_score(job)
        job.update(scores)

        scored_jobs.append(job)
        progress.progress((index + 1) / len(jobs))

    df = pd.DataFrame(scored_jobs)
    df = df.sort_values("Final Score", ascending=False)

    st.success("Job ranking completed.")

    display_cols = [
        "Final Score",
        "Job Title",
        "Company",
        "Source",
        "Location Search",
        "Years Required",
        "Relevance Score",
        "Role Fit Score",
        "Location Score",
        "Experience Score",
        "Live Job Score",
        "Reject Flag",
        "Job URL",
    ]

    st.subheader("Top Ranked Jobs")
    st.dataframe(df[[col for col in display_cols if col in df.columns]], use_container_width=True)

    st.subheader("Best Jobs to Apply First")

    top_apply = df[(df["Reject Flag"] == False) & (df["Final Score"] >= 70)].head(20)

    st.dataframe(
        top_apply[[col for col in display_cols if col in top_apply.columns]],
        use_container_width=True,
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Ranked Jobs CSV",
        data=csv,
        file_name="ranked_live_jobs.csv",
        mime="text/csv",
    )

else:
    st.info("Select roles and locations, then click **Find Live Jobs**.")
