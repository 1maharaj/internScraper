# Naukri.com Job Scraper

Selenium-driven scraper for Naukri.com job listings. Structure-independent — survives HTML/CSS changes.

## Setup

```bash
pip install selenium webdriver-manager spacy
python -m spacy download en_core_web_sm
```

## Usage

Edit the config at the top of `naukri_scraper.py`:

```python
SEARCH_KEYWORD = "software developer"   # Job title / keyword
SEARCH_LOCATION = "bangalore"           # City (leave empty for all India)
PAGES = 10                              # Pages to scrape (~20 jobs/page)
```

Then run:

```bash
python naukri_scraper.py
```

## Output

| File | Description |
|------|-------------|
| `jobs_raw.json` | Raw data before deduplication (saved every 5 pages) |
| `jobs.json` | Final deduplicated output |

## Fields Extracted

| Field | Description |
|-------|-------------|
| `title` | Job title |
| `company` | Company name |
| `location` | City / Remote |
| `salary` | Salary range (e.g. "5-10 LPA") |
| `experience` | Experience required (e.g. "2-5 Yrs") |
| `job_type` | Full Time / Part Time / Contract / Internship |
| `posted` | When posted (e.g. "2 days ago") |
| `skills` | Detected skills from description |
| `type` | Role category (Software Development, Data Science, etc.) |
| `link` | Direct URL to job listing |
| `description` | Job description snippet |

## Notes

- Naukri uses bot detection. If you get 0 results, try `headless=False` in `setup_driver()`.
- Progress is saved every 5 pages so data isn't lost on interruption.
- The scraper uses DOM traversal (not CSS selectors) so it works even after site redesigns.
