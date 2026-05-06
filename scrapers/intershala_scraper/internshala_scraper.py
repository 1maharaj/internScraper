"""
Internshala Scraper - Selenium-driven, structure-independent
=============================================================
Strategy:
  1. Load the page and scroll to trigger lazy loading
  2. Find ALL <a> tags whose href contains '/internship/detail/'
  3. For each unique link, scroll it into view and read its closest
     self-contained parent (the one with exactly 1 internship link)
  4. Parse fields from plain text - no class/id assumptions
  5. Paginate across multiple pages

This approach survives HTML/CSS restructuring because it only relies on:
  - URL pattern: /internship/detail/
  - Plain text content (stipend ₹, duration months/weeks, etc.)
"""

import re
import json
import time
import logging
from datetime import datetime, date
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)

# ── spaCy (optional) ─────────────────────────────────────────────────────────
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_ENABLED = True
    print("spaCy loaded successfully")
except Exception as e:
    print(f"spaCy not available: {e}")
    SPACY_ENABLED = False

# ── webdriver-manager ────────────────────────────────────────────────────────
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WDM = True
except ImportError:
    USE_WDM = False

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("iFind")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL = "https://internshala.com/internships/"
PAGES    = 40         # Scrape 40 pages (~1600 internships)
MAX_INTERNSHIPS = None  # No cap
TODAY    = date.today()

MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
}

SKILLS_LIST = [
    "python","java","c++","javascript","react","node","html","css",
    "sql","excel","machine learning","data analysis","figma","photoshop",
    "seo","angular","vue","django","flask","mongodb","mysql","aws",
    "azure","docker","git","tableau","power bi","kotlin","swift",
]

ROLE_MAP = {
    "Web Development":  ["web","frontend","backend","react","node","html","css","javascript","php","wordpress"],
    "Data Science":     ["machine learning","ml","ai","data analysis","analytics","pandas","numpy","data science"],
    "Marketing":        ["marketing","seo","social media","campaign","branding","digital marketing"],
    "Business":         ["sales","business","lead","client","crm","strategy","bd","business development"],
    "Design":           ["design","ui","ux","figma","photoshop","graphic","creative"],
    "Content":          ["content","writing","blog","copy","editor","copywriting"],
    "E-Commerce":       ["ecommerce","e-commerce","listing","catalog","amazon","flipkart"],
    "Customer Support": ["support","calling","customer","telecaller","bpo"],
    "Finance":          ["finance","accounting","equity","investment","audit","ca","tax"],
    "HR":               ["hr","human resources","recruitment","talent","hiring"],
}

# Patterns that indicate a promotional/fake card - not a real internship
FAKE_PATTERNS = [
    r"get internship.*training.*free",
    r"by enrolling in trainings",
    r"use coupon",
    r"offer ends in",
    r"^\s*offer\s*$",
]


# ============================================================================
#  DRIVER
# ============================================================================
def setup_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    if USE_WDM:
        svc = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=svc, options=opts)
    else:
        driver = webdriver.Chrome(options=opts)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    driver.implicitly_wait(0)
    log.info("Driver ready (headless=%s)", headless)
    return driver


# ============================================================================
#  HELPERS
# ============================================================================
def get_text(el, driver) -> str:
    """Get element text via JS - works even for off-screen/lazy elements."""
    try:
        t = driver.execute_script(
            "return arguments[0].innerText || arguments[0].textContent || '';", el
        )
        return (t or "").strip()
    except Exception:
        try:
            return el.text.strip()
        except Exception:
            return ""


def is_fake(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in FAKE_PATTERNS)


def clean_company(name: str) -> str:
    name = name.split("\n")[0]
    name = re.sub(r"\b(Actively\s*hiring|Actively|Hiring)\b", "", name, flags=re.I)
    return name.strip()


def extract_skills(text: str) -> list:
    lower = text.lower()
    return sorted({s.capitalize() for s in SKILLS_LIST
                   if re.search(rf"\b{re.escape(s)}\b", lower)})


def classify_role(title: str, text: str) -> str:
    combined = (title + " " + text).lower()
    scores = {}
    for cat, kws in ROLE_MAP.items():
        scores[cat] = sum(3 if kw in title.lower() else 1
                          for kw in kws if kw in combined)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


# ============================================================================
#  PARSE INTERNSHIP FROM CARD TEXT
# ============================================================================
def parse_card(card_el, href: str, driver) -> Optional[dict]:
    """
    Given a card element and its internship URL, extract all fields.
    Entirely text-based - no class/id assumptions.
    """
    text = get_text(card_el, driver)

    if not text or len(text) < 30:
        return None

    if is_fake(text):
        return None

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # ── TITLE ────────────────────────────────────────────────────────────────
    # Try to get title from the link text first (most reliable)
    title = "N/A"
    try:
        a = card_el.find_element(By.XPATH, ".//a[contains(@href,'/internship/detail/')]")
        title = get_text(a, driver)
    except Exception:
        pass

    if not title or title == "N/A":
        # Fall back to first non-metadata line
        skip = {"₹", "apply by", "/month", "lump sum", "work from home",
                "actively", "month", "week", "hiring"}
        for line in lines[:5]:
            if (3 < len(line) < 120
                    and not any(s in line.lower() for s in skip)):
                title = line
                break

    if not title or title == "N/A":
        return None

    # ── LINK ─────────────────────────────────────────────────────────────────
    link = href if href.startswith("http") else "https://internshala.com" + href

    # ── spaCy NER ────────────────────────────────────────────────────────────
    doc = None
    if SPACY_ENABLED:
        try:
            doc = nlp(text[:1000])
        except Exception:
            pass

    # ── COMPANY ──────────────────────────────────────────────────────────────
    company = "N/A"
    if doc:
        for ent in doc.ents:
            if ent.label_ == "ORG":
                c = clean_company(ent.text)
                if len(c) > 2:
                    company = c
                    break

    if company == "N/A":
        bad = {"₹", "/month", "apply", "work from home", "month",
               "week", "lump sum", "actively", "internship"}
        for line in lines[1:7]:
            if (line != title and 1 < len(line) < 100
                    and not any(b in line.lower() for b in bad)):
                c = clean_company(line)
                if len(c) > 1:
                    company = c
                    break

    # ── LOCATION ─────────────────────────────────────────────────────────────
    lower = text.lower()
    if "work from home" in lower:
        location = "Work From Home"
    else:
        location = "N/A"
        # Try spaCy GPE
        if doc:
            for ent in doc.ents:
                if ent.label_ == "GPE":
                    g = ent.text.strip()
                    if not re.search(r"₹|\d|month|week|strategy|excel", g.lower()):
                        location = g
                        break
        # Try text lines
        if location == "N/A":
            bad_loc = {"₹", "month", "apply", "internship",
                       "strategy", "excel", "analysis", "marketing"}
            for line in lines:
                if ("," in line and len(line) < 60
                        and not any(b in line.lower() for b in bad_loc)):
                    location = line
                    break
        # Fallback: extract city from URL slug (e.g. "internship-in-bangalore-at-...")
        if location == "N/A":
            city_match = re.search(r"/internship(?:-in-([a-z\-]+))?-at-", href)
            if city_match and city_match.group(1):
                city_slug = city_match.group(1).replace("-", " ").title()
                location = city_slug
            elif "work-from-home" in href:
                location = "Work From Home"

    # ── STIPEND ──────────────────────────────────────────────────────────────
    m = re.search(
        r"₹\s*[\d,]+(?:\s*[-–]\s*[\d,]+)?\s*(?:/\s*month|per month|lump sum)",
        text, re.IGNORECASE
    )
    stipend = m.group().strip() if m else "N/A"
    if stipend == "N/A":
        for line in lines:
            if any(kw in line.lower() for kw in ["unpaid", "performance based", "performance-based"]):
                stipend = line.strip()
                break

    # ── DURATION ─────────────────────────────────────────────────────────────
    m = re.search(r"\d+(?:\s*[-–]\s*\d+)?\s+(?:Months?|Weeks?)", text, re.IGNORECASE)
    duration = m.group().strip() if m else "N/A"

    # ── APPLY BY ─────────────────────────────────────────────────────────────
    apply_by = "N/A"
    m = re.search(
        r"Apply\s+[Bb]y\s+(\d{1,2}\s+[A-Za-z]{3}(?:'?\s*\d{2,4})?)",
        text, re.IGNORECASE
    )
    if m:
        apply_by = m.group(1).strip()

    # ── DESCRIPTION ──────────────────────────────────────────────────────────
    description = "N/A"
    skip_desc = {"₹", "apply by", "/month", "lump sum", "work from home"}
    for line in lines:
        if (len(line) > 40 and line != title and line != company
                and not any(s in line.lower() for s in skip_desc)):
            description = line
            break

    # ── SKILLS & ROLE ────────────────────────────────────────────────────────
    skills = extract_skills(text)
    role   = classify_role(title, text)

    return {
        "title":       title,
        "company":     company,
        "location":    location,
        "stipend":     stipend,
        "duration":    duration,
        "apply_by":    apply_by,
        "skills":      skills,
        "type":        role,
        "link":        link,
        "description": description,
    }


# ============================================================================
#  SCRAPE ONE PAGE
# ============================================================================
def scrape_page(driver: webdriver.Chrome, page_num: int) -> list:
    url = BASE_URL if page_num == 1 else f"{BASE_URL}page-{page_num}/"
    log.info("── Page %d → %s", page_num, url)
    driver.get(url)

    # Wait for internship links to appear
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, "//a[contains(@href,'/internship/detail/')]")
            )
        )
    except TimeoutException:
        log.warning("Page %d: timed out", page_num)
        return []

    # Scroll fully to trigger lazy loading
    log.info("Scrolling page to trigger lazy loading...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(20):
        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(0.3)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)

    # ── Find all unique internship links ─────────────────────────────────────
    all_links = driver.find_elements(
        By.XPATH, "//a[contains(@href,'/internship/detail/')]"
    )
    log.info("Found %d raw internship links on page %d", len(all_links), page_num)

    # Deduplicate by internship ID extracted from URL
    seen_ids = {}   # id -> (href, link_element)
    for a in all_links:
        try:
            href = a.get_attribute("href") or ""
            m = re.search(r"(\d{7,})", href)
            if m and m.group(1) not in seen_ids:
                seen_ids[m.group(1)] = (href, a)
        except StaleElementReferenceException:
            continue

    log.info("Unique internship IDs on page %d: %d", page_num, len(seen_ids))

    results = []
    for idx, (iid, (href, link_el)) in enumerate(seen_ids.items(), 1):
        try:
            # Scroll link into view so lazy content renders
            driver.execute_script(
                "arguments[0].scrollIntoView({block:'center',behavior:'instant'});",
                link_el
            )
            time.sleep(0.35)

            # ── Find the card container using pure Selenium DOM walking ───────
            # Strategy: walk UP from the link until we find a node whose text
            # contains the key internship fields (stipend ₹ OR duration months/weeks)
            # AND contains only ONE internship detail link (so it's a single card).
            # This works regardless of class names or HTML structure.
            card_el = None
            current = link_el
            prev = link_el

            for level in range(12):
                try:
                    parent = current.find_element(By.XPATH, "..")
                except Exception:
                    break

                tag = ""
                try:
                    tag = parent.tag_name.lower()
                except Exception:
                    pass

                # Stop if we've gone too far up
                if tag in ("body", "html", "main", "section"):
                    break

                # Count internship links inside this parent
                try:
                    n_links = len(parent.find_elements(
                        By.XPATH, ".//a[contains(@href,'/internship/detail/')]"
                    ))
                except Exception:
                    n_links = 0

                # Get text via JS (works for lazy/hidden elements)
                text = get_text(parent, driver)

                # A good card container has:
                # - exactly 1 internship link (single card, not a list)
                # - meaningful content (>80 chars)
                # - at least one of: stipend, duration, location indicator
                has_content = (
                    len(text) > 80 and
                    any(indicator in text.lower() for indicator in
                        ["₹", "month", "week", "work from home", "apply"])
                )

                if n_links == 1 and has_content:
                    card_el = parent
                    # Keep going up a bit more to capture full card
                    # (sometimes stipend is in a sibling div at a higher level)
                    prev = parent
                    current = parent
                    continue

                if n_links > 1:
                    # We've gone past the card into a list container
                    # Use the last good level we found
                    card_el = prev if prev != link_el else None
                    break

                current = parent

            # If we couldn't find a good card, use the best we have
            if card_el is None:
                card_el = link_el.find_element(By.XPATH, "..")

            full_href = href if href.startswith("http") else "https://internshala.com" + href
            item = parse_card(card_el, full_href, driver)
            if item:
                results.append(item)
                log.info("  [%d/%d] ✓ %s | %s | %s | %s",
                         idx, len(seen_ids),
                         item["title"][:40],
                         item["company"][:25],
                         item["stipend"],
                         item["location"])
            else:
                log.debug("  [%d/%d] ✗ skipped", idx, len(seen_ids))

        except StaleElementReferenceException:
            log.debug("  [%d/%d] stale, skipping", idx, len(seen_ids))
            continue
        except Exception as e:
            log.debug("  [%d/%d] error: %s", idx, len(seen_ids), e)
            continue

    log.info("Page %d done: %d extracted from %d unique links",
             page_num, len(results), len(seen_ids))
    return results


# ============================================================================
#  DETECT TOTAL PAGES
# ============================================================================
def detect_total_pages(driver: webdriver.Chrome) -> int:
    """
    Load the first page and detect how many pages exist.
    Tries multiple strategies:
      1. Parse total internship count from page text and divide by per-page count
      2. Find the last page number in pagination links
      3. Fall back to a large default so we paginate until empty pages
    """
    try:
        driver.get(BASE_URL)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, "//a[contains(@href,'/internship/detail/')]")
            )
        )
        time.sleep(2)

        page_source = driver.page_source

        # Strategy 1: find "X internships" count in page text
        m = re.search(r"([\d,]+)\s+(?:Total\s+)?[Ii]nternships?", page_source)
        if m:
            total = int(m.group(1).replace(",", ""))
            # Internshala shows ~16 per page but we get ~40 links (including duplicates)
            # Use 16 as conservative per-page count
            pages = max(1, (total + 15) // 16)
            log.info("Detected %d total internships → %d pages", total, pages)
            return pages

        # Strategy 2: find highest page number in pagination
        page_nums = re.findall(r"/internships/page-(\d+)/", page_source)
        if page_nums:
            max_page = max(int(p) for p in page_nums)
            # Pagination usually shows a window, so multiply by safety factor
            pages = max_page * 3
            log.info("Detected max pagination page %d → estimating %d pages", max_page, pages)
            return pages

    except Exception as e:
        log.warning("Could not detect total pages: %s", e)

    # Fallback: use a large number; scraper stops when pages return empty
    log.info("Using fallback: will scrape until empty pages")
    return 9999



def parse_apply_by(raw: str) -> Optional[date]:
    if not raw or raw == "N/A":
        return None
    raw = re.sub(r"'(\d{2})\b", r" 20\1", raw.strip())
    for fmt in ("%d %b %Y", "%b %d %Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    parts = raw.split()
    try:
        if len(parts) == 3:
            m = MONTH_MAP.get(parts[1].lower()[:3])
            if m:
                return date(int(parts[2]), m, int(parts[0]))
        elif len(parts) == 2:
            m = MONTH_MAP.get(parts[0].lower()[:3])
            if m:
                return date(int(parts[1]), m, 28)
    except Exception:
        pass
    return None


def filter_active(items: list) -> list:
    out = []
    for item in items:
        d = parse_apply_by(item.get("apply_by", ""))
        if d is None:
            item["_date_unresolved"] = True
            out.append(item)
        elif d >= TODAY:
            out.append(item)
    return out


# ============================================================================
#  DEDUPLICATE BY INTERNSHIP ID
# ============================================================================
def deduplicate(items: list) -> list:
    seen, out = set(), []
    for item in items:
        m = re.search(r"(\d{7,})", item.get("link", ""))
        key = m.group(1) if m else item.get("link", "")
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ============================================================================
#  MAIN
# ============================================================================
def _save_progress(items: list):
    """Save current raw items to disk so progress isn't lost on interruption."""
    try:
        with open("internships_raw.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(items, indent=2, ensure_ascii=False))
        log.info("Progress saved: %d raw items → internships_raw.json", len(items))
    except Exception as e:
        log.warning("Could not save progress: %s", e)


def main():
    driver = setup_driver(headless=True)
    all_items: list = []

    try:
        # Auto-detect total pages if PAGES is None
        if PAGES is None:
            total_pages = detect_total_pages(driver)
        else:
            total_pages = PAGES

        log.info("Will scrape up to %d pages", total_pages)
        consecutive_empty = 0  # Stop after 3 consecutive empty pages

        for page in range(1, total_pages + 1):
            page_data = scrape_page(driver, page)

            if not page_data:
                consecutive_empty += 1
                log.warning("Page %d returned 0 results (%d consecutive empty)",
                            page, consecutive_empty)
                if consecutive_empty >= 3:
                    log.info("3 consecutive empty pages — stopping pagination.")
                    break
            else:
                consecutive_empty = 0
                all_items.extend(page_data)
                log.info("Cumulative raw: %d (page %d/%d)", len(all_items), page, total_pages)

                # Save progress incrementally every 5 pages
                if page % 5 == 0:
                    _save_progress(all_items)

            # Stop if we've hit the cap
            if MAX_INTERNSHIPS and len(all_items) >= MAX_INTERNSHIPS:
                log.info("Reached MAX_INTERNSHIPS cap (%d) — stopping.", MAX_INTERNSHIPS)
                break

            time.sleep(1.5)  # Polite pause between pages

    finally:
        driver.quit()
        log.info("Browser closed.")

    # ── Save raw data BEFORE any processing ─────────────────────────────────
    raw_out = json.dumps(all_items, indent=2, ensure_ascii=False)
    with open("internships_raw.json", "w", encoding="utf-8") as fh:
        fh.write(raw_out)
    log.info("Raw data saved → internships_raw.json (%d entries)", len(all_items))

    log.info("=== POST-PROCESSING ===")
    unique = deduplicate(all_items)
    log.info("After dedup:  %d", len(unique))

    active = filter_active(unique)
    log.info("Active:       %d", len(active))

    # Quality report
    fields = ["company", "location", "stipend", "duration", "apply_by", "description"]
    log.info("=== DATA COMPLETENESS ===")
    for f in fields:
        if active:
            pct = sum(1 for i in active if i.get(f, "N/A") not in ("N/A", "")) / len(active) * 100
            log.info("  %-15s %.0f%%", f, pct)

    # Save
    out_str = json.dumps(active, indent=2, ensure_ascii=False)
    with open("internships.json", "w", encoding="utf-8") as fh:
        fh.write(out_str)
    log.info("Saved %d internships → internships.json", len(active))

    if len(active) < 200:
        log.warning("⚠️  Only %d internships scraped — expected >200. Try increasing PAGES.", len(active))
    else:
        log.info("✅  Scraped %d internships total", len(active))

    print(f"\nTotal scraped: {len(active)}")
    return active


if __name__ == "__main__":
    main()
