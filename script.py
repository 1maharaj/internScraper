"""
script.py — Master scraper orchestrator

Runs all scrapers sequentially, accumulates results in memory,
checkpoints to a JSON file after each scraper (so nothing is lost
if execution is interrupted), then pushes everything through the
moderation pipeline into MongoDB.

Usage (standalone):
    python script.py [--skip-scrape] [--scrapers github,internshala,indeed,naukri,unstop,freshersworld,letsintern]

Flags:
    --skip-scrape   Skip running scrapers; push existing checkpoint file to DB
    --scrapers      Comma-separated list of scrapers to run (default: all)
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRAPER_DIR = Path(__file__).parent
SCRAPERS_DIR = SCRAPER_DIR / "scrapers"
CHECKPOINT_FILE = SCRAPER_DIR / "checkpoint_internships.json"

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("script")

# ─── Scraper registry ─────────────────────────────────────────────────────────
# Each entry: id, label, import path (relative to SCRAPERS_DIR), main function name
SCRAPER_REGISTRY = [
    # {
    #     "id":       "github",
    #     "label":    "GitHub (2026 SWE Jobs)",
    #     "module":   "scrapers.github_scraper",
    #     "fn":       "scrape_github_internships",
    #     "source":   "web_scraping",
    # },
    {
        "id":       "internshala",
        "label":    "Internshala",
        "module":   "scrapers.internshala_scraper",
        "fn":       "scrape_internshala_internships",
        "source":   "web_scraping",
    },
    # {
    #     "id":       "indeed",
    #     "label":    "Indeed",
    #     "module":   "scrapers.indeed_scraper",
    #     "fn":       "scrape_indeed_internships",
    #     "source":   "web_scraping",
    # },
    # {
    #     "id":       "naukri",
    #     "label":    "Naukri",
    #     "module":   "scrapers.naukri_scraper.naukri_scraper",
    #     "fn":       "main",          # naukri main() returns list
    #     "source":   "web_scraping",
    # },
    # {
    #     "id":       "unstop",
    #     "label":    "Unstop",
    #     "module":   "scrapers.unstop_scraper.unstop_scraper",
    #     "fn":       "main",          # unstop main() returns list
    #     "source":   "web_scraping",
    # },
    # {
    #     "id":       "freshersworld",
    #     "label":    "Freshersworld",
    #     "module":   "scrapers.freshersworld_scraper.freshersworld_scraper",
    #     "fn":       "main",          # freshersworld main() returns list
    #     "source":   "web_scraping",
    # },
    # {
    #     "id":       "letsintern",
    #     "label":    "LetsIntern",
    #     "module":   "scrapers.letsintern_scraper.letsintern_scraper",
    #     "fn":       "main",          # letsintern main() returns list
    #     "source":   "web_scraping",
    # },
]


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint() -> list:
    """Load previously scraped internships from the checkpoint file."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info("📂 Loaded %d internships from checkpoint: %s", len(data), CHECKPOINT_FILE)
            return data
        except Exception as e:
            log.warning("Could not load checkpoint: %s", e)
    return []


def save_checkpoint(internships: list):
    """Save all accumulated internships to the checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(internships, f, indent=2, ensure_ascii=False, default=str)
        log.info("💾 Checkpoint saved: %d internships → %s", len(internships), CHECKPOINT_FILE)
    except Exception as e:
        log.warning("Could not save checkpoint: %s", e)


# ─── Scraper runner ───────────────────────────────────────────────────────────

def run_scraper(scraper: dict) -> list:
    """
    Dynamically import and run a scraper module.
    Returns a list of raw internship dicts.
    """
    # Add SCRAPER_DIR to sys.path so relative imports work
    scraper_dir_str = str(SCRAPER_DIR)
    if scraper_dir_str not in sys.path:
        sys.path.insert(0, scraper_dir_str)

    try:
        import importlib
        mod = importlib.import_module(scraper["module"])
        fn  = getattr(mod, scraper["fn"])
        result = fn()
        if result is None:
            return []
        if isinstance(result, list):
            return result
        log.warning("  Scraper '%s' returned unexpected type: %s", scraper["id"], type(result))
        return []
    except Exception as e:
        log.error("  ❌ Scraper '%s' failed: %s", scraper["id"], e)
        return []


# ─── Main orchestrator ────────────────────────────────────────────────────────

def run_all_scrapers(selected_ids: list[str]) -> list:
    """
    Run all selected scrapers sequentially.
    Accumulates results in memory AND checkpoints to file after each scraper.
    Returns the full list of raw internship dicts.
    """
    # Start from existing checkpoint (resume if interrupted)
    all_internships = load_checkpoint()
    already_scraped_count = len(all_internships)

    active = [s for s in SCRAPER_REGISTRY if s["id"] in selected_ids]

    log.info("\n╔══════════════════════════════════════════════════╗")
    log.info("║         iFind Scraper Orchestrator               ║")
    log.info("╚══════════════════════════════════════════════════╝")
    log.info("Scrapers: %s", ", ".join(s["label"] for s in active))
    log.info("Resuming from checkpoint: %d existing internships\n", already_scraped_count)

    for scraper in active:
        log.info("━━━ %s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", scraper["label"])
        items = run_scraper(scraper)

        if not items:
            log.warning("  No internships returned from %s", scraper["label"])
            continue

        # Tag each item with its source
        for item in items:
            item["_scraper_source"] = scraper["source"]
            item["_scraper_id"]     = scraper["id"]

        log.info("  ✓ %s returned %d internships", scraper["label"], len(items))
        all_internships.extend(items)

        # Checkpoint after every scraper so partial results survive crashes
        save_checkpoint(all_internships)

    log.info("\n📊 Total scraped: %d internships", len(all_internships))
    return all_internships


# ─── DB push ──────────────────────────────────────────────────────────────────

def push_all_to_db(internships: list) -> dict:
    """
    Push all scraped internships through the moderation pipeline into MongoDB.
    Returns aggregate stats.
    """
    from dotenv import load_dotenv
    import pymongo

    # Load .env from Scraper directory
    env_path = SCRAPER_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # fallback to CWD

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI not set in environment")

    log.info("\n🔌 Connecting to MongoDB...")
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    db  = client["ifind"]
    col = db["internships"]
    log.info("✅ Connected\n")

    from pipeline import push_to_pipeline
    from collections import defaultdict

    totals = {"saved": 0, "duplicate": 0, "rejected": 0, "errors": 0}

    # Group by scraper_id so each batch goes through the pipeline
    # with its own source label and gets logged separately
    by_scraper: dict[str, list] = defaultdict(list)
    for item in internships:
        scraper_id = item.get("_scraper_id", "unknown")
        by_scraper[scraper_id].append(item)

    log.info("━━━ Moderation Pipeline ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for scraper_id, batch in by_scraper.items():
        source = batch[0].get("_scraper_source", "web_scraping")
        log.info("▶ %s (%d items)", scraper_id, len(batch))
        stats = push_to_pipeline(batch, source, col, label=scraper_id)
        log.info(
            "  → saved:%d  dupes:%d  rejected:%d  errors:%d",
            stats["saved"], stats["duplicate"], stats["rejected"], stats["errors"],
        )
        for k in totals:
            totals[k] += stats[k]

    client.close()
    return totals


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main(selected_ids: list[str] | None = None, skip_scrape: bool = False) -> dict:
    """
    Main entry point — can be called from CLI or from app.py.

    Returns aggregate stats dict.
    """
    if selected_ids is None:
        selected_ids = [s["id"] for s in SCRAPER_REGISTRY]

    # Phase 1: Scrape
    if skip_scrape:
        log.info("⏭  --skip-scrape: loading from checkpoint only")
        internships = load_checkpoint()
        if not internships:
            log.warning("No checkpoint found and --skip-scrape was set. Nothing to push.")
            return {"saved": 0, "duplicate": 0, "rejected": 0, "errors": 0}
    else:
        internships = run_all_scrapers(selected_ids)

    if not internships:
        log.warning("No internships to push.")
        return {"saved": 0, "duplicate": 0, "rejected": 0, "errors": 0}

    # Phase 2: Push to DB
    totals = push_all_to_db(internships)

    # Summary
    log.info("\n━━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("✅ Saved to DB:   %d", totals["saved"])
    log.info("⏭  Duplicates:   %d", totals["duplicate"])
    log.info("❌ Rejected:      %d", totals["rejected"])
    log.info("💥 Errors:        %d", totals["errors"])
    log.info("\n🎯 Open the Moderation tab in the dashboard to review.")

    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iFind scraper orchestrator")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping; push existing checkpoint file to DB",
    )
    parser.add_argument(
        "--scrapers",
        type=str,
        default=",".join(s["id"] for s in SCRAPER_REGISTRY),
        help="Comma-separated list of scrapers to run",
    )
    args = parser.parse_args()

    selected = [s.strip() for s in args.scrapers.split(",") if s.strip()]
    main(selected_ids=selected, skip_scrape=args.skip_scrape)
