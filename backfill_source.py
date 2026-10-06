"""
backfill_source.py — one-off migration: replace the legacy source "web_scraping" with the real scraper id.

Before, every scraped listing was stored with source "web_scraping". The schema now says `source` is the
scraper (github, internshala, indeed, naukri, unstop, freshersworld, letsintern) or "ifind". The scraper id is
recovered from the apply link's host. Listings whose host is not recognised are left untouched and reported.

  python backfill_source.py            dry run: prints what would change, writes nothing
  python backfill_source.py --apply    performs the updates

Touches collections 'internships' and 'internships.mod-unvectorised'. Run it once; it is safe to re-run.
"""

import os
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pymongo
from dotenv import load_dotenv

HOST_TO_SOURCE = {
    "internshala.com": "internshala",
    "indeed.com": "indeed",
    "naukri.com": "naukri",
    "unstop.com": "unstop",
    "freshersworld.com": "freshersworld",
    "letsintern.com": "letsintern",
    "letsintern.in": "letsintern",
    "github.com": "github",
}
COLLECTIONS = ("internships", "internships.mod-unvectorised")


def source_for(link: str):
    host = (urlparse(link or "").hostname or "").lower()
    for domain, source in HOST_TO_SOURCE.items():
        if host == domain or host.endswith("." + domain):
            return source
    return None


def main() -> None:
    apply = "--apply" in sys.argv
    here = Path(__file__).resolve().parent
    for env in (here / ".env", here / ".env.local", here.parent / "ifind" / ".env.local"):
        if env.exists():
            load_dotenv(env)
    uri = os.getenv("MONGODB_URI")
    if not uri:
        sys.exit("MONGODB_URI is not set")

    db = pymongo.MongoClient(uri, serverSelectionTimeoutMS=10000)[os.getenv("MONGODB_DB_NAME", "ifind")]
    for name in COLLECTIONS:
        col = db[name]
        plan: Counter = Counter()
        unknown: Counter = Counter()
        updates: dict = {}
        for doc in col.find({"source": "web_scraping"}, {"applyLink": 1}):
            src = source_for(doc.get("applyLink"))
            if src:
                plan[src] += 1
                updates.setdefault(src, []).append(doc["_id"])
            else:
                unknown[(urlparse(doc.get("applyLink") or "").hostname or "?")] += 1
        print(f"{name}: would set {dict(plan)}; unrecognised hosts left as web_scraping: {dict(unknown)}")
        if apply:
            for src, ids in updates.items():
                res = col.update_many({"_id": {"$in": ids}}, {"$set": {"source": src}})
                print(f"  {src}: updated {res.modified_count}")
    if not apply:
        print("\nDry run only. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
