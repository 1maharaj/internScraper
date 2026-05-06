
"""
Test if Twenty19 has any accessible API endpoints or alternative URLs
"""
import urllib.request
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
}

urls_to_test = [
    "https://www.twenty19.com/",
    "https://www.twenty19.com/internships",
    "https://api.twenty19.com/internships",
    "https://www.twenty19.com/api/internships",
    "https://www.twenty19.com/api/v1/internships",
    "https://www.twenty19.com/sitemap.xml",
    "https://www.twenty19.com/robots.txt",
]

for url in urls_to_test:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            content = resp.read(500).decode("utf-8", errors="ignore")
            print(f"OK {status}: {url}")
            print(f"  Content: {content[:200]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {url}")
    except Exception as e:
        print(f"ERROR: {url} -> {e}")
    print()
