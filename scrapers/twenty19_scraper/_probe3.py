
import urllib.request
import ssl
import json

# Bypass SSL verification to test
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
}

urls_to_test = [
    "https://twenty19.com/",
    "https://twenty19.com/internships",
    "https://twenty19.com/robots.txt",
    "https://twenty19.com/sitemap.xml",
    "https://twenty19.com/api/internships",
    "https://twenty19.com/internship/search",
    "https://twenty19.com/internship/list",
]

for url in urls_to_test:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            content = resp.read(1000).decode("utf-8", errors="ignore")
            print(f"OK {status} [{content_type[:30]}]: {url}")
            print(f"  Content: {content[:300]}")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {url}")
    except Exception as e:
        print(f"ERROR: {url} -> {type(e).__name__}: {e}")
    print()
