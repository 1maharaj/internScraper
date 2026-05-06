
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

opts = Options()
# Try WITHOUT headless to bypass bot detection
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
svc = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=svc, options=opts)
driver.execute_cdp_cmd(
    "Page.addScriptToEvaluateOnNewDocument",
    {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
)

for url in [
    "https://www.twenty19.com/",
    "https://www.twenty19.com/internships",
    "https://www.twenty19.com/internship",
]:
    driver.get(url)
    time.sleep(5)
    title = driver.title
    current = driver.current_url
    body = driver.find_element(By.TAG_NAME, "body").text
    print(f"URL: {url}")
    print(f"  Title: {title[:70]}")
    print(f"  Current: {current}")
    print(f"  Body (first 400): {body[:400]}")
    links = driver.find_elements(By.TAG_NAME, "a")
    print(f"  Links: {len(links)}")
    for a in links[:15]:
        h = a.get_attribute("href") or ""
        t = a.text.strip()[:30]
        if h:
            print(f"    [{t}] -> {h}")
    print()

driver.quit()
