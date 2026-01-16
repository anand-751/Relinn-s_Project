import argparse
import json
import os
import re
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlparse

from bs4 import BeautifulSoup


# ===============================
# Configuration
# ===============================

OUTPUT_DIR = "scraped_data"
HEADERS = {"User-Agent": "RAG-Web-Scraper/1.0"}
TIMEOUT = 15
MAX_PAGES = 400


# ===============================
# URL Filtering Rules
# ===============================

EXCLUDED_PATH_KEYWORDS = [
    "/blogs",
    "/blog/",
    "/blog-",
]

ALLOWED_PATH_KEYWORDS = [
    "/chatbot",
    "/platform",
    "/solutions",
    "/ai-agents",
    "/tools",
    "/integrations",
    "/pricing",
    "/security",
    "/partners",
    "/alternatives",
    "/comparisons",
    "/glossary",
]


# ===============================
# Utility Functions
# ===============================

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_filename(url: str) -> str:
    domain = urlparse(url).netloc.replace(".", "_")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{domain}_fullsite_{timestamp}.json"


# ===============================
# URL Validation
# ===============================

def is_valid_content_url(url: str) -> bool:
    path = urlparse(url).path.lower()

    # ❌ Exclude blogs
    for kw in EXCLUDED_PATH_KEYWORDS:
        if kw in path:
            return False

    # ✅ Allow only product-related paths
    for kw in ALLOWED_PATH_KEYWORDS:
        if kw in path:
            return True

    return False


# ===============================
# Sitemap Handling
# ===============================

def get_urls_from_sitemap(base_url: str) -> list[str]:
    sitemap_url = base_url.rstrip("/") + "/sitemap.xml"
    print(f"[INFO] Fetching sitemap: {sitemap_url}")

    try:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Sitemap not accessible: {e}")
        return []

    root = ET.fromstring(resp.text)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls = []
    for loc in root.findall(".//ns:loc", ns):
        urls.append(loc.text.strip())

    print(f"[INFO] Found {len(urls)} URLs in sitemap")
    return urls


# ===============================
# Networking
# ===============================

def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text


# ===============================
# Parsing Logic
# ===============================

def extract_visible_text(soup: BeautifulSoup) -> list[str]:
    texts = []
    for tag in soup.find_all(["p", "li", "section", "article"]):
        text = clean_text(tag.get_text(" ", strip=True))
        if len(text) > 50:
            texts.append(text)
    return texts


def parse_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title = clean_text(soup.title.text) if soup.title else ""

    headings = []
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            txt = clean_text(h.get_text())
            if txt:
                headings.append({"tag": f"h{level}", "text": txt})

    paragraphs = extract_visible_text(soup)

    full_text = "\n".join(paragraphs) if paragraphs else soup.get_text(" ", strip=True)
    full_text = clean_text(full_text)

    return {
        "title": title,
        "headings": headings,
        "paragraphs": paragraphs,
        "full_text": full_text,
        "word_count": len(full_text.split())
    }


# ===============================
# Crawling Logic
# ===============================

def crawl_website(start_url: str) -> dict:
    visited = set()
    pages_data = []

    sitemap_urls = get_urls_from_sitemap(start_url)

    filtered_urls = [
        url for url in sitemap_urls
        if is_valid_content_url(url)
    ]

    print(f"[INFO] URLs after filtering (non-blog): {len(filtered_urls)}")

    for url in filtered_urls:
        if len(visited) >= MAX_PAGES:
            break

        if url in visited:
            continue

        try:
            print(f"[INFO] Crawling: {url}")
            html = fetch_html(url)
        except Exception as e:
            print(f"[WARN] Skipping {url}: {e}")
            continue

        visited.add(url)

        extracted = parse_html(html)
        extracted["page_url"] = url
        pages_data.append(extracted)

        time.sleep(0.3)  # polite crawling

    print(f"[INFO] Crawled {len(visited)} pages")

    return {
        "total_pages": len(visited),
        "pages": pages_data
    }


# ===============================
# Main Execution
# ===============================

def scrape_website(url: str) -> str:
    ensure_output_dir()

    site_data = crawl_website(url)

    output = {
        "source_url": url,
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "site_pages": site_data
    }

    filename = generate_filename(url)
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Full site data saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Full Website Scraper for RAG")
    parser.add_argument("--url", required=True, help="Website URL to scrape")
    args = parser.parse_args()

    scrape_website(args.url)


if __name__ == "__main__":
    main()
