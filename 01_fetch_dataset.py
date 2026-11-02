"""
Step 1 - Build the dataset: fetch 100 vessel images from the internet.

Source : Wikimedia Commons (free API, no key needed, images carry open licences)
Classes: 5 vessel types x 20 images = 100 images
Output : dataset/<class_name>/*.jpg   +   dataset/attribution.csv (author + licence per image)

Run    : python 01_fetch_dataset.py
"""
import os, re, time, csv, requests
from PIL import Image
from io import BytesIO

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "VesselCapstone/1.0 (student project; contact: your-email@example.com)"}
OUT_DIR = "dataset"
PER_CLASS = 20

# class name -> list of search phrases (more than one so we always reach 20 usable images)
CLASSES = {
    "container_ship": ["container ship at sea", "container ship port", "container vessel"],
    "oil_tanker":     ["oil tanker ship", "crude oil tanker at sea", "tanker vessel underway"],
    "cruise_ship":    ["cruise ship", "cruise liner at port", "passenger cruise ship"],
    "sailboat":       ["sailboat", "sailing yacht at sea", "sailing boat regatta"],
    "fishing_boat":   ["fishing boat", "fishing trawler", "fishing vessel harbour"],
}

def search_images(query, limit=50):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrnamespace": 6, "gsrsearch": f"{query} filetype:bitmap",
        "gsrlimit": limit, "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata", "iiurlwidth": 640,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for p in pages.values():
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") != "image/jpeg":
            continue
        meta = info.get("extmetadata", {})
        yield {
            "title": p["title"],
            "url": info.get("thumburl") or info.get("url"),
            "author": re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "unknown")),
            "licence": meta.get("LicenseShortName", {}).get("value", "unknown"),
            "page": info.get("descriptionurl", ""),
        }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for cls, queries in CLASSES.items():
        os.makedirs(f"{OUT_DIR}/{cls}", exist_ok=True)
        seen, saved = set(), 0
        for q in queries:
            if saved >= PER_CLASS:
                break
            for item in search_images(q):
                if saved >= PER_CLASS:
                    break
                if item["url"] in seen:
                    continue
                seen.add(item["url"])
                try:
                    resp = requests.get(item["url"], headers=HEADERS, timeout=30)
                    img = Image.open(BytesIO(resp.content)).convert("RGB")
                    if min(img.size) < 200:          # skip tiny images
                        continue
                    fname = f"{cls}_{saved+1:02d}.jpg"
                    img.save(f"{OUT_DIR}/{cls}/{fname}", quality=90)
                    rows.append({"class": cls, "file": fname, **item})
                    saved += 1
                    print(f"[{cls}] {saved}/{PER_CLASS}  {item['title']}")
                    time.sleep(0.5)                  # be polite to the server
                except Exception as e:
                    print("  skipped:", item["title"], "-", e)
        print(f"== {cls}: {saved} images saved ==")
    with open(f"{OUT_DIR}/attribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "file", "title", "author", "licence", "page", "url"])
        w.writeheader(); w.writerows(rows)
    print(f"\nDone. Total images: {len(rows)}")

if __name__ == "__main__":
    main()
