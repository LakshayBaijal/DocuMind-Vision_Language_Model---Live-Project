from __future__ import annotations

import argparse
import random
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import httpx

API_URL = "https://commons.wikimedia.org/w/api.php"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

CATEGORIES = [
    "Category:Scanned_documents",
    "Category:Identity_documents",
    "Category:Passports",
    "Category:Government_documents",
]


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "document"


def fetch_category_files(client: httpx.Client, category: str, limit: int = 100) -> List[Tuple[str, str, str]]:
    results: List[Tuple[str, str, str]] = []
    gcmcontinue = None
    remaining = max(limit, 1)

    while remaining > 0:
        chunk = min(remaining, 500)
        params = {
            "action": "query",
            "format": "json",
            "generator": "categorymembers",
            "gcmtitle": category,
            "gcmtype": "file",
            "gcmlimit": str(chunk),
            "prop": "imageinfo",
            "iiprop": "url|mime",
        }
        if gcmcontinue:
            params["gcmcontinue"] = gcmcontinue

        response = client.get(API_URL, params=params)
        response.raise_for_status()
        payload = response.json()

        pages = (payload.get("query") or {}).get("pages") or {}
        for page in pages.values():
            title = str(page.get("title") or "")
            if not title.startswith("File:"):
                continue
            imageinfo = page.get("imageinfo") or []
            if not imageinfo:
                continue
            info = imageinfo[0]
            url = str(info.get("url") or "")
            mime = str(info.get("mime") or "")
            if not url or not mime.startswith("image/"):
                continue
            filename = title.replace("File:", "", 1)
            ext = Path(filename).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            results.append((filename, url, mime))

        remaining -= chunk
        gcmcontinue = ((payload.get("continue") or {}).get("gcmcontinue"))
        if not gcmcontinue:
            break

    return results


def fetch_subcategories(client: httpx.Client, category: str, limit: int = 80) -> List[str]:
    subcategories: List[str] = []
    cmcontinue = None
    remaining = max(limit, 1)

    while remaining > 0:
        chunk = min(remaining, 500)
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "subcat",
            "cmlimit": str(chunk),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = client.get(API_URL, params=params)
        response.raise_for_status()
        payload = response.json()

        members = (payload.get("query") or {}).get("categorymembers") or []
        for member in members:
            title = str(member.get("title") or "")
            if title.startswith("Category:"):
                subcategories.append(title)

        remaining -= chunk
        cmcontinue = ((payload.get("continue") or {}).get("cmcontinue"))
        if not cmcontinue:
            break

    return subcategories


def build_category_pool(client: httpx.Client, roots: List[str], depth: int) -> List[str]:
    visited: Set[str] = set()
    queue = deque((root, 0) for root in roots)
    categories: List[str] = []

    while queue:
        category, level = queue.popleft()
        if category in visited:
            continue
        visited.add(category)
        categories.append(category)

        if level >= depth:
            continue

        try:
            children = fetch_subcategories(client, category, limit=120)
        except Exception:
            children = []

        for child in children:
            if child not in visited:
                queue.append((child, level + 1))

    return categories


def download_file(client: httpx.Client, url: str, target: Path) -> bool:
    response = client.get(url, follow_redirects=True)
    if response.status_code != 200:
        return False
    target.write_bytes(response.content)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download random scanned document images from Wikimedia Commons.")
    parser.add_argument("--count", type=int, default=30, help="Number of documents to download")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "data" / "images"),
        help="Directory to save images",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--category-depth", type=int, default=2, help="How deep to traverse subcategories")
    parser.add_argument("--per-category-file-limit", type=int, default=120, help="Max files fetched per category")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    with httpx.Client(timeout=90.0, headers={"User-Agent": "DocuMindBenchmarkBot/1.0"}) as client:
        pool: Dict[str, Tuple[str, str]] = {}

        categories = build_category_pool(client, CATEGORIES, depth=max(0, args.category_depth))
        print(f"Discovered {len(categories)} Wikimedia categories to scan")

        for category in categories:
            try:
                for filename, url, _mime in fetch_category_files(
                    client,
                    category,
                    limit=max(1, args.per_category_file_limit),
                ):
                    pool[filename] = (filename, url)
            except Exception:
                continue

        if not pool:
            print("No candidate documents found from Wikimedia categories.")
            return 2

        candidates = list(pool.values())
        random.shuffle(candidates)

        downloaded = 0
        for original_name, url in candidates:
            if downloaded >= args.count:
                break

            safe_name = sanitize_filename(original_name)
            target = output_dir / safe_name
            if target.exists():
                stem = target.stem
                suffix = target.suffix
                target = output_dir / f"{stem}_{downloaded + 1}{suffix}"

            ok = download_file(client, url, target)
            if not ok:
                continue
            downloaded += 1
            print(f"Saved {target.name}")

        print(f"Downloaded {downloaded} document image(s) to {output_dir}")
        if downloaded < args.count:
            print(f"Requested {args.count}; only {downloaded} could be downloaded from available categories.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
