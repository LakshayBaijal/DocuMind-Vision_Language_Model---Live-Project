from __future__ import annotations

import argparse
import mimetypes
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

import httpx

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "images"
DEFAULT_URLS_FILE = Path(__file__).resolve().parent / "data" / "urls.txt"


def sanitize_filename(name: str) -> str:
    name = unquote(name).strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "downloaded_image"


def guess_filename(url: str, content_type: str | None) -> str:
    parsed = urlparse(url)
    candidate = Path(unquote(parsed.path)).name
    if not candidate:
        candidate = "downloaded_image"

    if "." not in candidate:
        ext = None
        if content_type:
            ext = mimetypes.guess_extension(content_type.split(";")[0].strip().lower())
        if not ext or ext == ".jpe":
            ext = ".jpg"
        candidate += ext

    return sanitize_filename(candidate)


def load_urls(urls_file: Path) -> list[str]:
    if not urls_file.is_file():
        return []
    urls: list[str] = []
    for line in urls_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        urls.append(text)
    return urls


def download_one(client: httpx.Client, url: str, output_dir: Path, index: int) -> Path:
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type")
    filename = guess_filename(url, content_type)
    if not Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
        filename += ".jpg"

    output_path = output_dir / filename
    if output_path.exists():
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_dir / f"{stem}_{index}{suffix}"

    output_path.write_bytes(response.content)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download benchmark documents into Benchmark/data/images.")
    parser.add_argument("--urls-file", default=str(DEFAULT_URLS_FILE), help="Text file with one URL per line")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save downloaded images")
    parser.add_argument("urls", nargs="*", help="Optional direct URLs to download")
    args = parser.parse_args()

    urls_file = Path(args.urls_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = list(args.urls) if args.urls else load_urls(urls_file)
    if not urls:
        print(f"No URLs found. Add them to {urls_file} or pass them on the command line.", file=sys.stderr)
        return 2

    with httpx.Client(timeout=120.0) as client:
        for index, url in enumerate(urls, start=1):
            try:
                saved = download_one(client, url, output_dir, index)
                print(f"Saved {saved.name}")
            except Exception as exc:
                print(f"Failed: {url} -> {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
