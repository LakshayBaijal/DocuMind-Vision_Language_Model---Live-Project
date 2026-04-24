"""
POST a local image to the running API (default http://127.0.0.1:8000/extract).

Usage:
  python test_extract.py path/to/id_or_pan.jpg
  python test_extract.py --url http://127.0.0.1:8000/extract card.png
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Test /extract with a local image.")
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/extract",
        help="Full URL of the extract endpoint",
    )
    args = parser.parse_args()
    path: Path = args.image
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/jpeg"
    with path.open("rb") as f:
        files = {"file": (path.name, f, mime)}
        r = httpx.post(args.url, files=files, timeout=120.0)

    print(f"HTTP {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
    return 0 if r.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
