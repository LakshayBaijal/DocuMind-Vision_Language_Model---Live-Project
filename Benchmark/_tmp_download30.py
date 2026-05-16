import random
import sys
from pathlib import Path
import httpx

sys.path.insert(0, r"D:\Projects\Visual Reader")
from Benchmark.download_random_documents import CATEGORIES, build_category_pool, fetch_category_files, download_file, sanitize_filename

output_dir = Path(r"D:\Projects\Visual Reader\Benchmark\data\images")
output_dir.mkdir(parents=True, exist_ok=True)
random.seed(42)

with httpx.Client(timeout=60.0, headers={"User-Agent": "DocuMindBenchmarkBot/1.0"}) as client:
    categories = build_category_pool(client, CATEGORIES, 1)
    print("categories", len(categories))
    pool = {}
    for category in categories[:40]:
        try:
            rows = fetch_category_files(client, category, limit=30)
            for filename, url, _mime in rows:
                pool[filename] = (filename, url)
        except Exception:
            continue

    candidates = list(pool.values())
    random.shuffle(candidates)

    downloaded = 0
    for original_name, url in candidates:
        if downloaded >= 30:
            break
        safe_name = sanitize_filename(original_name)
        target = output_dir / safe_name
        if target.exists():
            stem, suffix = target.stem, target.suffix
            target = output_dir / f"{stem}_{downloaded+1}{suffix}"
        if download_file(client, url, target):
            downloaded += 1
            print("Saved", target.name)

print("downloaded", downloaded)
