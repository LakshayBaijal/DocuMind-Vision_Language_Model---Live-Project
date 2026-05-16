import random
import re
from pathlib import Path
import httpx

api = "https://api.github.com/repos/PaddlePaddle/PaddleOCR/git/trees/release/2.7?recursive=1"
output_dir = Path(r"D:\Projects\Visual Reader\Benchmark\data\images")
output_dir.mkdir(parents=True, exist_ok=True)

with httpx.Client(timeout=90.0, headers={"User-Agent": "DocuMindBenchmarkBot/1.0"}) as client:
    tree = client.get(api)
    tree.raise_for_status()
    payload = tree.json()
    rows = payload.get("tree", [])

    candidates = []
    for row in rows:
        if row.get("type") != "blob":
            continue
        path = str(row.get("path") or "")
        if not re.search(r"\.(jpg|jpeg|png)$", path, re.IGNORECASE):
            continue
        if not (
            path.startswith("doc/imgs/")
            or path.startswith("doc/imgs_en/")
            or path.startswith("doc/imgs_results/")
        ):
            continue
        if any(skip in path.lower() for skip in ["model_prod_flow", "wandb", "logo", "icon"]):
            continue
        candidates.append(path)

    random.seed(42)
    random.shuffle(candidates)
    chosen = candidates[:30]

    downloaded = 0
    for idx, rel_path in enumerate(chosen, start=1):
        raw = f"https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/{rel_path}"
        resp = client.get(raw, follow_redirects=True)
        if resp.status_code != 200:
            continue
        base = Path(rel_path).name
        target = output_dir / f"paddleocr_{idx:02d}_{base}"
        target.write_bytes(resp.content)
        downloaded += 1
        print(f"Saved {target.name}")

print(f"Downloaded {downloaded} files")
