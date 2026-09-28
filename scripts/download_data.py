"""Download the public QuantiPhy datasets from Hugging Face into data/.

Both datasets are public, non-gated, CC-BY-4.0. Revisions are pinned so that
reruns fetch exactly the same files. A manifest with file counts and sizes is
written to results/data_manifest.json.

Usage:
    uv run python scripts/download_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Pinned dataset revisions (commit sha on the Hub), recorded 2026-09-23.
DATASETS = {
    "PaulineLi/QuantiPhy": {
        "revision": "a640cf78a9ac07b17a270a63372f6f65ac95b8a9",
        "local_dir": DATA / "QuantiPhy",
    },
    "PaulineLi/QuantiPhy-validation": {
        "revision": "74aec82473912ccab6a589645a39fcd0e78ee23c",
        "local_dir": DATA / "QuantiPhy-validation",
    },
}

# Official submission template (public, linked from the competition page).
TEMPLATE_URL = "https://quantiphy.stanford.edu/competition/eval/quantiphy_submission_template.csv"
TEMPLATE_PATH = DATA / "submission_template" / "quantiphy_submission_template.csv"
# sha256 of the template as downloaded on 2026-09-23 (the 2026-09-14 corrected version).
TEMPLATE_SHA256_KNOWN = "5f1e8367308ebaee6718929d74a175fecd6813a8d5862fc72a709d07d2328c18"


def dir_stats(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts]
    by_ext: dict[str, int] = {}
    for p in files:
        by_ext[p.suffix or p.name] = by_ext.get(p.suffix or p.name, 0) + 1
    return {
        "n_files": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "files_by_extension": dict(sorted(by_ext.items())),
    }


def main() -> int:
    api = HfApi()
    manifest: dict = {"datasets": {}}
    for repo_id, cfg in DATASETS.items():
        info = api.dataset_info(repo_id, revision=cfg["revision"])
        if info.gated not in (False, None) or info.private:
            print(f"STOP: {repo_id} is gated/private ({info.gated=}, {info.private=})")
            return 1
        print(f"Downloading {repo_id}@{cfg['revision'][:10]} -> {cfg['local_dir']}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=cfg["revision"],
            local_dir=cfg["local_dir"],
        )
        stats = dir_stats(cfg["local_dir"])
        manifest["datasets"][repo_id] = {
            "revision": cfg["revision"],
            "local_dir": cfg["local_dir"].relative_to(ROOT).as_posix(),
            "license": [t for t in (info.tags or []) if t.startswith("license:")],
            "hub_file_count": len(info.siblings or []),
            **stats,
        }
        print(f"  files={stats['n_files']} bytes={stats['total_bytes']:,}")

    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading submission template -> {TEMPLATE_PATH}")
    urllib.request.urlretrieve(TEMPLATE_URL, TEMPLATE_PATH)
    sha = hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()
    if sha != TEMPLATE_SHA256_KNOWN:
        print(
            "WARNING: the official submission template changed since 2026-09-23 "
            f"(sha256 {sha[:12]} != {TEMPLATE_SHA256_KNOWN[:12]}). Check the competition page for notices."
        )
    manifest["submission_template"] = {
        "url": TEMPLATE_URL,
        "local_path": TEMPLATE_PATH.relative_to(ROOT).as_posix(),
        "bytes": TEMPLATE_PATH.stat().st_size,
        "sha256": sha,
        "matches_known_2026_09_23": sha == TEMPLATE_SHA256_KNOWN,
    }

    out = ROOT / "results" / "data_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
