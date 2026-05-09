"""
Copy every CSV from scrape_ufc_stats into this repo's data/ folder (full mirror).

Default source: sibling directory ../scrape_ufc_stats (same parent as UFC Predictor).

After syncing, rebuild API caches (optional but recommended before deploy):
  python scripts/export_feature_cache.py

Usage:
  python scripts/sync_data_from_scraper.py
  python scripts/sync_data_from_scraper.py --source "D:\\path\\to\\scrape_ufc_stats"

Override with env: UFC_SCRAPE_DATA_DIR
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Must be present for src/preprocessor.py combine_dataframes()
REQUIRED_FOR_ML = frozenset(
    {
        "ufc_event_details.csv",
        "ufc_fight_results.csv",
        "ufc_fight_stats.csv",
        "ufc_fighter_tott.csv",
    }
)


def default_scrape_dir(project_root: Path) -> Path:
    env = os.environ.get("UFC_SCRAPE_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (project_root.parent / "scrape_ufc_stats").resolve()


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    data_dir = project_root / "data"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Directory containing scraped CSVs (default: sibling scrape_ufc_stats or UFC_SCRAPE_DATA_DIR)",
    )
    args = parser.parse_args()
    src_dir = (args.source.expanduser().resolve() if args.source else default_scrape_dir(project_root))

    if not src_dir.is_dir():
        print(f"Error: source directory does not exist: {src_dir}", file=sys.stderr)
        return 1

    data_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(src_dir.glob("*.csv"))
    if not csv_paths:
        print(f"Error: no CSV files found in {src_dir}", file=sys.stderr)
        return 1

    copied_names: list[str] = []
    for src in csv_paths:
        dst = data_dir / src.name
        shutil.copy2(src, dst)
        copied_names.append(src.name)

    for name in copied_names:
        print(f"Copied {name} -> {data_dir / name}")

    missing_required = sorted(REQUIRED_FOR_ML - set(copied_names))
    if missing_required:
        print("Warning: ML pipeline expects these files in source (missing):", file=sys.stderr)
        for m in missing_required:
            print(f"  {m}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
