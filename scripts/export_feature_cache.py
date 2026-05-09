"""
Build Parquet caches for fast API startup (skip full preprocess + feature pipeline on boot).

Run after CSVs are up to date, e.g.:
  1. python scripts/sync_data_from_scraper.py   # optional: refresh data/*.csv
  2. python scripts/export_feature_cache.py   # writes data/ufc_preprocessed.parquet + ufc_features.parquet

The API loads these when both files exist (see src/fighters.py). Disable with env UFC_DISABLE_PARQUET_CACHE=1.

Usage (from repo root):
  python scripts/export_feature_cache.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

PREPROCESSED_NAME = "ufc_preprocessed.parquet"
FEATURES_NAME = "ufc_features.parquet"
MANIFEST_NAME = "feature_cache_manifest.json"


def main() -> int:
    sys.path.insert(0, str(SRC_DIR))

    import pandas as pd
    from preprocessor import preprocess_data
    from features import create_features

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pre_path = DATA_DIR / PREPROCESSED_NAME
    feat_path = DATA_DIR / FEATURES_NAME

    t0 = time.perf_counter()
    print("Building preprocessed frame (preprocess_data)...")
    df_pre = preprocess_data()
    if "DATE" in df_pre.columns:
        df_pre["DATE"] = pd.to_datetime(df_pre["DATE"], errors="coerce")
    df_pre.to_parquet(pre_path, index=False)
    print(f"  Wrote {pre_path} ({len(df_pre)} rows, {len(df_pre.columns)} cols)")

    print("Building full feature frame (create_features; may take a while)...")
    df_feat = create_features()
    if "DATE" in df_feat.columns:
        df_feat["DATE"] = pd.to_datetime(df_feat["DATE"], errors="coerce")
    df_feat.to_parquet(feat_path, index=False)
    print(f"  Wrote {feat_path} ({len(df_feat)} rows, {len(df_feat.columns)} cols)")

    # Small manifest for humans / CI (not required for loading)
    manifest = {
        "preprocessed_parquet": PREPROCESSED_NAME,
        "features_parquet": FEATURES_NAME,
        "rows_preprocessed": len(df_pre),
        "rows_features": len(df_feat),
        "seconds_elapsed": round(time.perf_counter() - t0, 2),
    }
    (DATA_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Wrote {DATA_DIR / MANIFEST_NAME}")
    print(f"Done in {manifest['seconds_elapsed']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
