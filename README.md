# UFC Fight Predictor

An end-to-end machine learning system that predicts UFC fight outcomes from historical fight stats, engineered matchup features, and an XGBoost classifier. The pipeline uses **temporal splits** (train on the past, test on future fights) so metrics reflect realistic forecasting. On the held-out test window, training currently reports about **76% test accuracy** and **~0.84 ROC-AUC**. Run `python src/train.py` from `src/` to print the exact figures—they change as data and `src/tuning.py` change.

**Live demo:** [Deployed on Render](https://ufc-matchup-predictor.onrender.com/)

## Features

- Fight outcome predictions with win probabilities (REST + simple web UI)
- **Order-invariant inference:** swapping which fighter is entered first does not change each fighter’s win probability (see `src/predict.py`)
- Optional **Parquet cache** for fast API cold starts when `data/ufc_preprocessed.parquet` and `data/ufc_features.parquet` are present
- Scripts to sync CSVs from a local `scrape_ufc_stats` repo and rebuild that cache

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn  
- **ML:** XGBoost, scikit-learn, pandas, numpy, joblib  
- **Frontend:** HTML, CSS, JavaScript (static assets served by FastAPI)  
- **I/O:** CSV ingestion; **pyarrow** for optional Parquet feature cache  
- **Deployment:** Render (or any host that can run a long-lived Python process)

## ML pipeline

### 1. Exploratory data analysis (`eda.py`)

Explores fight outcomes, methods, weight classes, fighter attributes, and missingness to guide preprocessing and features.

### 2. Data preprocessing (`preprocessor.py`)

- Merges UFC CSVs in `data/` (events, results, per-fight stats, fighter tale-of-the-tape).
- Parses heights, weights, reach, time, percentages, and strike fractions.
- Imputes missing physical fields using weight-class–aware rules where possible.

### 3. Feature engineering (`src/features/`)

Features are built in a fixed order (`features/__init__.py`).

| Module | Role |
|--------|------|
| `basic.py` | Age, physical diffs, stance matchup, title-fight flag, days since last fight, **layoff difference**, career length proxies |
| `historical.py` | Shifted win rates (last **3**, **5**, and **10** fights), rolling volume/accuracy stats, finish rates, **strikes absorbed** and **striking differential**, **opponent quality** (recent opponents’ win-rate proxy) |
| `title_fights.py` | Title-fight history and champion flags |
| `ratios.py` | **Capped** fighter1/fighter2 ratios; **difference** features (e.g. win-rate diff); **sig. strike %** and **takedown %** diffs |
| `momentum.py` | Career win rate, momentum, streaks, **career win-rate difference** |
| `interactions.py` | Reach × striking, age × experience, size/power interactions |
| `consistency.py` | Rolling variance / consistency metrics |
| `encoding.py` | Binary target; one-hot for referee, weight class, stance matchup |

**Leakage control:** rolling and “last *n* fights” stats use **`shift(1)`** (and analogous patterns) so only **prior** bouts inform the current row.

### 4. Model (`model.py`, `tuning.py`)

- **XGBoost** binary classifier; hyperparameters live in `tuning.py` (e.g. `n_estimators`, `max_depth`, `learning_rate`, subsampling, regularization).
- **`scale_pos_weight`** adjusts sensitivity to class imbalance for the label “fighter1 won” (see comments in `tuning.py`). Tune with `train.py` and your preferred validation/test tradeoff.
- **`train.py`:** temporal split, optional **early stopping** when a validation set is passed to `fit()`.
- **`trainFinal.py`:** fits on **all** labeled rows for deployment; final fit runs **without** early stopping (full `n_estimators` trees). Writes `models/ufc_model_final.pkl`.

### 5. Temporal splitting (`split_data.py`, dates in `tuning.py`)

- Training / validation / test are cut by **`DATE`** (`VALIDATION_SET_DATE`, `TEST_SET_DATE`), not random rows—so evaluation mimics predicting **future** fights from the past.

### 6. Inference (`predict.py`, `fighters.py`)

- Loads `models/ufc_model_final.pkl` and builds a feature row from each fighter’s **latest** historical appearance plus shared context.
- **Order invariance:** two internal forward passes (A→B and B→A) are averaged so UI order does not change each name’s win probability.

## Project structure

```
UFC Predictor/
├── data/                      # CSVs + optional Parquet cache (see Scripts)
├── frontend/                  # index.html, style.css, script.js
├── models/                    # ufc_model_final.pkl (production), ufc_model.pkl (dev train output)
├── scripts/
│   ├── sync_data_from_scraper.py   # Mirror CSVs from sibling ../scrape_ufc_stats
│   └── export_feature_cache.py    # Build ufc_preprocessed.parquet + ufc_features.parquet
├── src/
│   ├── backend/               # api.py, run_api.py
│   ├── features/              # Feature modules (see table above)
│   ├── preprocessor.py
│   ├── fighters.py            # Data caches; prefers Parquet when both files exist
│   ├── predict.py
│   ├── model.py
│   ├── train.py, trainFinal.py, split_data.py, tuning.py, listOfFeatures.py, eda.py
│   └── ...
└── requirements.txt
```

## Quick start (local)

**Prerequisites:** Python 3.10+ recommended (match your venv).

```bash
pip install -r requirements.txt
cd src
python backend/run_api.py
```

Open **http://localhost:8000** (FastAPI serves the UI and `/fighters`, `/predict`).

### Environment variables

| Variable | Effect |
|----------|--------|
| `UFC_SCRAPE_DATA_DIR` | Override source folder for `scripts/sync_data_from_scraper.py` |
| `UFC_DISABLE_PARQUET_CACHE` | If `1` / `true` / `yes`, forces full CSV recompute in `fighters.py` even if Parquet files exist |

## Scripts and data refresh

1. **Sync CSVs** from your local `scrape_ufc_stats` clone (default: sibling folder next to this repo):

   ```bash
   python scripts/sync_data_from_scraper.py
   ```

2. **Rebuild Parquet cache** (speeds up API startup; run after CSV or feature-code changes):

   ```bash
   python scripts/export_feature_cache.py
   ```

3. **Retrain** (from `src/`):

   ```bash
   python train.py          # temporal metrics + saves models/ufc_model.pkl
   python trainFinal.py     # all data → models/ufc_model_final.pkl
   ```

Training always rebuilds features from CSVs / code paths used in `split_data` and `trainFinal`; the Parquet files are for **serving** only.

## API

**`GET /fighters`** — list of fighter names for the UI.

**`POST /predict`** — JSON body `{"fighter1": "...", "fighter2": "..."}`. Response includes win probabilities and predicted winner; probabilities are **stable** if you swap `fighter1` / `fighter2`.

## Deployment notes (Render)

- Use a **non-sleeping** instance or accept **cold starts** on free tiers (first request after idle may be slow while data/model load).
- Commit **Parquet** artifacts if you want fast boots without running `export_feature_cache.py` on every deploy; ensure **pyarrow** is installed in the build environment.
- Point a **custom domain** at Render if you want a branded URL.

## License and credits

Data sourced from publicly available UFC statistics. Built for educational purposes. External scraping workflows may live in a separate `scrape_ufc_stats` repository; respect its **license** if you vendor or redistribute that code.
