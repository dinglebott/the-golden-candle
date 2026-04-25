# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running things

**Virtual environment:** `.venvg/` (not `.venv`)

**Fetch historical data** (run from repo root):
```bash
python fetch_data.py
```

**Training pipeline** (run from experiment subdirectory, e.g. `1_double-binary/PatchTST/`):
```bash
python select_features.py     # SHAP/permutation importance analysis
python tune_params.py         # Optuna hyperparameter search
python train_model.py         # Train and evaluate, saves model to models/
python use_model.py           # Live inference from terminal (XGBoost only)
```

**API server** (run from `dist/`):
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Docker** (from `dist/`):
```bash
docker build -t golden-candle .
docker run -p 8000:8000 --env-file ../.env golden-candle
```

There are no automated tests or linters configured.

## Code style

Only comment genuinely complex logic (e.g. non-obvious math, tricky index arithmetic). Don't comment what the code already says clearly through naming. Never remove existing comments when editing code.

## Architecture

### Overall structure
The repo is split into shared infrastructure, numbered experiment folders, and a deployment bundle:
- `data_processing/` — shared library (fetching + feature engineering), used by all experiments via `sys.path.insert`
- `<N>_<name>/` — one folder per experiment, each self-contained with its own `env.json`, model configs, trained models, and results
- `dist/` — self-contained deployment bundle; has its own copies of `patchtst.py` and `data_processing.py` that must be kept in sync with the source

### Data flow
1. `fetch_data.py` pulls OHLCV candles from OANDA and saves them to `raw_data/` as JSON files (one per instrument/granularity)
2. `dataparser.parseData()` unpacks raw JSON into a DataFrame and engineers ~40 features (returns, ATR, Bollinger, EMAs, ultSmoother variants, RSI, MACD, ADX, Williams %R, time-of-day sin/cos, directional features)
3. `dataparser.parseCorrelated()` / `parseLiveCorrelated()` computes 4 cross-pair divergence features (close return, return spread, rolling correlation, cross z-score) for an optional correlated pair
4. `addGateTarget()` / `addDirectionTarget()` apply triple-barrier labelling (k × ATR14 barriers, n-candle time barrier)

### Experiment 1: double-binary (`1_double-binary/`)
Two sequential binary models: gate (flat vs directional) then direction (up vs down). Each model type has its own subfolder (`XGBoost/`, `PatchTST/`, `CNN-LSTM/`). Config lives in `env.json` at the experiment root.

**Key `env.json` fields:**
- `corr_pair` — set to `0` (integer) to disable correlated features, or a string like `"GBP_USD"` to enable them
- `binary` — `0` for gate task, `1` for direction task
- `train_version` / `use_version` — nested under `"xgb"` and `"patchtst"` keys respectively

**PatchTST model** (`classes.py`):
- Encoder-only transformer (no decoder)
- Two-stage training: pretrain on multiple pairs (MSE reconstruction loss on close returns), then finetune on the target pair (binary cross-entropy)
- Architecture: shared encoder blocks → correlated pair adapter (additive, per-patch) → task-specific encoder blocks → mean-pool → MLP head
- `freeze_for_finetune()` freezes shared blocks and optionally unfreezes the last N during finetune
- Checkpoints (`.pt`) bundle everything needed for inference: `config`, `core_features`, `corr_features`, `normalization` (mean/std for core and corr separately), `model_state_dict`

### Deployment (`dist/`)
FastAPI server exposing `/predict` and `/candle`. Models are loaded once at startup via `lifespan`. Artifact paths are hardcoded in `main.py` and `inference.py` — update the version integers there when deploying a new model. Trained models must be manually copied from `1_double-binary/*/models/` into `dist/artifacts/gate/` or `dist/artifacts/dir/`.

**Critical:** `dist/api/data_processing.py` is a copy of `data_processing/dataparser.py` (plus live `getData()`). When features are added to `dataparser.py`, they must be manually synced to `dist/api/data_processing.py`.
