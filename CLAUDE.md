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

### Experiment 2: event-detection (`2_event-detection/`)
Pattern-gated approach: a hardcoded detector first filters candles matching a specific pattern, then a model predicts whether the pattern resolves as expected. The model only outputs a signal when its pattern fires, giving sparser but higher-quality signals.

**Labelling:** Triple-barrier labelling anchored to the pattern. TP is the pattern's expected resolution (e.g. 50% of the FVG). SL is `k × ATR` beyond the 3rd candle's extreme. Time barrier is `n` candles after detection. Label `1` = fill (TP hit first), `0` = no fill.

**Patterns** (`patterns/`):
- `fair_value_gap.py` — 3-candle pattern; gap between candle[i-2] high and candle[i] low (bullish) or vice versa (bearish). Metadata features: `gap_atr_ratio`, `direction`.
- `registry.py` — maps pattern name strings to their detector modules.

**CNN-LSTM model** (`CNN-LSTM/classes.py`):
- Two-input architecture: a sequence input (OHLCV + engineered features over a lookback window) and a metadata input (pattern-specific scalar features injected at detection time).
- Conv1D (2 layers) extracts local patterns → LSTM reads temporal structure → final hidden state concatenated with metadata → MLP head outputs a single logit.
- Checkpoints (`.pt`) store `model_state_dict`, `config`, `seq_features`, `meta_features`, and normalization stats.

**Key `env.json` fields:**
- `pattern` — name of the active pattern (e.g. `"fvg"`); controls which detector and labeller are used
- `k_value` / `n_value` — SL multiplier and time-barrier length
- `train_version` / `use_version` — nested under `"xgb"` and `"cnn_lstm"` keys

**Adding a new pattern:** implement `METADATA_FEATURES`, `detect(df)`, and `label_instances(df, instances, n_candles, k)` in `patterns/<name>.py`, register it in `registry.py`, set `"pattern"` in `env.json`, and add versioned feature/param configs under each model architecture's `model_configs/`.

**Deploying:** copy the trained model and feature list into `dist/artifacts/<pattern_name>/`, add the pattern to `PATTERN_VERSIONS` in `dist/api/inference.py`, add an entry to `PATTERN_REGISTRY` in `dist/api/main.py`, and add a `PATTERN_CONFIGS` entry in `web_interface/js/config.js`.

### Deployment (`dist/`)
FastAPI server exposing `/predict` and `/candle`. Models are loaded once at startup via `lifespan`. Artifact paths are hardcoded in `main.py` and `inference.py` — update the version integers there when deploying a new model. Trained models must be manually copied from `1_double-binary/*/models/` into `dist/artifacts/gate/` or `dist/artifacts/dir/`.

**Critical:** `dist/api/data_processing.py` is a copy of `data_processing/dataparser.py` (plus live `getData()`). When features are added to `dataparser.py`, they must be manually synced to `dist/api/data_processing.py`.
