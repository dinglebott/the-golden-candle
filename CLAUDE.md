# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running things

**Virtual environment:** `.venvg/` (not `.venv`)

**Fetch historical data** (run from repo root):
```bash
python fetch_data.py
```

**Training pipeline** (run from a model subdirectory, e.g. `1_double-binary/PatchTST/` or `2_event-detection/CNN-LSTM/`):
```bash
python select_features.py     # SHAP/permutation importance analysis
python tune_params.py         # Optuna hyperparameter search
python train_model.py         # Train and evaluate, saves model to models/
python use_model.py           # Live inference from terminal (XGBoost only)
```

**Backtesting** (run from `3_algo-backtest/`):
```bash
python backtest.py            # Runs the strategy named in env.json
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
- `dist/` — self-contained deployment bundle; has its own copies of `patchtst.py`, `cnn_lstm.py`, pattern detectors, `symmetry.py`, and `data_processing.py` that must be kept in sync with the source

### Data flow
1. `fetch_data.py` pulls OHLCV candles from OANDA and saves them to `raw_data/` as JSON files (one per instrument/granularity)
2. `dataparser.parseData()` unpacks raw JSON into a DataFrame and engineers ~40 features (returns, ATR, Bollinger, EMAs, ultSmoother variants, RSI, MACD, ADX, Williams %R, time-of-day sin/cos, directional features)
3. `dataparser.parseCorrelated()` / `parseLiveCorrelated()` computes 4 cross-pair divergence features (close return, return spread, rolling correlation, cross z-score) for an optional correlated pair
4. `addGateTarget()` / `addDirectionTarget()` apply triple-barrier labelling (k × ATR14 barriers, n-candle time barrier)

### Experiment 1: double-binary (`1_double-binary/`)
Two sequential binary models: gate (flat vs directional) then direction (up vs down). Each model type has its own subfolder (`XGBoost/`, `PatchTST/`). Config lives in `env.json` at the experiment root.

**Key `env.json` fields:**
- `corr_pair` — set to `0` (integer) to disable correlated features, or a string like `"GBP_USD"` to enable them
- `binary` — `0` for gate task, `1` for direction task
- `train_version` / `use_version` — nested under `"xgb"` and `"patchtst"` keys respectively
- `patchtst.skip_pretrain` — when true, finetune directly without the multi-pair pretrain stage

**PatchTST model** (`classes.py`):
- Encoder-only transformer (no decoder)
- Two-stage training: pretrain on multiple pairs (MSE reconstruction loss on close returns), then finetune on the target pair (binary cross-entropy)
- Architecture: shared encoder blocks → correlated pair adapter (additive, per-patch) → task-specific encoder blocks → mean-pool → MLP head
- `freeze_for_finetune()` freezes shared blocks and optionally unfreezes the last N during finetune
- Checkpoints (`.pt`) bundle everything needed for inference: `config`, `core_features`, `corr_features`, `normalization` (mean/std for core and corr separately), `model_state_dict`

### Experiment 2: event-detection (`2_event-detection/`)
Pattern-gated approach: a hardcoded detector first filters candles matching a specific pattern, then a model predicts whether the pattern resolves as expected. The model only outputs a signal when its pattern fires, giving sparser but higher-quality signals.

**Labelling:** Triple-barrier labelling anchored to the pattern. TP is the pattern's expected resolution. SL is `k × ATR` beyond a pattern-specific reference point. Time barrier is `n` candles after detection. Label `1` = fill (TP hit first), `0` = no fill.

**Patterns** (`patterns/`):
- `fair_value_gap.py` — 3-candle pattern; gap between candle[i-2] high and candle[i] low (bullish) or vice versa (bearish). TP at 50% of the gap, SL 1.5×ATR beyond the 3rd candle's extreme. Metadata: `gap_atr_ratio`, `direction`.
- `order_block.py` — last opposing candle before a strong impulse; detection candle is the first close back inside the OB range within `OB_EXPIRY` bars. TP at close ± 1.0×ATR, SL 0.5×ATR beyond the far side of the OB. Metadata: `impulse_atr_ratio`, `zone_size_atr_ratio`, `candles_elapsed`.
- `registry.py` — maps pattern name strings to their detector modules.

**Symmetry helper** (`symmetry.py`):
Mirrors bearish setups onto bullish geometry so the model sees one symmetry class. Sign-flips directional features (returns, RSI-centred, etc.), swaps pairs (`high_return`/`low_return`, `upper_wick`/`lower_wick`, etc.), and applies `1 - x` to bounded features like `bb_position`. Imported by the training scripts and the deployment inference path; not applied in the dataparser, which keeps raw, direction-agnostic features.

**Model architectures** (`XGBoost/`, `CNN-LSTM/`, `TCN/`):
- **CNN-LSTM** (`CNN-LSTM/classes.py`) — two-input: sequence (OHLCV + engineered features over a lookback window) and metadata (pattern-specific scalars at the detection candle). Conv1D ×2 → LSTM → concat with metadata → MLP head → single logit. Checkpoints store `model_state_dict`, `config`, `seq_features`, `meta_features`, and normalization stats.
- **TCN** (`TCN/classes.py`) — same two-input shape, but uses a stack of dilated causal temporal blocks (Conv1D + chomp + BatchNorm + ReLU + dropout, residual) and reads the last causal timestep before concatenating metadata.

**Model config layout:**
Each architecture's `model_configs/` is split into `current_models/` (versions used in production / `use_version`) and `training_models/` (work-in-progress versions targeted by `train_version`).

**Key `env.json` fields:**
- `pattern` — name of the active pattern (e.g. `"fvg"`, `"order_block"`); controls which detector and labeller are used
- `n_value` — time-barrier length (SL multiplier and other pattern-specific knobs live as module-level constants at the top of each detector in `patterns/`, not in `env.json`)
- `train_version` / `use_version` — nested under `"xgb"`, `"cnn_lstm"`, and `"tcn"` keys

**Adding a new pattern:** implement `METADATA_FEATURES`, `detect(df)`, and `label_instances(df, instances, n_candles)` in `patterns/<name>.py` (SL/TP multipliers live as module-level constants at the top of the detector file), register it in `registry.py`, set `"pattern"` in `env.json`, and add versioned feature/param configs under each model architecture's `model_configs/training_models/`. If the pattern has a +1/-1 direction that's a true geometric mirror, the symmetry helper applies automatically — no per-pattern wiring needed beyond ensuring detector instances carry a `direction` field.

**Deploying:** copy the trained model into `dist/artifacts/` (flat layout, e.g. `fvg_CNN-LSTM_EUR_USD_H1_2026_v3.pt`), add the pattern to `PATTERN_VERSIONS` in `dist/api/inference.py`, add an entry to `PATTERN_REGISTRY` in `dist/api/main.py`, and add a `PATTERN_CONFIGS` entry in `web_interface/js/config.js`. Neural-network checkpoints bundle their own features/config; only XGBoost models need an accompanying features JSON.

### Experiment 3: algo-backtest (`3_algo-backtest/`)
Pure-rule strategies (no ML). Each strategy is its own module exposing `get_entries(df) -> DataFrame` with `entry`, `tp`, `sl`, and optional `tp_type` columns. `backtest.py` is the shared framework: it loads `env.json`, slices into train/test, calls the strategy's `get_entries`, runs triple-barrier evaluation, and prints metrics broken down by direction and TP type plus MFE/MAE forensics.

**Strategies:**
- `london_orb.py` — London opening range breakout. ORB on `RANGE_WINDOW` bars from 08:00 London (DST-aware), trades in the breakout direction within `TRADE_WINDOW` bars. TP tries Asian session high/low, then nearest valid H1 swing high/low, with a range-multiple fallback.
- `trend_pullback.py` — Trend-following pullback entries gated by ADX and RSI, sized off an H1 swing. TP selected from H1 swing structure.

**Key `env.json` fields:**
- `strategy` — module name of the active strategy (e.g. `"london_orb"`, `"trend_pullback"`)
- `n_value` — time-barrier length in candles
- `dataset` — `"train"` (first 80%), `"test"` (last 20%), or anything else for the full series
- `log_results` — when true, appends each run's printout to `backtest_results.log`

`BE_TRIGGER_R` at the top of `backtest.py` enables a break-even SL move once price reaches that R-multiple of favourable excursion (set to `0` to disable). Strategy-specific knobs (SL/TP multiples, filter thresholds, lookbacks) live as module-level constants at the top of each strategy file.

### Deployment (`dist/`)
FastAPI server exposing `/predict` (PatchTST gate), `/pattern/{name}` (per-pattern CNN-LSTM), `/candle`, and `/health`. Models are loaded once at startup via `lifespan`. Artifact paths are hardcoded in `main.py` and `inference.py` — update the version integers in `patchTstGateVersion` and `PATTERN_VERSIONS` when deploying a new model. Trained models are copied into a flat `dist/artifacts/` (e.g. `gate_PatchTST_EUR_USD_H1_2026_v1.pt`, `fvg_CNN-LSTM_EUR_USD_H1_2026_v3.pt`, `order_block_CNN-LSTM_EUR_USD_H1_2026_v2.pt`).

Adding a new pattern endpoint: the generic `/pattern/{name}` handler reads from `PATTERN_REGISTRY` in `main.py`, so a new pattern only requires (1) a detector module in `dist/api/`, (2) a `PATTERN_REGISTRY` entry with `detector`, `detector_kwargs`, `n_active`, `pred_labels`, and `get_meta`, and (3) a `PATTERN_VERSIONS` entry in `inference.py`.

**Critical sync points:** `dist/api/data_processing.py` is a copy of `data_processing/dataparser.py` (plus live `getData()`). `dist/api/symmetry.py`, `dist/api/fair_value_gap.py`, `dist/api/order_block.py`, `dist/api/patchtst.py`, and `dist/api/cnn_lstm.py` are copies of their counterparts in `2_event-detection/` and `1_double-binary/PatchTST/`. When features or detector logic are changed in the source, they must be manually synced to `dist/api/`.
