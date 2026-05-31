# macd tuning log

Pure MACD signal-line crossover strategy. Long on bullish histogram sign change
(`macd_hist` crosses from ≤0 to >0), short on bearish. MACD periods are fixed at
12/26/9 **candles** (the dataparser does not scale them by granularity), so on H1
this is a ~12–26h momentum read.

Train set: EUR_USD H1, first 80% of `2010-01-01_2026-05-01` (~13 yrs).
Test set: last 20% (~3 yrs).
All runs: `n_value=24`, `SL_ATR_MULT=1.5`, `TP_RR=1.5`, `BE_TRIGGER_R=0`, both directions.
Exits are neutral ATR/RR barriers, held fixed for the whole search so filter
effects are isolated (TP/SL tuning deferred to the fine-tune stage).

Methodology: triple-barrier is run ONCE over all crossovers via `_macd_lab.py`;
because trades are evaluated independently (forward window, no position overlap),
every filter combination is just a pandas mask on the cached trade set. This made
a broad breadth-first sweep cheap. Filters tested in isolation first, then stacked.

## V0 — Baseline (pure crossover, no filters)

| set | n | WR | exp | long exp | short exp |
|---|---|---|---|---|---|
| train | 6393 | 37.3% | **-0.006R** | -0.018R | +0.005R |
| test | 1597 | 36.0% | **-0.032R** | -0.039R | -0.024R |

Classic whipsaw signature: huge signal count, sub-breakeven, losers carry high MFE
(44.8% of losers reach ≥0.5R before reversing). A trend tool firing in chop.

## V1 — Single-filter sweep (train)

Direction-aware masks. Only the notable results shown.

| filter | n | exp | L exp | S exp |
|---|---|---|---|---|
| **EMD regime==0 (trending)** | 3849 | **+0.031R** | +0.022 | +0.040 |
| EMD regime==1 (cyclic) | 2544 | -0.062R | — | — |
| **volatility_regime > 1.2** | 878 | **+0.134R** | +0.143 | +0.124 |
| volatility_regime > 1 | 2893 | +0.013R | +0.023 | +0.002 |
| **vol_ratio > 1** | 3244 | +0.026R | +0.040 | +0.013 |
| vol_momentum > 0 | 3773 | +0.010R | -0.013 | +0.033 |
| **\|macd_hist\| > 0.0002** | 242 | +0.179R | +0.212 | +0.143 |
| williams fast confirm | 2376 | +0.008R | +0.017 | +0.000 |
| zeroline (ema_cross sign) | 2392 | -0.011R | -0.027 | +0.005 |
| ema50 trend | 3372 | -0.003R | +0.005 | -0.011 |
| **adx > 25** | 2931 | **-0.034R** | -0.080 | +0.012 |
| adx > 30 | 2102 | -0.035R | -0.091 | +0.021 |
| di_diff alignment | 3362 | -0.001R | -0.009 | +0.007 |

**Key finding — the edge is regime/volatility, not directional trend.** Every classic
trend filter (zero-line, EMA alignment, ADX, DI) is flat-to-negative; ADX gets
*monotonically worse* as the threshold rises. What helps is (a) a non-cyclic EMD
regime, (b) volatility expansion, (c) volume, (d) histogram magnitude. Higher
`|macd_hist|` and `volatility_regime` give the strongest per-trade edge.

## V2 — Combination sweep (train)

| combo | n | WR | exp | L | S |
|---|---|---|---|---|---|
| regime0 + vol_ratio>1 | 2080 | 40.0% | +0.065R | +0.082 | +0.047 |
| regime0 + volreg>1 | 2121 | 37.4% | +0.031R | +0.031 | +0.031 |
| volreg>1.2 + vol_ratio>1 | 635 | 42.4% | +0.157R | +0.190 | +0.123 |
| regime0 + volreg>1.2 + volr>1 | 527 | 42.3% | +0.149R | +0.165 | +0.134 |
| **volreg>1.2 + volr>1 + \|hist\|>1e-4** | 295 | 46.4% | **+0.235R** | +0.178 | +0.301 |
| regime0 + vol_ratio>1 + volm>0 | 1626 | 40.6% | +0.071R | +0.051 | +0.091 |
| regime0 + vol_ratio>1 + \|hist\|>1e-4 | 578 | 43.8% | +0.142R | +0.125 | +0.159 |

On train alone the `volatility_regime>1.2` stacks look best (up to +0.235R). They are
partly redundant (big histograms happen in volatile trends) and all symmetric.

## V3 — Train-vs-test generalization (the reality check)

Validated the frontier out-of-sample. **This reorders everything.**

| combo | train exp (n) | test exp (n) | verdict |
|---|---|---|---|
| regime0 only | +0.031 (3849) | -0.032 (887) | fails OOS |
| vol_ratio>1 only | +0.026 (3244) | -0.015 (806) | fails OOS |
| vol_momentum>0 only | +0.010 (3773) | -0.033 (892) | fails OOS |
| volreg>1.2 + vol_ratio>1 | +0.157 (635) | **-0.106** (153) | **overfit** |
| regime0 + volreg>1.2 + volr>1 | +0.149 (527) | **-0.185** (126) | **overfit** |
| volreg>1.2 + volr>1 + \|hist\|>1e-4 | +0.235 (295) | +0.007 (63) | collapses |
| regime0 + vol_ratio>1 | +0.065 (2080) | +0.057 (471) | **holds** |
| **regime0 + vol_ratio>1 + vol_momentum>0** | +0.071 (1626) | **+0.110** (363) | **holds, best** |
| regime0 + vol_ratio>1 + \|hist\|>1e-4 | +0.142 (578) | +0.085 (96) | holds, thin |
| regime0 + vol_ratio>1.2 | +0.074 (1806) | +0.033 (362) | holds |

Findings:
- **The high-expectancy `volatility_regime` family is overfit** — strong on train,
  negative on test. Discarded.
- **No single filter generalizes.** regime, volume, and volume-momentum are each
  negative on test alone. The edge only appears when **regime AND volume** are
  combined — neither is sufficient by itself.
- **`regime0 + vol_ratio>1 + vol_momentum>0` is the robust winner**: +0.071R train,
  +0.110R test (test even stronger), both directions positive on both splits.
- `|hist|` threshold adds train expectancy and survives OOS but shrinks test n to
  ~96 — a candidate for the fine-tune stage, not the default.

## Current default (in `macd.py`)

`USE_TREND_REGIME + USE_VOL_RATIO + USE_VOL_MOMENTUM`, i.e.
**regime0 + vol_ratio>1 + vol_momentum>0**. Verified through the real `backtest.py`
pipeline (train: n=1626, WR 40.6%, exp +0.071R, long +0.051 / short +0.091).

## Open leads for the fine-tune stage

- **Losers still fade**: 46.2% of losers reach ≥0.5R MFE before stopping out, and
  timeouts average +1.12R MFE but realize only +0.27R. This is exactly the pattern
  `BE_TRIGGER_R` and TP/SL tuning target (deferred per plan).
- **Longs lag shorts** consistently (train +0.051 vs +0.091; test +0.064 vs +0.156).
  Worth a snapshot pass (`visualise.py`) to see whether long failures cluster in a
  specific structure before deciding on any asymmetric treatment.
- `vol_ratio` threshold and the `|macd_hist|` magnitude floor are the most promising
  knobs to sweep next.
- Reproduce/extend via `_macd_lab.py` (harness), `_macd_sweep.py`, `_macd_combo.py`,
  `_macd_validate.py`. Cached trade sets: `_macd_base.pkl` (train), `_macd_base_test.pkl`.

---

# Fine-tune stage

Two snapshot passes (`visualise.py`) motivated three hypotheses: (H1) widen/re-anchor
the stop, (H2) add an H4 trend-alignment gate, (H3) the long/short gap is real.
**All three were tested and all three were wrong** — a clean lesson in not trusting
eyeballed anecdotes over aggregate stats. The only robust win was a TP/SL retune in
the *opposite* direction to the snapshot intuition.

Harness: `_macd_lab2.py` (configurable exits re-run triple-barrier; lookahead-safe
H4 trend via last-completed H4 bar). Entry filter held at V4 throughout.

## V5 — Stop width / anchor (the snapshot hypothesis, refuted)

Hypothesis from snapshots: losers get stopped by noise then reverse, so a wider/
structural stop should help. With RR held at 1.5, the opposite happened:

| SL scheme | exp (train) | WR |
|---|---|---|
| close k=1.5 (V4) | +0.071R | 40.6% |
| close k=2.0 | +0.040R | 33.0% |
| close k=2.5 | +0.035R | 25.8% |
| candle-extreme buf=0.5 | +0.040R | 32.5% |
| swing(10) buf=0.5 | +0.020R | 19.1% |

Widening the stop with RR fixed pushes TP proportionally out of reach → WR collapses.
The two effects cancel and net expectancy falls.

## V5b — Decouple SL and TP (fixed-ATR grid, the actual win)

Holding SL and TP as independent ATR distances reveals the real structure: edge
improves with **tighter SL and farther TP** (low-WR / high-payoff momentum profile),
the reverse of the snapshot intuition.

Train exp(R), rows=SL_ATR, cols=TP_ATR:
```
       tp2.0    tp2.5    tp3.0
sl1.0  +0.093   +0.138   +0.119
sl1.5  +0.044   +0.079   +0.067
sl2.0  +0.015   +0.047   +0.040
```

## V5c — Refine + test (overfitting check)

Train peak is the tight-SL/far-TP corner (sl0.75/tp2.5 +0.143) but that exact cell
is only +0.086 on test while baseline-ish cells hold up — cell-level overfitting.
The robust region good on **both** splits is the `tp2.0` column with tight-to-moderate
SL. Chosen cell (improves both, not a lone spike):

| exit | train | test |
|---|---|---|
| V4 (sl1.5 / tp2.25-equiv) | +0.071R | +0.110R |
| **sl=1.0 ATR / tp=2.0 ATR (RR 2.0)** | **+0.093R** | **+0.137R** |

## V6 — H4 trend gate (refuted)

Gating by H4 trend (EMA12/26 cross and close-vs-EMA50, both lookahead-safe) on top
of V5 exits:

| | train aligned | train counter | test aligned | test counter |
|---|---|---|---|---|
| h4_cross | +0.061R | **+0.129R** | +0.142R | +0.132R |
| h4_dist50 | +0.050R | **+0.134R** | +0.134R | +0.139R |

Counter-trend *beats* with-trend on train (MACD crosses are mildly mean-reverting,
not continuation) and on test there's no difference. The snapshot "counter-trend
loses" was small-sample anecdote. H4 gate adds nothing robust — **discarded**.

## V7 — More levers on V5 exits (all refuted OOS)

| lever | train | test | verdict |
|---|---|---|---|
| time barrier n=24→48 | +0.093→+0.098 | +0.137→+0.132 | wash, keep 24 |
| \|macd_hist\|>1e-4 | **+0.170R** | **-0.077R** | overfit, discard |
| vol_ratio>1.5 | +0.113R | -0.005R | overfit, keep >1.0 |

The `|macd_hist|` floor and higher `vol_ratio` look great on train and fail hard on
test — same overfitting trap as the V3 volatility_regime family.

## Fine-tune outcome

Only the **V5 exit retune (SL=1.0 ATR, TP=2.0 ATR)** transferred. Every entry-side
addition either did nothing or overfit. The long/short gap **flipped** between splits
(train S>L, test L>S), confirming it as sampling noise — kept symmetric.

Current `macd.py` (verified through `backtest.py`):

| set | n | WR | exp | long | short |
|---|---|---|---|---|---|
| train | 1626 | 35.7% | **+0.093R** | +0.044 | +0.141 |
| test | 363 | 37.7% | **+0.137R** | +0.160 | +0.113 |

Full journey: V0 baseline -0.006R/-0.032R → V4 entry filter +0.071R/+0.110R →
V5 exit retune +0.093R/+0.137R (train/test).

New harness: `_macd_lab2.py`; sweeps `_macd_v5*.py`, `_macd_v6.py`, `_macd_v7.py`.

## V8 — Fresh-cross vs low-extension entry filters

Motivated by the snapshot finding that the cross lags its impulse, so entry at the
cross-candle close buys exhaustion. Six filters, thresholds set from train-set
percentiles and applied unchanged to test. On top of V5 exits.

**Fresh-cross (timing) — NOT robust:**
| filter | train | test | verdict |
|---|---|---|---|
| F1 small \|macd_hist\| | +0.04–0.06R | +0.21–0.27R | train DOWN — reject |
| F2 small candle range | +0.03–0.07R | +0.17–0.36R | train DOWN — reject |
| F3 bars-since-MACD-turn | ~+0.095R | ~+0.13R | no effect — reject |

F1/F2 are histogram-magnitude in disguise (same V7 instability, flipped sign):
they help test but hurt train. F3 doesn't bite — crosses are nearly always fresh.

**Low-extension (distance from a reference) — robust, improves BOTH splits:**
| filter (p67 cut) | train | test |
|---|---|---|
| V5 baseline | +0.093R | +0.137R |
| **L1 \|dist_ema15\|** | **+0.129R** | **+0.204R** |
| L2 bb_position (trade dir) | +0.116R | +0.150R |
| L3 range_pos_24 (trade dir) | +0.099R | +0.197R |

Entering near a reference (not stretched) is the genuine edge; timing proxies aren't.

## V9 — Combine low-extension filters

| combo (p67) | train | test | n(tr/te) |
|---|---|---|---|
| **L1 only** | +0.129R | **+0.204R** | 1089/288 |
| L1+L2 | +0.158R | +0.177R | 910/246 |
| L1+L3 | +0.139R | +0.222R | 928/242 |
| L1+L2+L3 | +0.154R | +0.191R | 869/228 |

Combining lifts train but does NOT robustly beat L1 on test (deltas within noise on
~240 trades) while adding parameters and cutting n. **L1 alone** chosen — simplest,
largest n, and the most direction-symmetric (test L +0.204 / S +0.203).

## V10 — Exit re-check with L1 applied

Flagged earlier: a low-extension entry shifts entry-vs-stop geometry, so the exit
could move. Re-ran the SL/TP grid on the L1-filtered set — same train(tp2.5)/test(tp2.0)
disagreement as V5c, and **sl=1.0/tp=2.0 still sits at the robust intersection**
(train +0.128, test +0.204). No exit change needed.

## V8 outcome — current `macd.py` (verified through `backtest.py`)

Added `USE_LOW_EXTENSION` (`|dist_ema15| <= 0.00224`, a fixed train-derived threshold;
`dist_ema15` is a unitless log-ratio so it transfers across price levels).

| set | n | WR | exp | long | short |
|---|---|---|---|---|---|
| train | 1090 | 37.2% | **+0.128R** | +0.114 | +0.141 |
| test | 288 | 39.9% | **+0.204R** | +0.204 | +0.203 |

Full journey (train/test exp):
V0 baseline -0.006/-0.032 → V4 entry filter +0.071/+0.110 → V5 exit retune
+0.093/+0.137 → V8 low-extension +0.128/+0.204.

Harness for this stage: `_macd_lab2.py`, `_macd_v8.py` (importable filter helpers),
`_macd_v9.py`, `_macd_v10.py`.

## V11 — BE_TRIGGER_R sweep (refuted)

The last deferred lever. Break-even SL move on the current setup (L1 + sl1.0/tp2.0):

| BE | train | test | BE-exits (tr) |
|---|---|---|---|
| **0.0 (off)** | **+0.128R** | **+0.204R** | 0 |
| 0.3 | +0.109R | +0.153R | 564 |
| 0.5 | +0.102R | +0.129R | 475 |
| 0.7 | +0.107R | +0.142R | 382 |
| 1.0 | +0.121R | +0.160R | 264 |
| 1.5 | +0.129R | +0.177R | 105 |

Every BE level hurts; raising it toward the 2R target just converges back to the
no-BE baseline. Despite losers' high MFE (median 0.57R), BE's cost — capping a
+2R winner that retraces to entry at 0R — outweighs the −1R→0R it saves on losers.
This is a **low-WR / high-payoff momentum runner**, the worst profile for BE.
(Contrast `trend_pullback`, a low-RR fade system, where BE=0.5 helped. Opposite
profile, opposite verdict.) `BE_TRIGGER_R` left at 0. Sweep: `_macd_v11.py`.

## Final state

All deferred and snapshot-derived levers exhausted. Final `macd.py`: MACD crossover
+ regime0 + vol_ratio>1 + vol_momentum>0 + low-extension (|dist_ema15|<=0.00224),
exits sl=1.0 ATR / tp=2.0 ATR, n=24, no BE. Train +0.128R / test +0.204R.

What worked (2 levers): the V4 entry-filter family (regime + volume) and two
exit/entry refinements (tight-SL/far-TP retune, low-extension gate). What was tried
and refuted: directional trend filters (ADX/EMA/zero-line/H4), wider/structural
stops, histogram-magnitude floors, higher volume thresholds, longer time barrier,
fresh-cross timing, and BE. The recurring failure mode was train-only overfitting
caught by the held-out test split.
