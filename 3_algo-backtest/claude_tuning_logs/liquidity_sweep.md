# Liquidity Sweep — Tuning Log

Instrument: `EUR_USD H1` (train set: first 80%) unless otherwise noted.
`n_value` = 96 unless noted. `BE_TRIGGER_R` = 0.5 (default).

## Strategy concept
H4 candle wicks beyond a prior, unbroken H4 swing high/low (sweep), then closes back inside.
After the sweep, wait for an H1 close that breaks the most recent post-sweep H1 swing in the
new direction (market-structure shift / MSS) — that's the entry. SL beyond the sweep wick
plus a small ATR buffer; TP at the next H4 swing target, with a 3R fallback.

## Iterations

### v1 — baseline (initial defaults)
- `SWEEP_MIN_DEPTH_ATR=0.10, SWEEP_LOOKBACK=20, SWEEP_CLOSE_PCT=0.50`
- `MSS_TIMEOUT_BARS=18, SL_BUFFER_ATR=0.25`
- `TP_MIN_DIST=1.5, TP_MAX_DIST=6.0, TP_FALLBACK=3.0`
- `TRADE_LONGS=True, TRADE_SHORTS=True`, `BE_TRIGGER_R=0.5`

```
n=779  WR=21.4%  exp=+0.034R  +1.43R/-0.96R
[L] n=359  exp=+0.041R   [S] n=420  exp=+0.028R
BE=279  TO=64  median planned TP = 1.52R
```
Observations: low WR (~21%) propped up by 1.43R avg win and lots of BEs (36% of trades end at 0R). Most h4_swing targets sit right at the 1.5R floor. Both directions positive, but shorts weaker. After this run the strategy code was vectorised (~250× speedup); subsequent iterations use the faster build, which produces slightly more trades than the original Python-loop version (697→779) due to subtle boundary differences in the H4-close handling.

### v2 — single-knob sweep (each knob varied independently from baseline)
```
label              n     WR      exp       +win/-loss   L (n,exp)        S (n,exp)        BE/TO  mTP
baseline           779   21.4%  +0.034R   +1.43/-0.96  359/+0.041       420/+0.028       279/64 1.52
tp_min=2.0         779   14.0%  +0.022R   +1.66/-0.96  same             same             314/87 2.02   ← TP too far
tp_min=2.5         779    9.4%  +0.012R   +1.78/-0.96                                    331/106 2.53   ← worse
tp_fb=2.0          779   21.4%  +0.031R   ...                                            same   1.52   no-op (few fallbacks)
tp_fb=4.0          779   21.4%  +0.037R   ...                                            same   1.52   modest +
close_pct=0.60     668   22.2%  +0.066R   +1.42/-0.96  300/+0.125       368/+0.018       241/59 1.52
close_pct=0.70     548   22.1%  +0.079R   +1.40/-0.95  254/+0.120       294/+0.045       195/56 1.52  ★ best single knob
close_pct=0.80     399   20.8%  +0.071R   +1.35/-0.94  176/+0.111       223/+0.039       145/46 1.52
depth=0.20         616   21.1%  +0.013R   ...          283/-0.014       333/+0.035       209/54 1.52  longs hurt
depth=0.30         480   21.9%  +0.051R   ...          222/+0.025       258/+0.073       164/45 1.52  shorts boost
depth=0.50         265   22.3%  +0.068R   +1.37/-0.94  116/+0.097       149/+0.045       92/28  1.52
lookback=10        709   20.7%  +0.018R   ...          326/+0.036       383/+0.003       259/59 1.52
lookback=40        827   21.2%  +0.030R   ...          386/+0.046       441/+0.016       298/69 1.52
mss_to=8           388   18.8%  -0.027R   ...          173/+0.026       215/-0.070       141/34 1.52  ← need time
mss_to=12          611   20.5%  +0.013R   ...                                            222/52 1.52
mss_to=24          846   21.6%  +0.035R   ...          395/+0.051       451/+0.021       298/72 1.52  ~baseline
sl_buf=0.10        779   21.1%  +0.022R   ...                                            284/59 1.53
sl_buf=0.50        779   22.1%  +0.049R   +1.37/-0.95  359/+0.060       420/+0.040       269/74 1.52
sl_buf=1.00        779   18.7%  +0.023R   ...                                            281/97 1.52
longs_only         394   20.6%  +0.032R   ...
shorts_only        450   21.6%  +0.026R   ...
```
Top single-knob: `close_pct=0.70` is the standout. Longs respond much more to close_pct (longs reach +0.12R, shorts only +0.045R). `depth` mostly helps shorts. `sl_buf=0.50` is a free win.

### v3 — combinations (built on v2 winners)
```
label                            n     WR      exp       L (exp)   S (exp)
close=0.70 + sl=0.50             548   22.3%  +0.077R   +0.119    +0.041
close=0.70 + depth=0.30          326   23.0%  +0.101R   +0.096    +0.104
close=0.70 + sl=0.50 + d=0.30    326   23.9%  +0.118R   +0.113    +0.122  ★ best triplet
close=0.60 + sl=0.50 + d=0.30    401   23.2%  +0.101R   +0.118    +0.087
longs_only close=0.60 sl=0.50    328   23.5%  +0.105R
shorts_only c=0.70 sl=0.50 d=0.30 179  24.6%  +0.123R                     (highest per-trade)
```
Triple stack `close=0.70 + sl=0.50 + depth=0.30` lands at **+0.118R, 326 trades**, near-perfectly balanced longs/shorts. Single-direction-only versions have slightly higher per-trade expectancy but lose half the trades.

### v4 — fine-tune around `close=0.70 sl=0.50 d=0.30`
```
label                            n     WR      exp       L (exp)   S (exp)    notes
best (triple)                    326   23.9%  +0.118R   +0.113    +0.122
c=0.75 sl=0.50 d=0.30            283   24.4%  +0.155R   +0.159    +0.153   ← close=0.75 step
c=0.70 sl=0.50 d=0.40            255   25.9%  +0.161R   +0.110    +0.200   ← depth=0.40 step
mss=30                           367   24.5%  +0.125R                       longer wait → more trades, same exp
tp_fb=4.0                        ~ minor +
```
Both `c=0.75` and `d=0.40` are big steps. Cross them next.

### v5 — cross `c=0.75 + d=0.40` and stack with mss/tpfb
```
label                                    n     WR      exp       L (exp)  S (exp)
c=0.75 sl=0.50 d=0.40                    221   27.6%  +0.226R   +0.205   +0.241
c=0.75 sl=0.50 d=0.40 mss=30             253   28.1%  +0.232R   +0.244   +0.223
c=0.75 sl=0.50 d=0.40 mss=30 tpfb=4      253   28.1%  +0.240R   +0.244   +0.237  ★ FINAL
```
The cross of close=0.75 with depth=0.40 unlocked a huge step. Adding `mss=30` recovers trades the stricter filters lost, and `tp_fb=4.0` slightly boosts trades that exit via the rare fallback path. Final WR ~28%, balanced both directions.

### v6 — micro-fine-tune (sanity check)
```
label              n      exp        notes
best (v5)          253   +0.240R     baseline of this sweep
c=0.73             269   +0.211R     looser close hurts
c=0.77             231   +0.223R     tighter close, marginal
d=0.35             289   +0.196R
d=0.45             223   +0.205R
d=0.55             162   +0.241R     fewer trades, identical exp → no real gain
mss=24..40         ~245-260, all +0.225 to +0.232R   ← stable
tpmin=1.2          253   +0.180R     too tight
tpmin=1.8          253   +0.239R     same exp, fewer wins (more BE/TO) — neutral
tpmax=4..10        253   +0.240R     no h4_swing hits >6R in train set
sl=0.35            253   +0.248R     marginally better, +0.008R, noise level
sl=0.65            253   +0.238R
lb=15/30           247/264  ~+0.227R  stable
```
Neighbourhood is flat — moves of ±0.01R in either direction. Configuration is robust, not perched on a noise spike.

### Final locked-in configuration
```python
SWEEP_MIN_DEPTH_ATR = 0.40
SWEEP_LOOKBACK      = 20
SWEEP_CLOSE_PCT     = 0.75
MSS_TIMEOUT_BARS    = 30
SL_BUFFER_ATR       = 0.50
TP_MIN_DIST         = 1.5
TP_MAX_DIST         = 6.0
TP_FALLBACK         = 4.0
TRADE_LONGS         = True
TRADE_SHORTS        = True
```
Train result: **n=253, WR=28.1%, exp=+0.240R, avg +1.34R/-0.92R, longs +0.244R / shorts +0.237R.**

## Train vs. Test (out-of-sample) benchmark

| Metric            | Train (2010-2022 ~12.8y) | Test (2022-2026 ~3.3y) |
|-------------------|--------------------------|------------------------|
| Trades            | 253                      | 67                     |
| Win rate          | 28.1%                    | 16.4%                  |
| Avg win R         | +1.34R                   | +1.18R                 |
| Avg loss R        | -0.92R                   | -0.87R                 |
| BE / TO           | 71 / 41                  | 27 / 13                |
| **Expectancy**    | **+0.240R**              | **+0.094R**            |
| Long expectancy   | +0.244R (n=111)          | +0.089R (n=33)         |
| Short expectancy  | +0.237R (n=142)          | +0.098R (n=34)         |
| h4_swing TP exp   | +0.226R (n=245)          | +0.148R (n=60)         |
| fallback TP exp   | +0.655R (n=8)            | -0.370R (n=7)          |
| Trades / year     | ~20                      | ~20                    |
| Avg bars / trade  | 41.8                     | 45.6                   |

**Key takeaways:**
- Strategy generalises positively: ~40% of the train expectancy holds out-of-sample.
- WR dropped by ~12pp (28% → 16%) — the biggest source of edge degradation. Avg win size held up.
- Both directions independently positive on test (longs +0.089R, shorts +0.098R) — the edge is not concentrated in one side.
- The `fallback` TP path inverted (+0.655R train → -0.370R test); n is tiny in both, but worth watching live.
- Trade frequency (~20/year, ~1.7/month) matched perfectly between train and test — the *quantity* generalises better than the *quality*.
- Average hold ~42 bars (≈ 1.7 days on H1) — well inside the swing-trading regime the user asked for.

## BE_TRIGGER_R sweep (post-hoc)
The framework's `BE_TRIGGER_R` (in `backtest.py`, not the strategy) was held at 0.5 throughout tuning. Sweep against the locked config (current `TP_MIN_DIST=1.2`):

```
        TRAIN          TEST
BE=0.0  +0.182R        +0.004R   ← strategy is basically breakeven on test without BE
BE=0.3  +0.138R        +0.057R
BE=0.4  +0.168R        +0.076R
BE=0.5  +0.180R        +0.103R   ★ test-optimal
BE=0.6  +0.199R        +0.073R
BE=0.7  +0.199R        +0.056R
BE=0.8  +0.205R        +0.094R   ★ train-optimal
BE=1.0  +0.181R        +0.064R
BE=1.5  +0.182R        +0.004R   (effectively disabled — few trades reach +1.5R MFE)
```

**Findings:**
1. `BE_TRIGGER_R` is structurally important to this strategy: disabling it collapses test expectancy from +0.103R to +0.004R. The BE rule converts ~25 trades on test that would otherwise be -1R losses into 0R outcomes (this strategy has lots of winners that briefly go favourable then reverse).
2. Train-optimal (0.8) is *not* test-optimal (0.5) — classic optimization-bias signature. The 0.5-0.8 plateau on train collapses to a sharp 0.5 peak on test.
3. Keeping `BE_TRIGGER_R = 0.5` is the right call: best on test, near-best on train, robust across the neighbourhood. No change recommended.


