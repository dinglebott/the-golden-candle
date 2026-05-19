# trend_pullback tuning log

Train set: EUR_USD H1, first 80% of `2010-01-01_2026-05-01` data (~13 years).
Test set: last 20% (~3 years).
All runs use `n_value=96`, `SL_BUFFER=1.5`.

## V0 — Baseline (pre-BE)

`BE_TRIGGER_R=0`, `ADX_MIN=25`, `RSI_LONG_MIN=40`, `RSI_SHORT_MAX=60`, `TOLERANCE=0.20`, both directions.

| set | n | W/L/TO/BE | WR | exp |
|---|---|---|---|---|
| train | 304 | 96/186/22/0 | 31.6% | **-0.103R** |

Diagnostic: 50.8% of losers had ≥0.5R MFE before reversing. The fade pattern motivated adding a break-even trigger.

## V1 — BE_TRIGGER_R sweep (train)

Holding all other params at V0. Best is 0.5 by a clear margin.

| BE | n | W/L/TO/BE | WR | exp |
|---|---|---|---|---|
| 0.3 | 304 | 46/76/3/179 | 15.1% | -0.012R |
| 0.4 | 304 | 58/86/5/155 | 19.1% | +0.022R |
| **0.5** | 304 | 70/98/6/130 | 23.0% | **+0.047R** |
| 0.6 | 304 | 72/110/6/116 | 23.7% | +0.018R |
| 0.7 | 304 | 75/115/6/108 | 24.7% | +0.017R |

Tighter triggers kill too many winners; looser ones don't convert enough losers. Sweet spot is 0.5.

## V2 — Direction filter (train)

With `BE=0.5`. The full V0 had longs at +0.103R and shorts at -0.004R. Disabling shorts isolates the edge.

| variant | n | exp | long-only exp |
|---|---|---|---|
| both directions | 304 | +0.047R | +0.103R |
| **longs-only** | 145 | **+0.103R** | +0.103R |

Shorts contribute 159 trades at near-zero expectancy — pure noise drag on train.

## V3 — Trend strength (train, longs-only + BE=0.5)

| ADX_MIN | n | WR | exp |
|---|---|---|---|
| 25 (baseline) | 145 | 27.6% | +0.103R |
| **30** | 91 | 28.6% | **+0.163R** |
| 35 | 64 | 26.6% | +0.135R |

ADX=30 best. Above that, signal count drops too fast.

## V4 — RSI tightening (train, longs-only + BE=0.5 + ADX=30)

| RSI_LONG_MIN | n | WR | exp |
|---|---|---|---|
| 40 (baseline) | 91 | 28.6% | +0.163R |
| **45** | 85 | 30.6% | **+0.187R** |
| 50 | 71 | 26.8% | +0.143R |

Requiring slightly stronger H4 RSI helps. 50 overshoots.

## V5 — Tolerance sweep (train, longs-only + BE=0.5 + ADX=30 + RSI=45)

Total R column highlights diminishing returns past 0.15.

| TOLERANCE | n | exp | total R |
|---|---|---|---|
| 0.30 | 109 | +0.146R | 15.9 |
| 0.20 | 85 | +0.187R | 15.9 |
| **0.15** | 69 | **+0.234R** | **16.1** |
| 0.10 | 50 | +0.246R | 12.3 |
| 0.05 | 29 | +0.377R | 10.9 |

Below 0.15, expectancy rises but total R falls — fewer trades for diminishing marginal expectancy.

## V6 — TP_MIN_DIST sweep (train, longs-only + BE=0.5 + ADX=30 + RSI=45 + TOL=0.15)

Larger TP targets dramatically lift R per win.

| TP_MIN_DIST | n | W | WR | avg win | exp | total R |
|---|---|---|---|---|---|---|
| 1.0 | 69 | 30 | 43.5% | +1.11R | +0.185R | 12.8 |
| 1.5 | 69 | 23 | 33.3% | +1.59R | +0.234R | 16.1 |
| 2.0 | 69 | 19 | 27.5% | +2.17R | +0.300R | 20.7 |
| 2.5 | 69 | 16 | 23.2% | +2.63R | +0.311R | 21.5 |
| **3.0** | 69 | 13 | 18.8% | +3.00R | **+0.354R** | 24.4 |
| 3.5 | 69 | 8 | 11.6% | +3.21R | +0.354R | 24.4 |

TP_MIN=3.0 maximizes train expectancy at 24.4R total. But only 13 wins — high variance.

## V7 — Test-set validation

This is where the picture changed. Most train-tuned tightenings did not generalise.

| config | train exp | test exp | test n | test total R |
|---|---|---|---|---|
| V0 baseline + BE=0.5 (both dirs) | +0.047R | **+0.048R** | 107 | +5.13 |
| **Longs-only + BE=0.5 (V2)** | +0.103R | **+0.061R** | 81 | **+4.94** |
| Longs-only + BE=0.5 + ADX=30 (V3) | +0.163R | +0.025R | 46 | +1.15 |
| Full V6 stack (best train) | +0.354R | **-0.104R** | 31 | -3.22 |
| Full V6 stack with TP_MIN=2.0 | +0.300R | -0.086R | 31 | -2.67 |
| Full V6 stack with TP_MIN=1.5 | +0.234R | -0.127R | 31 | -3.94 |

**V2 (longs-only baseline + BE=0.5) is the only tightening that generalises.** Every additional filter from V3–V6 helps on train but degrades on test, with the stacked variants going outright negative.

Shorts on test were not the drag they were on train (+0.007R, basically flat), so the longs-only call is supported by per-trade economics rather than because shorts actively bleed on test.

## Final config (committed)

```python
ADX_MIN = 25
RSI_LONG_MIN = 40
RSI_SHORT_MAX = 60
TOLERANCE = 0.20
SL_BUFFER = 1.5
TP_MIN_DIST = 1.5
TP_MAX_DIST = 6.0
TP_FALLBACK = 2.0
# bear_trend hard-disabled (longs-only)
BE_TRIGGER_R = 0.5  # in backtest.py
```

Performance:

| set | n | WR | avg win | exp | total R |
|---|---|---|---|---|---|
| train | 145 | 27.6% | +1.59R | **+0.103R** | +14.9 |
| test | 81 | 27.2% | +1.54R | **+0.061R** | +4.9 |
| combined | 226 | 27.4% | — | ~+0.088R | ~19.8 |

Both sets positive, consistent WR/avg-win, ~+20R across the full series.

## What didn't work / lessons

- **Tightening SL_BUFFER (previously tested at 1.0)**: cost shows up immediately (winners killed) but R per win barely moves because the buffer is only a slice of total risk. Reverted to 1.5.
- **Tighter BE triggers (0.3, 0.4)**: convert too many winners to break-even without enough offsetting saved losers.
- **Strong-trend ADX (30, 35) on the broader strategy**: helps longs on train but hurts shorts; with longs-only, helps train but loses ~75% of the test edge per trade.
- **Tighter RSI / TOLERANCE / bigger TP targets**: classic curve-fit pattern — each lifts train expectancy while shrinking the sample, and the whole stack inverts on the test set.
- **Reactively dropping shorts because of train**: on the test set shorts were essentially flat, not negative. The longs-only call still narrowly wins on a per-trade basis but it's closer than train suggested.

The robust gains are entirely from (1) the break-even trigger and (2) running long-only. Filter parameters that *look* like edge on the train set were over-fitting in every case I tried.
