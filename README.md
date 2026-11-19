# gta6-trophy-rarity-forecast-01 — v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DKAnbu/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DKAnbu/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecaster for the make.fun competition
"Forecast how many GTA VI players earn each trophy in week one" ($250).
It reads every `*.in` case in the input directory and writes one
`/output/<id>.out` per case: five quantiles (5/25/50/75/95%) of each
trophy's Exophase earn rate at the case's target.

## How it works

Per trophy, three median models compete and the winner is chosen by how
well it predicts the **last reads** (fit before them, scored on them):

- (a) earned-count / player-growth decomposition — fit `ln(players)` and
  `ln(earned)` separately against log-time, rate = earned / players;
- (b) a straight line of `logit(rate)` against log-time;
- (c) persistence — the last observed rate.

The winner is scored on the reads after the last step change in the
history (fit before it, scored on the new plateau), and fits use only
the last 24 hours of reads. The fitted median move is shrunk 40%
toward persistence and capped at 0.7 logit from the last rate, so a
slope that predicts the next refresh well cannot run away over 4.5
days of saturation.

The spread (logit sigma) is built in quadrature per trophy: fit residual
+ a horizon term growing like `sqrt(days)` + step-refresh volatility
(RMS of the non-zero logit jumps between consecutive reads) + a rarity
term scaling like `1/sqrt(earned count)` + an extra term for flat
histories. A trophy at 100% gets a near-zero spread.

Light monotone corrections: the story-chapter trophies (ids 3–7) are
pooled to non-increasing medians, and the platinum median is capped at
the smallest other trophy's median.

## Changes from v1

- Model selection is now explicit per trophy on last-read residuals
  (v1 blended two models by in-sample residual weight).
- Spread now includes step-refresh volatility and a rarity term scaled
  by `1/sqrt(earned)`, instead of one spread for all trophies.
- The chapter monotone correction is kept and applied in logit space
  via adjacent pooling.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Deterministic: two
runs on the same case write byte-identical output.
