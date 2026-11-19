# GTA VI trophy rarity forecaster — gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DKAnbu/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/DKAnbu/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty "Forecast how many GTA VI players earn each trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250 in USDC).

Forecasts, for each of the 55 GTA VI PlayStation 5 trophies, its Exophase
earn rate at the case target, as five quantiles (5%, 25%, 50%, 75%, 95%),
scored by pinball loss on the logit scale.

## How it works

Per trophy, the forecaster rebuilds earned counts (`E = rate/100 *
players_tracked`) and fits `log P(t)` and `log E(t)` separately against
`ln(1 + hours since 2026-11-19T00:00Z)`; the ratio at the target gives a
rate forecast that separates player-base dilution from player progression.
A direct `logit(rate)` fit is blended in, weighted by in-sample residual
error. A stale-page prior shifts the median down 0.30 logit when the whole
history is a single unchanged snapshot (a refresh is overdue), the platinum
median is capped at the smallest other median, and story-chapter medians
(ids 3–7) are pooled to be non-increasing. Spreads combine fit residual,
horizon (`0.45·√days`), stepwise-refresh jump volatility, and a small-count
term.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` per case.
Python 3.13, standard library only, deterministic, no network.
