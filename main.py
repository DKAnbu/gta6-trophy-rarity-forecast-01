"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads every ``*.in`` case in the input directory and writes one
``<id>.out`` forecast per case: five quantiles of each trophy's Exophase
earn rate (percent) at the case target.

Method, per trophy
------------------
1.  Earned-count / player-growth decomposition.  From each read the
    earned count is rebuilt as ``E = rate / 100 * players_tracked``
    (the printed rate is rounded to two decimals, so this is accurate
    to a fraction of an earner).  ``log P(t)`` and ``log (E(t) + 0.5)``
    are each fitted by least squares against ``x = ln(1 + hours since
    2026-11-19T00:00Z)`` and read at the target; the ratio gives a
    rate forecast.  Slopes are fitted on the last 24 hours of distinct
    snapshots only, so the launch transient (rates starting near zero)
    cannot dominate the extrapolation, and the move is damped when
    the trend is already flattening.  This separates the two forces that move a rate:
    new players arriving (dilution, pushes rare / late-game rates
    down) and tracked players progressing (pushes early-story rates
    up).  A trophy whose earned count grows slower than the player
    base falls, and vice versa -- the direction is learned per
    trophy, never assumed.
2.  A direct fit of ``logit(rate)`` against the same ``x`` (the
    house curve baseline's shape) is fitted alongside it.  The two
    median logits are blended with weights from each model's
    in-sample residual error, so the better-fitting view dominates.
    With fewer than three distinct snapshots the fits are unusable
    and the forecast falls back to persistence.
3.  Stale-page prior.  Exophase refreshes its counts stepwise, hours
    apart, not continuously.  If a case's whole history is a single
    unchanged snapshot (the launch-night practice shape), no trend
    exists to fit and a refresh is overdue: the median is shifted
    down by 0.30 logit -- the size of the first refresh the house
    recorded (players 638 -> 3,868, total earned 943 -> 4,276,
    average rate x0.75) -- and the spread is widened.  This rule
    fires only when the history is literally flat, never when real
    movement has been seen.
4.  Logical caps.  The platinum (id 1) can only be held by players
    holding every other trophy, so its median is capped at the
    smallest other median.  The story-chapter trophies (ids 3-7)
    are played in order, so their medians are pooled to be
    non-increasing in id order.

Spread (quantiles)
------------------
Pinball loss punishes an overconfident forecast as much as a wrong
median, so the logit spread per trophy combines, in quadrature:
the fit residual (observed logit volatility), a horizon term
``0.45 * sqrt(days)``, the stepwise-refresh volatility (RMS of
successive logit jumps between distinct snapshots), and a
small-count term for trophies with few earners.  The total is
clamped to [0.25, 1.6] logit units; a trophy at 100% gets a near-zero
spread.  Quantiles use the standard-normal z values
(-1.645, -0.674, 0, 0.674, 1.645).

Deterministic, standard library only, O(reads x trophies).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Launch-night stale-page prior: median logit shift when the whole
# history is one unchanged snapshot, and the spread to use then.
FLAT_SHIFT = -0.30
FLAT_SIGMA = 0.90
# Largest logit move allowed away from the last observed rate.
MAX_LOGIT_MOVE = 1.5


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_origin(moment):
    return (moment - ORIGIN).total_seconds() / 3600.0


def x_of(moment):
    """Log-time coordinate; hours are clamped so 1 + h stays positive."""
    return math.log(1.0 + max(hours_since_origin(moment), -0.95))


def clamp_pct(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least-squares line y ~ a + b*x; returns (a, b, rmse) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid


def distinct_snapshots(history):
    """Collapse consecutive identical reads to one snapshot each."""
    snaps = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, read))
    return [read for _, read in snaps]


def forecast_trophy(xs, players, earned, rates, x_target, horizon_days,
                    last_rate, x_cut):
    """Return (median logit, sigma logit) for one trophy.

    Slopes are fitted on the recent window only (x >= x_cut, the last
    24 hours of snapshots): the launch transient -- rates starting
    near zero and exploding upward in the first hours -- would
    otherwise dominate a full-history fit and send the extrapolation
    far past any saturating truth.  The extrapolation is further
    damped when the recent slope is already flattening (concavity).
    """
    last_lg = logit(last_rate)
    n = len(xs)
    if n < 3:
        # No usable trend: persistence, with the stale-page prior
        # when the history never moved at all.
        if n <= 1:
            return last_lg + FLAT_SHIFT, FLAT_SIGMA
        return last_lg, 0.55

    # Recent window for the slope fits (launch transient excluded).
    rx = [i for i, x in enumerate(xs) if x >= x_cut]
    if len(rx) >= 3:
        idx = rx
    else:
        idx = list(range(n))
    fxs = [xs[i] for i in idx]
    fps = [players[i] for i in idx]
    fes = [earned[i] for i in idx]
    frs = [rates[i] for i in idx]
    fn = len(fxs)

    # Concavity: is the logit slope already flattening in the recent
    # window?  Compare the second half's slope with the first's.
    damp = 0.75
    half = fn // 2
    if half >= 2 and fn - half >= 2:
        early = ols(fxs[:half], [logit(r) for r in frs[:half]])
        late = ols(fxs[half:], [logit(r) for r in frs[half:]])
        if early is not None and late is not None and early[1] > 1e-9:
            ratio = late[1] / early[1]
            damp = 0.75 * min(max(ratio, 0.4), 1.0)

    # --- direct logit(rate) fit -------------------------------------
    direct = ols(fxs, [logit(r) for r in frs])
    # --- decomposition: log players and log earned -------------------
    p_fit = ols(fxs, [math.log(max(p, 1)) for p in fps])
    e_fit = ols(fxs, [math.log(max(e, 0.0) + 0.5) for e in fes])
    candidates = []  # (median logit, weight)
    if direct is not None:
        a, b, resid = direct
        candidates.append((a + b * x_target, 1.0 / (resid * resid + 0.04),
                           resid))
    if p_fit is not None and e_fit is not None:
        ap, bp, rp = p_fit
        ae, be, re = e_fit
        log_p = ap + bp * x_target
        log_e = ae + be * x_target
        rate = 100.0 * math.exp(log_e - log_p)
        # Implied logit residual of the decomposition on history.
        imp_resid = math.sqrt(sum(
            (logit(r) - (math.log(max(e, 0.0) + 0.5)
                         - math.log(max(p, 1)))) ** 2
            for r, e, p in zip(frs, fes, fps)) / fn)
        candidates.append((logit(rate),
                           1.0 / (imp_resid * imp_resid + 0.04), imp_resid))
    if not candidates:
        return last_lg, 0.55

    # Dampen slopes when extrapolating past the observed x range,
    # and more when the trend is already flattening (see above).
    x_max = max(fxs)
    damp *= 1.0 / (1.0 + 0.12 * max(0.0, x_target - x_max))
    w_sum = sum(w for _, w, _ in candidates)
    med = 0.0
    resid_w = 0.0
    for cand, w, resid in candidates:
        moved = last_lg + (cand - last_lg) * damp
        med += (w / w_sum) * moved
        resid_w += (w / w_sum) * resid
    med = min(max(med, last_lg - MAX_LOGIT_MOVE),
              last_lg + MAX_LOGIT_MOVE)

    # --- spread ------------------------------------------------------
    # Stepwise-refresh volatility: RMS of successive logit jumps.
    jumps = [logit(b) - logit(a) for a, b in zip(frs, frs[1:])]
    step = math.sqrt(sum(j * j for j in jumps) / len(jumps)) if jumps else 0.0
    last_earned = earned[-1] if earned else 0.0
    count_term = 0.6 / math.sqrt(last_earned + 2.0)
    horizon_term = 0.45 * math.sqrt(max(horizon_days, 0.0))
    sigma = math.sqrt(max(resid_w, 0.12) ** 2 + horizon_term ** 2
                      + (0.7 * step) ** 2 + count_term ** 2)
    sigma = min(max(sigma, 0.25), 1.6)
    if last_rate >= 99.99:
        sigma = min(sigma, 0.05)
    return med, sigma


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = x_of(target)

    snaps = distinct_snapshots(history)
    if not snaps:
        # No history at all: a neutral, very wide forecast.
        return {"trophies": [
            {"id": t["id"], "q": [0.05, 0.25, 1.0, 4.0, 20.0]}
            for t in trophies]}

    snap_x = [x_of(parse_time(r["at"])) for r in snaps]
    snap_p = [r.get("players_tracked") or 0 for r in snaps]
    last_rates = snaps[-1]["rates"]
    # Recent-window cut: 24 hours before the checkpoint, in x units.
    from datetime import timedelta
    x_cut = x_of(parse_time(case["at"]) - timedelta(hours=24))

    medians = []
    sigmas = []
    for i, trophy in enumerate(trophies):
        xs, ps, es, rs = [], [], [], []
        for j, read in enumerate(snaps):
            rate = read["rates"][i]
            if rate is None or snap_p[j] <= 0:
                continue
            xs.append(snap_x[j])
            ps.append(snap_p[j])
            es.append(rate / 100.0 * snap_p[j])
            rs.append(rate)
        if not rs:
            medians.append(logit(0.5))
            sigmas.append(1.2)
            continue
        last = rs[-1]
        med, sig = forecast_trophy(xs, ps, es, rs, x_target,
                                   horizon_days, last, x_cut)
        medians.append(med)
        sigmas.append(sig)

    # Logical caps on medians only (spreads are left untouched).
    if len(medians) > 1:
        others = min(medians[1:])
        medians[0] = min(medians[0], others)
    # Story chapters, ids 3-7, are played in order: pool medians so
    # they do not increase with chapter id.
    chapter = [i for i, t in enumerate(trophies) if 3 <= t["id"] <= 7]
    for k in range(1, len(chapter)):
        prev, cur = chapter[k - 1], chapter[k]
        if medians[cur] > medians[prev]:
            medians[cur] = medians[prev]

    rows = []
    for trophy, med, sig in zip(trophies, medians, sigmas):
        qs = []
        for z in Z:
            q = inv_logit(med + z * sig)
            qs.append(round4(min(100.0, max(0.01, q))))
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """A valid persistence forecast, used only if the main path fails."""
    rows = []
    last = (case.get("history") or [{}])[-1]
    rates = last.get("rates") or []
    spread = 0.5 * math.sqrt(max(case.get("horizon_minutes", 0), 0) / 1440.0)
    for i, trophy in enumerate(case["trophies"]):
        rate = rates[i] if i < len(rates) and rates[i] is not None else 1.0
        lg = logit(rate)
        qs = [round4(min(100.0, max(0.01, inv_logit(lg + z * spread))))
              for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
