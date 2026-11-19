"""GTA VI trophy rarity forecaster, v2 (make.fun gta6-trophy-rarity-forecast-01).

Reads every *.in case in the input directory and writes one forecast file
per case: five quantiles (5/25/50/75/95%) of each trophy's Exophase earn
rate at the case's target, as percent.

Method, per trophy (all fits in log-time, x = ln(1 + hours since
2026-11-19T00:00Z)):

  * Three candidate median models are scored on the reads AFTER the
    last step change in the history: each model is fit on the history
    before that change and asked to predict the new plateau, so the
    winner is the model that best predicts the next refresh step, not
    the one that fits the past best. (Scoring on the last few reads
    would be meaningless: Exophase holds its page constant between
    refreshes, so neighbouring reads are identical.)
      (a) earned-count / player-growth decomposition: fit ln(players)
          and ln(earned) separately against x, rate = earned/players.
      (b) a straight line of logit(rate) against x.
      (c) persistence: the last observed rate.
    Fits use only the last 24 hours of reads: launch-night growth was
    far steeper than the regime the target lives in, and a slope fit
    over the whole history overshoots. The winner is refit on the
    recent window and read at the target; the median may move at most
    1.0 logit from the last rate, so no extrapolation runs away.
    With fewer than 3 distinct reads there is no honest selection, so
    persistence is used. When the whole history is perfectly flat
    (launch night: one Exophase page, unchanged), the median is shifted
    down by 0.30 logit: the first refresh after launch night diluted
    almost every rate as newly tracked players held fewer trophies.
  * Spread (logit sigma) is built in quadrature per trophy, so rare
    and jumpy trophies get wider bands than common, steady ones:
    residual of the winning fit on the recent window (capped) + a
    horizon term growing like sqrt(days) + step-refresh volatility
    (RMS of the non-zero logit jumps between consecutive reads, the
    real signature of Exophase's stepwise refreshes) + a rarity term
    scaling like 1/sqrt(earned count) + an extra term when the history
    is flat (a refresh nobody has seen yet is coming). A trophy at
    100% gets a near-zero spread above its median.
  * Light monotone correction: the story-chapter trophies (ids 3-7)
    must have non-increasing medians (a later chapter cannot be
    earned by more players than an earlier one); violations are pooled
    to a common value in logit space. The platinum (id 1) median is
    capped at the smallest other trophy's median.

Deterministic, standard library only, no network, no randomness.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Chapter trophies whose medians must be non-increasing, in id order.
CHAPTER_IDS = (3, 4, 5, 6, 7)
# Prior shift (logit) applied when the history is perfectly flat: the
# first unseen refresh dilutes rates as new players are tracked.
FLAT_SHIFT = -0.30
# Extra logit spread when the history is flat (an unseen refresh looms).
FLAT_SPREAD = 0.55
HORIZON_COEF = 0.08
STEP_COEF = 0.35
RARITY_COEF = 0.9
# Fits use only reads from the last this many hours.
RECENT_HOURS = 24.0
# The median may not move further than this (logit) from the last rate.
MAX_MOVE = 0.7
# Residual contribution to the spread is capped here.
RESID_CAP = 0.18
# The fitted model's move from the last rate is shrunk this much toward
# persistence: a slope that predicts the next 4-hour step well still
# overstates a 4.5-day move, because earn curves saturate.
MEDIAN_SHRINK = 0.4


def parse_time(text):
    """ISO-8601 timestamp -> aware datetime (accepts a trailing Z)."""
    if text is None:
        return EPOCH
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def hours_since_epoch(moment):
    return (moment - EPOCH).total_seconds() / 3600.0


def clamp_pct(pct):
    return min(99.99, max(0.01, pct))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 20.0:
        return 99.99
    if lg <= -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def fit_line(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def mean_abs(values):
    return sum(values) / len(values) if values else 0.0


def rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


class TrophySeries:
    """One trophy's observed history, cleaned into parallel lists."""

    def __init__(self, case, index):
        self.xs = []       # ln(1 + hours since epoch) per read
        self.hours = []    # hours since epoch per read
        self.rates = []    # observed percent per read
        self.players = []  # players tracked per read
        self.earned = []   # reconstructed earned count per read
        for read in case.get("history") or []:
            rates = read.get("rates") or []
            if index >= len(rates):
                continue
            rate = rates[index]
            if rate is None:
                continue
            try:
                rate = float(rate)
                players = float(read.get("players_tracked") or 0)
            except (TypeError, ValueError):
                continue
            if players <= 0 or not (0.0 <= rate <= 100.0):
                continue
            moment = parse_time(read.get("at"))
            hours = hours_since_epoch(moment)
            self.hours.append(hours)
            self.xs.append(math.log1p(max(0.0, hours)))
            self.rates.append(rate)
            self.players.append(players)
            self.earned.append(rate / 100.0 * players)

    @classmethod
    def _blank(cls):
        obj = cls.__new__(cls)
        obj.xs = []
        obj.hours = []
        obj.rates = []
        obj.players = []
        obj.earned = []
        return obj

    def slice(self, lo, hi):
        """A new series with reads [lo, hi) of this one."""
        obj = self._blank()
        obj.xs = self.xs[lo:hi]
        obj.hours = self.hours[lo:hi]
        obj.rates = self.rates[lo:hi]
        obj.players = self.players[lo:hi]
        obj.earned = self.earned[lo:hi]
        return obj

    def recent(self):
        """Reads from the last RECENT_HOURS hours (all of them if few)."""
        if self.n < 2:
            return self
        cutoff = self.hours[-1] - RECENT_HOURS
        lo = 0
        while lo < self.n and self.hours[lo] < cutoff:
            lo += 1
        if self.n - lo < 3:
            lo = max(0, self.n - 3)
        return self.slice(lo, self.n)

    @property
    def n(self):
        return len(self.rates)

    @property
    def last_rate(self):
        return self.rates[-1] if self.rates else 1.0

    @property
    def last_players(self):
        return self.players[-1] if self.players else 0.0

    @property
    def is_flat(self):
        if self.n < 2:
            return self.n <= 1
        first_rate = self.rates[0]
        first_players = self.players[0]
        return all(r == first_rate for r in self.rates) and all(
            p == first_players for p in self.players
        )

    def step_rms(self):
        """RMS of the non-zero logit jumps between consecutive reads."""
        jumps = []
        for a, b in zip(self.rates, self.rates[1:]):
            d = logit_of(b) - logit_of(a)
            if d != 0.0:
                jumps.append(d)
        return rms(jumps)

    def last_change_index(self):
        """Index of the first read after the last step change, or None.

        A step change is a read whose rate or player count differs from
        the previous read. The reads from that index on form the newest
        plateau, the natural hold-out for one-step-ahead selection.
        """
        for i in range(self.n - 1, 0, -1):
            if self.rates[i] != self.rates[i - 1] or self.players[i] != self.players[i - 1]:
                return i
        return None


def decomp_predictor(train):
    """Fit ln(P) and ln(E) vs x on the training series; predict logit(rate)."""
    xs = train.xs
    ln_p = [math.log(max(1.0, p)) for p in train.players]
    ln_e = [math.log(max(1.0, e)) for e in train.earned]
    fp = fit_line(xs, ln_p)
    fe = fit_line(xs, ln_e)
    if fp is None or fe is None:
        return None

    def predict(x):
        ln_rate = (fe[0] + fe[1] * x) - (fp[0] + fp[1] * x)
        pct = 100.0 * math.exp(ln_rate)
        return logit_of(pct)

    return predict


def logitline_predictor(train):
    fit = fit_line(train.xs, [logit_of(r) for r in train.rates])
    if fit is None:
        return None

    def predict(x):
        return fit[0] + fit[1] * x

    return predict


def persistence_predictor(train):
    last = logit_of(train.last_rate)

    def predict(x):
        return last

    return predict


CANDIDATES = (
    ("decomp", decomp_predictor),
    ("logit", logitline_predictor),
    ("persist", persistence_predictor),
)


def select_model(series):
    """Score each candidate on the newest plateau; return the winner.

    Each candidate is fit on the recent history before the last step
    change and scored by mean absolute logit error on the reads after
    it. Ties (within 0.005 logit) go to the earlier candidate, i.e.
    the decomposition first. With no step change, or too little
    history, persistence is used.
    """
    if series.n < 3:
        return "persist"
    change = series.last_change_index()
    if change is None or change < 2:
        return "persist"
    train = series.slice(0, change).recent()
    if train.n < 2:
        return "persist"
    test = series.slice(change, series.n)
    test_y = [logit_of(r) for r in test.rates]
    best_name = "persist"
    best_err = None
    for name, builder in CANDIDATES:
        try:
            predict = builder(train)
        except (ValueError, OverflowError):
            predict = None
        if predict is None:
            continue
        try:
            errs = [abs(predict(x) - y) for x, y in zip(test.xs, test_y)]
        except (ValueError, OverflowError):
            continue
        err = mean_abs(errs)
        if best_err is None or err < best_err - 0.005:
            best_err = err
            best_name = name
    return best_name


def fit_median_logit(series, model_name, x_target):
    """Refit the chosen model on the recent window; predict at target."""
    builders = dict(CANDIDATES)
    recent = series.recent()
    predict = builders[model_name](recent)
    if predict is None:
        return logit_of(series.last_rate)
    try:
        value = predict(x_target)
    except (ValueError, OverflowError):
        return logit_of(series.last_rate)
    if not math.isfinite(value):
        return logit_of(series.last_rate)
    return max(-9.0, min(9.0, value))


def fit_residual_rms(series, model_name):
    """RMS logit residual of the chosen model on the recent window."""
    builders = dict(CANDIDATES)
    recent = series.recent()
    predict = builders[model_name](recent)
    if predict is None or recent.n == 0:
        return 0.3
    try:
        diffs = [predict(x) - logit_of(r) for x, r in zip(recent.xs, recent.rates)]
    except (ValueError, OverflowError):
        return 0.3
    if model_name == "persist":
        base = logit_of(series.last_rate)
        diffs = [logit_of(r) - base for r in recent.rates]
    return rms(diffs)


def forecast_trophy(series, x_target, horizon_days):
    """Return (median_logit, sigma_logit) for one trophy."""
    if series.n == 0:
        return logit_of(1.0), 1.2
    model = select_model(series)
    median = fit_median_logit(series, model, x_target)
    last_lg = logit_of(series.last_rate)
    if series.is_flat:
        median = last_lg + FLAT_SHIFT
    else:
        # Shrink the fitted move toward persistence, then cap it: no
        # extrapolation may run away from the last observed rate.
        median = last_lg + MEDIAN_SHRINK * (median - last_lg)
        median = max(last_lg - MAX_MOVE, min(last_lg + MAX_MOVE, median))
    resid = min(RESID_CAP, fit_residual_rms(series, model))
    step = series.step_rms()
    earned_last = series.last_rate / 100.0 * max(1.0, series.last_players)
    rarity = RARITY_COEF / math.sqrt(max(1.0, earned_last))
    if rarity > RARITY_COEF:
        rarity = RARITY_COEF
    flat_extra = FLAT_SPREAD if series.is_flat else 0.0
    sigma2 = (
        resid * resid
        + (HORIZON_COEF * math.sqrt(max(0.0, horizon_days))) ** 2
        + (STEP_COEF * step) ** 2
        + rarity * rarity
        + flat_extra * flat_extra
    )
    sigma = math.sqrt(sigma2)
    sigma = max(0.18, min(1.2, sigma))
    if series.last_rate >= 99.99:
        sigma = min(sigma, 0.06)
        median = max(median, logit_of(99.99))
    return median, sigma


def pool_nonincreasing(medians):
    """Pool adjacent violators so the chapter medians never increase.

    `medians` is a list of logit medians in chapter order; returns a
    new list, non-increasing, pooling each violating run to its mean.
    """
    out = list(medians)
    i = 0
    while i < len(out) - 1:
        if out[i] < out[i + 1] - 1e-12:
            j = i + 1
            total = out[i] + out[i + 1]
            count = 2
            while j + 1 < len(out) and total / count < out[j + 1] - 1e-12:
                j += 1
                total += out[j]
                count += 1
            pooled = total / count
            for k in range(i, j + 1):
                out[k] = pooled
            i = 0
        else:
            i += 1
    return out


def forecast_case(case):
    trophies = case.get("trophies") or []
    horizon_minutes = float(case.get("horizon_minutes") or 0)
    horizon_days = horizon_minutes / 1440.0
    target = parse_time(case.get("target"))
    if case.get("target") is None and case.get("at") is not None:
        target = parse_time(case.get("at"))
    x_target = math.log1p(max(0.0, hours_since_epoch(target)))

    medians = {}
    sigmas = {}
    for idx, trophy in enumerate(trophies):
        series = TrophySeries(case, idx)
        median, sigma = forecast_trophy(series, x_target, horizon_days)
        medians[trophy["id"]] = median
        sigmas[trophy["id"]] = sigma

    # Platinum cannot be earned by more players than any other trophy.
    if 1 in medians and len(medians) > 1:
        others = [v for k, v in medians.items() if k != 1]
        cap = min(others)
        if medians[1] > cap:
            medians[1] = cap

    # Story chapters (ids 3-7) in non-increasing order.
    present = [t for t in CHAPTER_IDS if t in medians]
    if len(present) > 1:
        pooled = pool_nonincreasing([medians[t] for t in present])
        for tid, value in zip(present, pooled):
            medians[tid] = value

    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        median = medians.get(tid, logit_of(1.0))
        sigma = sigmas.get(tid, 1.0)
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(inv_logit(median + z * sigma)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
