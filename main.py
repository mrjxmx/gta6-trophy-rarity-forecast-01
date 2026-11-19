"""gta6-trophy-rarity-forecast-01: forecast every GTA VI PS5 trophy's Exophase earn rate at the case's target.

Model (per trophy, on the logit scale the contest scores on):

  * Trend. A weighted least-squares line of logit(rate) against ln(1 + hours since 2026-11-19T00:00Z),
    fitted over ALL of the case's reads (the house curve baseline uses only the last 24 h), with weights
    that halve every 30 hours so recent reads speak loudest. Each trophy's slope is shrunk toward the
    mean slope of its peer tiers (platinum borrows from gold+silver, gold from gold+silver, silver from
    silver+gold, bronze from bronze) by an empirical-Bayes factor var_between / (var_between + var_slope),
    so a rare trophy with a noisy handful of reads inherits its tier's drift instead of its own noise.
  * Anchor. The forecast starts from the trophy's LAST observed rate, not from the fitted line's level:
    median_logit = logit(last rate) + shrunk_slope * (x_target - x_last). The line supplies the drift;
    the last read supplies the level, which a least-squares line through step-shaped data gets wrong.
  * Dilution. Earn rate = earners / players tracked. The player base grows fast in week one, which
    mechanically drags down trophies whose earners accumulate slower than the crowd arrives. The player
    growth slope is fitted from log(players_tracked) over the same log-time axis; the projected growth
    over the horizon is charged against the median in proportion to (1 - phi), where phi is the share of
    player growth the trophy's earners plausibly keep up with: 1.0 for near-universal trophies, down to
    0.35 for the platinum. The charge is capped at 0.5 logit units and is zero when the base is not
    growing, so it can nudge a median but never wreck it.
  * Spread. sigma^2 = (0.55 * sqrt(days))^2 + resid_var + slope_var * (dx)^2 + step_var: a random-walk
    term a little wider than the house baseline's 0.5, the fit's residual variance (shrunk toward the
    tier median when reads are few), the slope's posterior variance over the extrapolation distance,
    and a jump term from the sizes of the trophy's observed refresh steps (Exophase updates in jumps,
    not smoothly), scaled by how many refreshes the horizon plausibly holds. The practice baseline's
    band was too narrow for the one refresh it met; this one prices refreshes in.
  * Degenerate trophies (rate 0 or missing in the last read, or a single observation, or a one-read
    case) fall back to the house baseline's formula for that trophy: last rate, 0.5 * sqrt(days) spread.

Quantiles are inv_logit(median + z_tau * sigma) with z = (-1.645, -0.674, 0, 0.674, 1.645), rounded to
4 decimals, clamped into [0.001, 99.999] and made non-decreasing. Pure standard library, no randomness,
no clock reads: two runs of a case write the same bytes.

Run like the grader does: python3 /bot/main.py, reading $MAKEFUN_INPUT_DIR (else /input) and writing
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first, in case a sibling module is ever shipped.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each scored level (NormalDist covers any other level a case asks for).
Z_BY_LEVEL = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
_NORM = NormalDist()

# The log-time axis starts at the game's release midnight, as the house curve baseline's does.
ORIGIN_TS = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()

# Random-walk spread per square-root day: the house baseline's 0.5, plus a margin for refresh risk.
K_RW = 0.55
# The house baseline's own spread, used for degenerate trophies.
BASE_K = 0.5
# Fit weights halve every this many hours of age before the case's last read.
WEIGHT_HALF_LIFE_H = 30.0
WEIGHT_FLOOR = 0.05
# Floors that keep a perfectly flat fit (the page simply had not refreshed) from claiming certainty.
RESID_FLOOR = 0.0025
VAR_BETWEEN_FLOOR = 0.01
SLOPE_VAR_FLOOR = 0.001
# Residual variance is shrunk toward the tier median with this many phantom reads of prior weight.
RESID_PRIOR_N = 6.0
# Dilution: projected player growth charged to a median is capped, in log and logit units respectively.
DLOGP_CAP = 1.4
DILUTION_CAP = 0.5
# Refresh-step variance: half the observed mean squared residual jump per expected refresh, capped.
STEP_SCALE = 0.5
STEP_VAR_CAP = 0.8
MAX_EXPECTED_JUMPS = 30.0

# Peer tiers whose slopes a tier borrows strength from (the platinum is a tier of one).
PEERS = {
    "platinum": ("platinum", "gold", "silver"),
    "gold": ("gold", "silver"),
    "silver": ("silver", "gold"),
    "bronze": ("bronze",),
}
TIERS = ("platinum", "gold", "silver", "bronze")


def clamp_rate(pct: float) -> float:
    """A rate into the scorer's clamp, 0.01..99.99, before any logit."""
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(v: float) -> float:
    """Logit back to percent, in the form that never overflows."""
    if v >= 0.0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up (as JavaScript's Math.round rounds), whole numbers as ints."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_ts(value) -> float | None:
    """An ISO 8601 timestamp to epoch seconds; None when it does not parse."""
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def x_of(ts: float) -> float:
    """ln(1 + hours since the release midnight); the first reads sit a few minutes before it."""
    hours = (ts - ORIGIN_TS) / 3600.0
    return math.log(1.0 + max(hours, -0.95))


def z_for(level: float) -> float:
    z = Z_BY_LEVEL.get(level)
    return z if z is not None else _NORM.inv_cdf(level)


def wls(xs, ys, ws):
    """Weighted least squares y ~ x. Returns slope/intercept/residual variance/slope variance/effective n,
    with slope and slope variance None when x never varies."""
    sw = sum(ws)
    if sw <= 0.0:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx > 1e-12:
        sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
        slope = sxy / sxx
        intercept = ybar - slope * xbar
        ssr = sum(w * (y - intercept - slope * x) ** 2 for x, y, w in zip(xs, ys, ws))
        resid = ssr / max(sw - 2.0, 1.0)
        slope_var = max(resid, RESID_FLOOR) / sxx
    else:
        slope = None
        intercept = ybar
        ssr = sum(w * (y - ybar) ** 2 for y, w in zip(ys, ws))
        resid = ssr / max(sw - 1.0, 1.0)
        slope_var = None
    return {"slope": slope, "intercept": intercept, "resid_var": resid, "slope_var": slope_var, "n_eff": sw}


def median_of(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def finish_row(median_logit: float, sigma: float, levels) -> list:
    """Quantiles from a median logit and a spread: rounded, clamped, non-decreasing."""
    row = []
    for level in levels:
        q = inv_logit_pct(median_logit + z_for(level) * sigma)
        row.append(min(99.999, max(0.001, round4(q))))
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def baseline_row(rate, days: float, levels) -> list:
    """The house baseline's row for one trophy: last rate, random-walk spread, nothing else."""
    lg = logit(rate if rate is not None else 0.0)
    return finish_row(lg, BASE_K * math.sqrt(days), levels)


def earner_fraction(tier: str, secret: bool, rate_pct: float) -> float:
    """The share of player-base growth a trophy's earners plausibly match (phi in the dilution charge).

    A trophy nearly everyone tracked has already earned keeps pace with the crowd by construction; the
    platinum's earners are the few finishers, who arrive far slower than new players do."""
    if rate_pct >= 50.0:
        return 1.0
    if rate_pct >= 25.0:
        return 0.9
    if tier == "platinum":
        return 0.35
    if tier == "gold":
        return 0.5
    if secret and rate_pct < 10.0:
        return 0.55
    if rate_pct < 2.0:
        return 0.6
    if rate_pct < 10.0:
        return 0.7
    return 0.8


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    horizon_minutes = float(case.get("horizon_minutes") or 0.0)
    days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0

    # Read times and log-time coordinates, oldest first.
    reads = []
    for read in history:
        ts = parse_ts(read.get("at")) or parse_ts(read.get("slot"))
        if ts is None:
            continue
        reads.append({"ts": ts, "x": x_of(ts), "read": read})
    reads.sort(key=lambda r: r["ts"])

    if len(reads) <= 1:
        last_rates = reads[-1]["read"].get("rates") if reads else None
        rows = []
        for i, trophy in enumerate(trophies):
            rate = last_rates[i] if last_rates and i < len(last_rates) else None
            rows.append({"id": trophy["id"], "q": baseline_row(rate, days, levels)})
        return {"trophies": rows}

    t_first = reads[0]["ts"]
    t_last = reads[-1]["ts"]
    span_hours = max((t_last - t_first) / 3600.0, 0.0)

    # Target on the same axis: the case's stated target, else checkpoint plus horizon.
    target_ts = parse_ts(case.get("target"))
    if target_ts is None:
        at_ts = parse_ts(case.get("at")) or t_last
        target_ts = at_ts + horizon_minutes * 60.0
    x_target = x_of(target_ts)

    # Recency weights, shared by every fit in the case.
    for r in reads:
        age_h = (t_last - r["ts"]) / 3600.0
        r["w"] = max(WEIGHT_FLOOR, 0.5 ** (age_h / WEIGHT_HALF_LIFE_H))

    # Player-base growth on the log-time axis (dilution), from every read that carries a count.
    gp = 0.0
    p_pts = [(r["x"], math.log(r["read"]["players_tracked"]), r["w"]) for r in reads
             if isinstance(r["read"].get("players_tracked"), (int, float)) and r["read"]["players_tracked"] > 0]
    if len(p_pts) >= 3:
        fit = wls([p[0] for p in p_pts], [p[1] for p in p_pts], [p[2] for p in p_pts])
        if fit and fit["slope"] is not None:
            gp = fit["slope"]

    last_read_rates = reads[-1]["read"].get("rates") or []

    # First pass: each trophy's own fit, last observation and refresh steps.
    per = []
    for i, trophy in enumerate(trophies):
        pts = []  # (x, logit rate, weight, ts)
        for r in reads:
            rates = r["read"].get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is not None:
                pts.append((r["x"], logit(rate), r["w"], r["ts"]))
        last_rate = last_read_rates[i] if i < len(last_read_rates) else None
        entry = {
            "trophy": trophy,
            "pts": pts,
            "last_rate": last_rate,
            "degenerate": last_rate is None or last_rate == 0 or len(pts) <= 1,
            "fit": None,
            "changes": [],
        }
        entry["last_rate_val"] = None  # filled by the sweep below, in percent
        if pts:
            entry["last_x"] = pts[-1][0]
            entry["last_logit"] = pts[-1][1]
        if len(pts) >= 2:
            entry["fit"] = wls([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
            # Refresh steps: consecutive distinct values, as (dx, dy) between level starts.
            level_x, level_y = pts[0][0], pts[0][1]
            for x, y, _w, _ts in pts[1:]:
                if y != level_y:
                    entry["changes"].append((x - level_x, y - level_y))
                    level_x, level_y = x, y
        per.append(entry)

    # The raw last observed rate per trophy (for the dilution table, which reads percent, not logits).
    for i, entry in enumerate(per):
        val = None
        for r in reversed(reads):
            rates = r["read"].get("rates") or []
            if i < len(rates) and rates[i] is not None:
                val = rates[i]
                break
        entry["last_rate_val"] = val

    # Peer-tier statistics over the trophies that have a slope: mean slope, spread of slopes, median
    # residual variance. Platinum's peers include gold and silver because a tier of one has no spread.
    def tier_stats(tier_names):
        slopes = []
        resids = []
        for entry in per:
            if entry["trophy"].get("tier") in tier_names and entry["fit"]:
                resids.append(entry["fit"]["resid_var"])
                if entry["fit"]["slope"] is not None:
                    slopes.append((entry["fit"]["slope"], entry["fit"]["slope_var"]))
        if slopes:
            weights = [1.0 / ((sv if sv is not None else 1.0) + 0.05) for _s, sv in slopes]
            mean = sum(w * s for w, (s, _sv) in zip(weights, slopes)) / sum(weights)
            var = sum(w * (s - mean) ** 2 for w, (s, _sv) in zip(weights, slopes)) / sum(weights)
        else:
            mean, var = None, None
        return mean, var, median_of(resids)

    global_mean, global_var, global_resid = tier_stats(TIERS)
    stats = {}
    for tier in TIERS:
        mean, var, resid = tier_stats(PEERS[tier])
        stats[tier] = {
            "mean": mean if mean is not None else global_mean,
            "var": max(var if var is not None else (global_var if global_var is not None else VAR_BETWEEN_FLOOR),
                       VAR_BETWEEN_FLOOR),
            "resid_med": resid if resid is not None else global_resid,
        }

    # Second pass: shrink, dilute, spread, emit.
    rows = []
    for entry in per:
        trophy = entry["trophy"]
        if entry["degenerate"] or entry["fit"] is None:
            rows.append({"id": trophy["id"], "q": baseline_row(entry["last_rate"], days, levels)})
            continue
        fit = entry["fit"]
        st = stats.get(trophy.get("tier"), stats["bronze"])
        var_between = st["var"]
        if fit["slope"] is None:
            slope = st["mean"] if st["mean"] is not None else 0.0
            slope_var = var_between
        else:
            lam = var_between / (var_between + fit["slope_var"])
            tier_mean = st["mean"] if st["mean"] is not None else fit["slope"]
            slope = lam * fit["slope"] + (1.0 - lam) * tier_mean
            slope_var = 1.0 / (1.0 / fit["slope_var"] + 1.0 / var_between)
        slope_var = max(slope_var, SLOPE_VAR_FLOOR)

        dx = x_target - entry["last_x"]

        # Residual variance, shrunk toward the tier median by effective reads.
        resid_med = st["resid_med"] if st["resid_med"] is not None else 0.04
        n_eff = fit["n_eff"]
        resid_used = (n_eff * fit["resid_var"] + RESID_PRIOR_N * resid_med) / (n_eff + RESID_PRIOR_N)

        # Refresh-step variance: residual jumps around the fitted drift, times the refreshes the
        # horizon plausibly holds at the rate the history showed.
        step_var = 0.0
        if entry["changes"] and span_hours > 0.0:
            jump_ms = sum((dy - slope * dxj) ** 2 for dxj, dy in entry["changes"]) / len(entry["changes"])
            expected = min(len(entry["changes"]) / span_hours * horizon_hours, MAX_EXPECTED_JUMPS)
            step_var = min(STEP_SCALE * jump_ms * expected, STEP_VAR_CAP)

        # Dilution: the player base's projected growth, charged at (1 - phi) for this trophy.
        phi = earner_fraction(trophy.get("tier"), bool(trophy.get("secret")), entry["last_rate_val"] or 0.0)
        dlogp = min(max(gp * dx, 0.0), DLOGP_CAP)
        dilution = max(-DILUTION_CAP, -(1.0 - phi) * dlogp)

        median_logit = entry["last_logit"] + slope * dx + dilution
        var = (K_RW * math.sqrt(days)) ** 2 + resid_used + slope_var * dx * dx + step_var
        sigma = max(math.sqrt(var), 0.05)
        rows.append({"id": trophy["id"], "q": finish_row(median_logit, sigma, levels)})
    return {"trophies": rows}


def fallback_case(case: dict) -> dict:
    """Starter-equivalent rows for every trophy, built to never raise: a case that forecasts badly
    still ranks; a case that crashes does not."""
    trophies = case.get("trophies") or []
    levels = case.get("levels") or LEVELS
    days = float(case.get("horizon_minutes") or 0.0) / 1440.0
    history = case.get("history") or []
    last_rates = (history[-1].get("rates") if history else None) or []
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i] if i < len(last_rates) else None
        rows.append({"id": trophy.get("id", i + 1), "q": baseline_row(rate, days, levels)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
            if len(forecast["trophies"]) != len(case.get("trophies") or []):
                raise ValueError("row count mismatch")
        except Exception:
            forecast = fallback_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
