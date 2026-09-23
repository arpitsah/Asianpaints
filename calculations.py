"""
calculations.py - the single formula engine for the Product Lifecycle &
Inventory Decision System.

EVERY business formula lives in this file. The UI (app.py / components.py)
only calls these functions and renders what they return.

To change a formula, edit ONLY its function:

    Momentum ................ calculate_momentum()
    Predictability .......... calculate_predictability()
    Reach ................... calculate_reach()
    Position ................ calculate_position()
    Signal -> 0-100 score ... normalise_signals()
    LSS ..................... calculate_lss()
    Lifecycle stage ......... determine_lifecycle()
    Two-month validation .... apply_confirmation_rule()
    SS / cycle stock / ROP .. calculate_inventory()
    Recommendation .......... generate_strategy()
    Alerts .................. generate_alerts()

Each calculate_* function returns a `Metric` that carries its own inputs,
formula text and working steps, so the "View calculation" panels in the UI
always describe the formula that actually ran.

Sources (Breaker of Chains deck, IIM Bangalore):
  * Slide 4/5     - four LSS signals, weights, raw formulas
  * Slide 3/4     - lifecycle thresholds (Intro age <= 9 months, 60/40/20)
  * Slide 8       - two-layer architecture and stage-wise policy playbook
  * Slide 9       - Steer incentive treatment by stage
  * Annexure A    - signal normalisation (portfolio percentile; Position = peak ratio)
  * Annexure B    - Silver-Pyke-Peterson safety stock, cycle stock, ROP, Z by stage
  * Slide 7 image - EOQ = sqrt(2DS/H) (optional cycle-stock method)
Anything not in the deck is labelled basis="Assumption" and is configurable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# =============================================================================
# 1. CONFIGURATION (edit here - nothing below hard-codes these numbers)
# =============================================================================

SIGNALS = ["momentum", "predictability", "reach", "position"]
SIGNAL_LABELS = {
    "momentum": "Momentum",
    "predictability": "Predictability",
    "reach": "Reach",
    "position": "Position",
}
SIGNAL_QUESTIONS = {
    "momentum": "Is genuine demand speeding up or slowing down?",
    "predictability": "Can we plan around this demand?",
    "reach": "How widely is the SKU being bought?",
    "position": "How close is the SKU to its own best days?",
}
LSS_WEIGHTS = {"momentum": 0.35, "predictability": 0.25, "reach": 0.20, "position": 0.20}

STAGES = ["Introduction", "Growth", "Maturity", "Decline", "Exit"]
STAGE_COLORS = {  # validated for colour-vision deficiency; always paired with a text label
    "Introduction": "#A77BEF",
    "Growth": "#2A9D6F",
    "Maturity": "#2456A6",
    "Decline": "#E39B2D",
    "Exit": "#A8283A",
    "N/A": "#9AA3AF",
}
LIFECYCLE_THRESHOLDS = {
    "introduction_max_age_months": 9,  # age-gated
    "growth_min_lss": 60,
    "maturity_min_lss": 40,
    "decline_min_lss": 20,
}

DATA_RULES = {
    "min_months_momentum": 27,       # current T3M + same T3M 1 and 2 years back
    "min_months_predictability": 3,  # periods with forecast and actual
    "mape_window_months": 12,        # deck: measured over 12 months
    "peak_window_months": 24,        # deck: peak over last 24 months
}

# Stage-wise inventory policy playbook (deck slide 8 + Annexure B Z values).
STAGE_POLICY: dict[str, dict[str, Any]] = {
    "Introduction": {
        "headline": "Build awareness and distribution",
        "business_action": "Build availability and drive trial",
        "objective": "Learn + ensure availability",
        "service_priority": "High",
        "z": 1.65,
        "safety_stock_approach": "Analogue / reference SKU variability",
        "cycle_stock_approach": "Small / frequent: d̄·T/2, weekly T",
        "cycle_stock_method": "review",
        "rop_approach": "Sell-through trigger",
        "network_footprint": "Launch locations",
        "push_pull": "Pull",
        "review_cadence": "Weekly",
        "review_period_days": 7,
        "replenishment_posture": "Small, frequent replenishment against early sell-through",
        "incentive": "Controlled launch buffer; allocate selectively (Serve)",
        "network_coverage": 0.20,
        "zero_safety_stock": False, "zero_cycle_stock": False, "no_rop": False,
    },
    "Growth": {
        "headline": "Scale supply and maximise availability",
        "business_action": "Invest and scale",
        "objective": "Capture adoption",
        "service_priority": "Highest",
        "z": 1.75,
        "safety_stock_approach": "Rolling recent variability",
        "cycle_stock_approach": "Re-optimise frequently: d̄·T/2, biweekly T",
        "cycle_stock_method": "review",
        "rop_approach": "Frequent recalculation",
        "network_footprint": "Expand Push",
        "push_pull": "Push (expanding)",
        "review_cadence": "Weekly / biweekly",
        "review_period_days": 14,
        "replenishment_posture": "Fast replenishment; recalculate buffers as volume ramps",
        "incentive": "Support scaling - higher trade support (Steer)",
        "network_coverage": 0.60,
        "zero_safety_stock": False, "zero_cycle_stock": False, "no_rop": False,
    },
    "Maturity": {
        "headline": "Optimise inventory and defend share",
        "business_action": "Optimise",
        "objective": "Maximise productivity",
        "service_priority": "Stable",
        "z": 1.53,
        "safety_stock_approach": "Stable long-history variability",
        "cycle_stock_approach": "Stable cycle: d̄·T/2, monthly T",
        "cycle_stock_method": "review",
        "rop_approach": "Standard MEIO logic",
        "network_footprint": "Full Push",
        "push_pull": "Push (full)",
        "review_cadence": "Monthly",
        "review_period_days": 30,
        "replenishment_posture": "Standard make-to-stock replenishment",
        "incentive": "Standard treatment (Steer)",
        "network_coverage": 1.00,
        "zero_safety_stock": False, "zero_cycle_stock": False, "no_rop": False,
    },
    "Decline": {
        "headline": "Manage down stock and plan rationalisation",
        "business_action": "Rationalise",
        "objective": "Release working capital",
        "service_priority": "Selectively lower",
        "z": 1.16,
        "safety_stock_approach": "Reduced buffer",
        "cycle_stock_approach": "EOQ √(2DS/H) — cap exposure",
        "cycle_stock_method": "eoq",
        "rop_approach": "Reduced locations",
        "network_footprint": "Pareto Push + regional Pull",
        "push_pull": "Pareto Push + regional Pull",
        "review_cadence": "Quarterly",
        "review_period_days": 90,
        "replenishment_posture": "Replenish selectively; cap exposure",
        "incentive": "Reduce push incentive (Steer)",
        "network_coverage": 0.40,
        "zero_safety_stock": False, "zero_cycle_stock": False, "no_rop": False,
    },
    "Exit": {
        "headline": "Liquidate stock and consider delisting",
        "business_action": "Exit",
        "objective": "Minimise residual exposure",
        "service_priority": "Committed orders only",
        "z": 0.94,
        "safety_stock_approach": "No forward safety stock",
        "cycle_stock_approach": "Confirmed demand only",
        "cycle_stock_method": None,
        "rop_approach": "None (no proactive ROP)",
        "network_footprint": "Centralise",
        "push_pull": "Pull (centralised)",
        "review_cadence": "Event-driven",
        "review_period_days": None,
        "replenishment_posture": "Replenish only against committed orders",
        "incentive": "No push incentive (Steer)",
        "network_coverage": 0.0,  # single central location
        "zero_safety_stock": True, "zero_cycle_stock": True, "no_rop": True,
    },
}

ALERT_THRESHOLDS = {
    "low_reach_ratio": 0.25,
    "low_forecast_accuracy": 0.60,
    "low_position_ratio": 0.60,
    "position_drop_points": 10.0,     # fall in Position over the last 3 months
    "transition_band_lss": 3.0,       # LSS within +/- this of a stage boundary
    "intro_graduation_window": 2,     # months before Introduction gate ends
    "low_lss": 40.0,
}


@dataclass
class EngineSettings:
    """User-adjustable engine options (sidebar). Defaults reproduce the deck."""
    normalisation: str = "percentile"      # "percentile" (Annexure A) | "absolute" (assumption)
    min_peers: int = 5                     # below this, percentile falls back to absolute
    confirm_enabled: bool = True           # two-month validation rule (slide 4)
    confirm_months: int = 2
    momentum_fallback: str = "chain"       # "strict" | "yoy" | "chain"  (see calculate_momentum)
    ordering_cost: float = 1000.0          # Rs/order (slide 7 assumption box)
    holding_rate: float = 0.20             # 20% p.a. (slide 7 assumption box)
    shelf_life_cap: bool = True            # guardrail (assumption)
    shelf_life_max_fraction: float = 0.5   # never hold more than 50% of shelf life as cover
    weights: dict = field(default_factory=lambda: dict(LSS_WEIGHTS))


# =============================================================================
# 2. RESULT TYPE
# =============================================================================

@dataclass
class Metric:
    name: str
    value: float | None = None            # final value (score 0-100, or units)
    raw: float | None = None              # raw signal before normalisation
    inputs: dict = field(default_factory=dict)
    formula: str = ""
    steps: list = field(default_factory=list)
    status: str = "ok"                    # "ok" | "na" | "pending"
    reason: str = ""
    basis: str = "Deck"                   # "Deck" | "Assumption"
    unit: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.value is not None

    def display(self, fmt: str = "{:,.1f}") -> str:
        if self.status == "pending":
            return "Formula pending"
        if not self.ok:
            return "N/A — insufficient data"
        return fmt.format(self.value)


def _na(name: str, reason: str, inputs: dict | None = None, formula: str = "") -> Metric:
    return Metric(name=name, status="na", reason=reason, inputs=inputs or {}, formula=formula)


def _num(x) -> float | None:
    """Coerce to float; None/NaN/'' -> None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# =============================================================================
# 3. INPUT DERIVATION FROM MONTHLY HISTORY
# =============================================================================

HISTORY_COLUMNS = ["month", "sales_volume", "forecast", "active_points"]


def history_frame(history) -> pd.DataFrame:
    df = pd.DataFrame(history or [], columns=HISTORY_COLUMNS)
    if df.empty:
        return df
    df = df.dropna(subset=["month"]).copy()
    df["month"] = df["month"].astype(str).str.slice(0, 7)
    for c in ["sales_volume", "forecast", "active_points"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("month").drop_duplicates("month", keep="last").reset_index(drop=True)


def derive_inputs_from_history(history, upto: int | None = None) -> dict:
    """Turn monthly history into the snapshot inputs the signal functions use.
    `upto` = index of the last month to use (for month-by-month replay)."""
    df = history_frame(history)
    if upto is not None:
        df = df.iloc[: upto + 1]
    n = len(df)
    out = {"months_available": n, "as_of": df["month"].iloc[-1] if n else None,
           "current_t3m": None, "previous_t3m": None, "t3m_y1": None, "t3m_y2": None, "current_volume": None,
           "peak_volume": None, "actuals": [], "forecasts": [], "active_points": None}
    if n == 0:
        return out
    vol = df["sales_volume"]
    if n >= 3 and vol.iloc[-3:].notna().all():
        out["current_t3m"] = float(vol.iloc[-3:].sum())
        out["current_volume"] = out["current_t3m"]
        window = vol.iloc[-DATA_RULES["peak_window_months"]:]
        out["peak_volume"] = float(window.rolling(3).sum().max())
    if n >= 6 and vol.iloc[-6:-3].notna().all():
        out["previous_t3m"] = float(vol.iloc[-6:-3].sum())
    if n >= 15 and vol.iloc[-15:-12].notna().all():
        out["t3m_y1"] = float(vol.iloc[-15:-12].sum())
    if n >= 27 and vol.iloc[-27:-24].notna().all():
        out["t3m_y2"] = float(vol.iloc[-27:-24].sum())
    recent = df.iloc[-DATA_RULES["mape_window_months"]:]
    pairs = recent.dropna(subset=["sales_volume", "forecast"])
    out["actuals"] = pairs["sales_volume"].tolist()
    out["forecasts"] = pairs["forecast"].tolist()
    ap = df["active_points"].dropna()
    out["active_points"] = float(ap.iloc[-1]) if len(ap) else None
    return out


def resolve_inputs(product: dict, upto: int | None = None) -> dict:
    """Return the demand-signal inputs for a product, from history or manual entry."""
    if product.get("input_mode") == "history":
        d = derive_inputs_from_history(product.get("history"), upto)
        d["source"] = "Monthly history"
    else:
        f, a = _num(product.get("forecast")), _num(product.get("actual"))
        d = {
            "current_t3m": _num(product.get("current_t3m")),
            "previous_t3m": _num(product.get("previous_t3m")),
            "t3m_y1": _num(product.get("t3m_y1")),
            "t3m_y2": _num(product.get("t3m_y2")),
            "current_volume": _num(product.get("current_volume")),
            "peak_volume": _num(product.get("peak_volume")),
            "actuals": [a] if a is not None else [],
            "forecasts": [f] if f is not None else [],
            "active_points": _num(product.get("active_points")),
            "months_available": None,
            "source": "Manual entry",
        }
        if f is None or a is None:
            d["actuals"], d["forecasts"] = [], []
    d["total_points"] = _num(product.get("total_points"))
    return d


# =============================================================================
# 4. THE FOUR SIGNALS (raw values)
# =============================================================================

def calculate_momentum(current_t3m, t3m_y1, t3m_y2, previous_t3m=None, fallback: str = "chain") -> Metric:
    """Final formula - year-on-year T3M momentum:
        Momentum = [ (S_T3M,y0 ÷ S_T3M,y−1) + (S_T3M,y0 ÷ S_T3M,y−2) ] ÷ 2 − 1
    S_T3M,y−1 / y−2 = sales in the same three months one / two years earlier.

    `fallback` (ASSUMPTION - for SKUs too young to have y−2 / y−1 history):
        "strict" → N/A unless both prior years exist
        "yoy"    → if y−2 is missing, use the y−1 ratio alone
        "chain"  → as "yoy"; if y−1 is also missing, use sequential T3M growth (deck slide 5)"""
    formula = "Momentum = [ (S_T3M,y0 ÷ S_T3M,y−1) + (S_T3M,y0 ÷ S_T3M,y−2) ] ÷ 2 − 1"
    cur, y1, y2, prev = _num(current_t3m), _num(t3m_y1), _num(t3m_y2), _num(previous_t3m)
    inputs = {"Current T3M sales (y0)": cur, "Same T3M last year (y−1)": y1,
              "Same T3M two years ago (y−2)": y2}
    if prev is not None:
        inputs["Previous T3M sales (fallback only)"] = prev
    if cur is None:
        return _na("Momentum", "Needs current T3M sales.", inputs, formula)
    if any(v is not None and v < 0 for v in (cur, y1, y2, prev)):
        return _na("Momentum", "Sales cannot be negative.", inputs, formula)
    for v, lbl in ((y1, "y−1"), (y2, "y−2")):
        if v == 0:
            return _na("Momentum", f"T3M sales for {lbl} is zero - ratio undefined.", inputs, formula)
    if y1 is not None and y2 is not None:
        r1, r2 = cur / y1, cur / y2
        m = (r1 + r2) / 2 - 1
        return Metric("Momentum", raw=m, inputs=inputs, formula=formula, unit="growth",
                      steps=[f"y0 ÷ y−1 = {cur:,.0f} ÷ {y1:,.0f} = {r1:.3f}",
                             f"y0 ÷ y−2 = {cur:,.0f} ÷ {y2:,.0f} = {r2:.3f}",
                             f"({r1:.3f} + {r2:.3f}) ÷ 2 − 1 = {m:+.1%}"])
    if fallback in ("yoy", "chain") and y1 is not None:
        m = cur / y1 - 1
        return Metric("Momentum", raw=m, inputs=inputs, formula=formula, unit="growth", basis="Assumption",
                      steps=["y−2 history not available → fallback: y−1 ratio only (assumption)",
                             f"{cur:,.0f} ÷ {y1:,.0f} − 1 = {m:+.1%}"])
    if fallback == "chain" and prev is not None and prev > 0:
        m = (cur - prev) / prev
        return Metric("Momentum", raw=m, inputs=inputs, formula=formula, unit="growth", basis="Assumption",
                      steps=["No prior-year history → fallback: sequential T3M growth, deck slide 5 (assumption)",
                             f"({cur:,.0f} − {prev:,.0f}) ÷ {prev:,.0f} = {m:+.1%}"])
    return _na("Momentum", "Needs the same T3M one and two years earlier (27 months of history).", inputs, formula)


def calculate_predictability(actuals, forecasts) -> Metric:
    """Deck slide 5: Forecast Accuracy = 1 − MAPE, MAPE = mean(|Actual − Forecast| / Actual)."""
    formula = "Predictability = 1 − MAPE,   MAPE = mean(|Actual − Forecast| ÷ Actual)"
    a = [_num(x) for x in (actuals or [])]
    f = [_num(x) for x in (forecasts or [])]
    pairs = [(x, y) for x, y in zip(a, f) if x is not None and y is not None]
    inputs = {"Periods with actual & forecast": len(pairs),
              "Actuals": [round(x, 1) for x, _ in pairs][-12:],
              "Forecasts": [round(y, 1) for _, y in pairs][-12:]}
    if not pairs:
        return _na("Predictability", "Missing forecast or actual demand.", inputs, formula)
    if any(x < 0 or y < 0 for x, y in pairs):
        return _na("Predictability", "Actual/forecast cannot be negative.", inputs, formula)
    usable = [(x, y) for x, y in pairs if x > 0]
    if not usable:
        return _na("Predictability", "Actual demand is zero - MAPE undefined.", inputs, formula)
    apes = [abs(x - y) / x for x, y in usable]
    mape = float(np.mean(apes))
    acc = max(0.0, 1.0 - mape)
    return Metric("Predictability", raw=acc, inputs=inputs, formula=formula, unit="accuracy",
                  steps=[f"MAPE over {len(usable)} period(s) = {mape:.1%}",
                         f"Forecast accuracy = 1 − {mape:.1%} = {acc:.1%}"
                         + (" (floored at 0)" if mape > 1 else "")])


def calculate_reach(active_points, total_points) -> Metric:
    """Deck slide 5: Reach = Active buying points_t ÷ Total potential buying points."""
    formula = "Reach = Active buying points ÷ Total potential buying points"
    act, tot = _num(active_points), _num(total_points)
    inputs = {"Active buying points": act, "Total potential buying points": tot}
    if act is None or tot is None:
        return _na("Reach", "Needs active and total potential buying points.", inputs, formula)
    if act < 0 or tot <= 0:
        return _na("Reach", "Buying points must be positive.", inputs, formula)
    if act > tot:
        return _na("Reach", "Active buying points exceed total potential.", inputs, formula)
    r = act / tot
    return Metric("Reach", raw=r, inputs=inputs, formula=formula, unit="share",
                  steps=[f"{act:,.0f} ÷ {tot:,.0f} = {r:.1%}"])


def calculate_position(current_volume, peak_volume) -> Metric:
    """Deck slide 5: Position = Current T3M volume ÷ Peak T3M volume (last 24 months)."""
    formula = "Position = Current T3M volume ÷ Peak T3M volume (24 months)"
    cur, peak = _num(current_volume), _num(peak_volume)
    inputs = {"Current T3M volume": cur, "Peak T3M volume": peak}
    if cur is None or peak is None:
        return _na("Position", "Needs current and peak T3M volume.", inputs, formula)
    if cur < 0 or peak <= 0:
        return _na("Position", "Volumes must be positive.", inputs, formula)
    ratio = cur / peak
    steps = [f"{cur:,.0f} ÷ {peak:,.0f} = {ratio:.2f}"]
    if ratio > 1:
        steps.append("Current exceeds recorded peak → capped at 1.00 (current is the new peak)")
        ratio = 1.0
    return Metric("Position", raw=ratio, inputs=inputs, formula=formula, unit="ratio", steps=steps)


SIGNAL_FUNCTIONS = {
    "momentum": lambda d: calculate_momentum(d.get("current_t3m"), d.get("t3m_y1"), d.get("t3m_y2"),
                                             d.get("previous_t3m"), d.get("momentum_fallback", "chain")),
    "predictability": lambda d: calculate_predictability(d.get("actuals"), d.get("forecasts")),
    "reach": lambda d: calculate_reach(d.get("active_points"), d.get("total_points")),
    "position": lambda d: calculate_position(d.get("current_volume"), d.get("peak_volume")),
}


def calculate_raw_signals(inputs: dict, settings: EngineSettings | None = None) -> dict[str, Metric]:
    inputs = {**inputs, "momentum_fallback": (settings or EngineSettings()).momentum_fallback}
    return {s: SIGNAL_FUNCTIONS[s](inputs) for s in SIGNALS}


# =============================================================================
# 5. NORMALISATION (raw signal -> 0-100 score)
# =============================================================================

# Assumption-only scales used when percentile ranking is not possible/selected.
ABSOLUTE_SCALES = {
    "momentum": (-0.50, 0.50),     # -50% growth -> 0, +50% -> 100
    "predictability": (0.0, 1.0),  # accuracy 0-100%
    "reach": (0.0, 1.0),           # 0-100% of buying points
}


def _absolute_score(signal: str, raw: float) -> float:
    lo, hi = ABSOLUTE_SCALES[signal]
    return float(np.clip((raw - lo) / (hi - lo) * 100, 0, 100))


def normalise_signals(raw_by_product: dict[str, dict[str, Metric]],
                      settings: EngineSettings) -> None:
    """Annexure A: Momentum, Predictability and Reach are scored as the SKU's
    percentile rank within the portfolio (0 = worst, 100 = best).
    Position is scored as the peak ratio × 100.
    Mutates each Metric in place, setting .value and adding steps."""
    for s in SIGNALS:
        peers = {pid: m[s].raw for pid, m in raw_by_product.items() if m[s].raw is not None}
        vals = np.array(list(peers.values()), dtype=float)
        n = len(vals)
        for pid, sigs in raw_by_product.items():
            m = sigs[s]
            if m.raw is None:
                continue
            if s == "position":
                m.value = round(m.raw * 100, 1)
                m.steps.append(f"Score = peak ratio × 100 = {m.value:.1f}  (Annexure A)")
                continue
            use_pct = settings.normalisation == "percentile" and n >= settings.min_peers
            if use_pct:
                below = float((vals < m.raw).sum())
                equal = float((vals == m.raw).sum())
                rank = below + (equal - 1) / 2
                m.value = round(rank / (n - 1) * 100, 1)
                m.steps.append(f"Portfolio percentile: {rank:g} of {n - 1} SKUs ranked below "
                               f"→ {rank:g} ÷ {n - 1} × 100 = {m.value:.1f}  (Annexure A)")
            else:
                m.value = round(_absolute_score(s, m.raw), 1)
                lo, hi = ABSOLUTE_SCALES[s]
                why = ("absolute scale selected" if settings.normalisation != "percentile"
                       else f"only {n} SKUs have this signal (< {settings.min_peers}) - percentile fallback")
                m.basis = "Assumption"
                m.steps.append(f"Absolute scale ({why}): ({m.raw:.3f} − {lo}) ÷ ({hi} − {lo}) × 100 "
                               f"= {m.value:.1f}")


# =============================================================================
# 6. LSS AND LIFECYCLE
# =============================================================================

def calculate_lss(scores: dict[str, float | None], weights: dict | None = None) -> Metric:
    """Deck slide 3/4: LSS = 35% Momentum + 25% Predictability + 20% Reach + 20% Position."""
    w = weights or LSS_WEIGHTS
    formula = "LSS = " + " + ".join(f"{SIGNAL_LABELS[s]} × {w[s]:.0%}" for s in SIGNALS)
    missing = [SIGNAL_LABELS[s] for s in SIGNALS if scores.get(s) is None]
    contributions = {s: (scores[s] * w[s] if scores.get(s) is not None else None) for s in SIGNALS}
    inputs = {SIGNAL_LABELS[s]: scores.get(s) for s in SIGNALS}
    if missing:
        m = _na("LSS", f"Missing signal(s): {', '.join(missing)}.", inputs, formula)
        m.steps = contributions
        return m
    total = sum(contributions.values())
    steps = [f"{SIGNAL_LABELS[s]}: {scores[s]:.1f} × {w[s]:.0%} = {contributions[s]:.2f}" for s in SIGNALS]
    steps.append(f"LSS = {total:.1f} / 100")
    return Metric("LSS", value=round(total, 1), inputs=inputs, formula=formula, steps=steps)


def lss_contributions(lss: Metric, scores: dict, weights: dict | None = None) -> dict:
    w = weights or LSS_WEIGHTS
    return {s: (scores[s] * w[s] if scores.get(s) is not None else None) for s in SIGNALS}


def determine_lifecycle(age_months, lss: float | None) -> dict:
    """Deck slide 3/4: Introduction is age-gated (age ≤ 9 months); otherwise
    Growth LSS ≥ 60, Maturity 40-60, Decline 20-40, Exit < 20."""
    t = LIFECYCLE_THRESHOLDS
    age = _num(age_months)
    if age is not None and age <= t["introduction_max_age_months"]:
        return {"stage": "Introduction", "gated": True,
                "reason": f"Age {age:.0f} months ≤ {t['introduction_max_age_months']} → Introduction "
                          f"(age-gated; LSS is not used to classify new launches)."}
    if lss is None:
        return {"stage": None, "gated": False,
                "reason": "N/A — insufficient data to compute LSS, so the SKU cannot be classified."}
    bands = [("Growth", t["growth_min_lss"], None), ("Maturity", t["maturity_min_lss"], t["growth_min_lss"]),
             ("Decline", t["decline_min_lss"], t["maturity_min_lss"]), ("Exit", None, t["decline_min_lss"])]
    for stage, lo, hi in bands:
        if (lo is None or lss >= lo) and (hi is None or lss < hi):
            rule = (f"LSS ≥ {lo}" if hi is None else f"LSS < {hi}" if lo is None else f"{lo} ≤ LSS < {hi}")
            age_txt = f"Age {age:.0f} months > {t['introduction_max_age_months']} (past launch gate); " if age is not None else ""
            return {"stage": stage, "gated": False, "reason": f"{age_txt}LSS {lss:.1f} satisfies {rule} → {stage}."}
    return {"stage": None, "gated": False, "reason": "Unclassified"}


def stage_boundaries_near(lss: float | None, band: float) -> list[tuple[str, float]]:
    """Which lifecycle boundaries is this LSS within `band` points of?"""
    if lss is None:
        return []
    t = LIFECYCLE_THRESHOLDS
    marks = {"Growth/Maturity": t["growth_min_lss"], "Maturity/Decline": t["maturity_min_lss"],
             "Decline/Exit": t["decline_min_lss"]}
    return [(k, v) for k, v in marks.items() if abs(lss - v) <= band]


def apply_confirmation_rule(stages: list, confirm_months: int, enabled: bool = True) -> list[dict]:
    """Slide 4: a lifecycle shift must hold for N consecutive months before the
    inventory policy changes. Returns, per month, the policy (confirmed) stage and
    any pending candidate. `None` stages (insufficient data) keep the prior policy."""
    k = confirm_months if enabled else 1
    out, confirmed, cand, count = [], None, None, 0
    for s in stages:
        if not isinstance(s, str):  # None / NaN = insufficient data
            s = None
        if s is None:
            out.append({"policy_stage": confirmed, "pending": cand, "pending_count": count})
            continue
        if confirmed is None:
            confirmed = s
        elif s == confirmed:
            cand, count = None, 0
        else:
            count = count + 1 if s == cand else 1
            cand = s
            if count >= k:
                confirmed, cand, count = s, None, 0
        out.append({"policy_stage": confirmed, "pending": cand, "pending_count": count})
    return out


# =============================================================================
# 7. INVENTORY ENGINE (Layer 2 - how much to keep)
# =============================================================================

def service_level_from_z(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def calculate_lead_time(local_share, lt_local, lt_cross=None, lt_local_std=None,
                        lt_cross_std=None) -> tuple[Metric, Metric]:
    """Lead time as a mix of local-depot and cross-city fulfilment (p = share fulfilled locally):
        L̄    = p·L_local + (1−p)·L_cross
        σ_L² = p·σ_local² + (1−p)·σ_cross² + p(1−p)·(L_cross − L_local)²
    Returns (mean Metric, std-dev Metric)."""
    f_mean = "L̄ = p·L_local + (1−p)·L_cross"
    f_std = "σ_L² = p·σ_local² + (1−p)·σ_cross² + p(1−p)·(L_cross − L_local)²"
    p, ll, lc = _num(local_share), _num(lt_local), _num(lt_cross)
    sl, sc = _num(lt_local_std), _num(lt_cross_std)
    inputs = {"p (local share)": p, "L_local (days)": ll, "L_cross-city (days)": lc,
              "σ_local (days)": sl, "σ_cross (days)": sc}
    reason = None
    if p is None:
        reason = "Needs the fraction fulfilled from the local depot (p)."
    elif not 0 <= p <= 1:
        reason = "Local share p must be between 0 and 1."
    elif p > 0 and ll is None:
        reason = "Needs local lead time."
    elif p < 1 and lc is None:
        reason = "Needs cross-city lead time (p < 1)."
    elif (p > 0 and ll <= 0) or (p < 1 and lc <= 0):
        reason = "Lead time must be greater than zero."
    elif (sl is not None and sl < 0) or (sc is not None and sc < 0):
        reason = "Lead-time variability cannot be negative."
    if reason:
        return _na("Lead time", reason, inputs, f_mean), _na("Lead-time std dev", reason, inputs, f_std)
    ll = ll if ll is not None else 0.0
    lc = lc if lc is not None else 0.0
    notes = []
    if sl is None:
        notes.append("σ_local not given → taken as 0")
    if sc is None:
        notes.append("σ_cross not given → taken as 0")
    sl, sc = sl or 0.0, sc or 0.0
    mean = p * ll + (1 - p) * lc
    t1, t2, t3 = p * sl ** 2, (1 - p) * sc ** 2, p * (1 - p) * (lc - ll) ** 2
    var = t1 + t2 + t3
    m_mean = Metric("Lead time L̄", value=mean, inputs=inputs, formula=f_mean, unit="days",
                    steps=[f"L̄ = {p:g} × {ll:g} + {1 - p:g} × {lc:g} = {mean:.2f} days"])
    m_std = Metric("Lead-time std dev σ_L", value=math.sqrt(var), inputs=inputs, formula=f_std, unit="days",
                   steps=notes + [f"p·σ_local² = {p:g} × {sl:g}² = {t1:.2f}",
                                  f"(1−p)·σ_cross² = {1 - p:g} × {sc:g}² = {t2:.2f}",
                                  f"p(1−p)·(L_cross − L_local)² = {p:g} × {1 - p:g} × ({lc:g} − {ll:g})² = {t3:.2f}",
                                  f"σ_L² = {var:.2f} → σ_L = {math.sqrt(var):.2f} days"])
    return m_mean, m_std


def calculate_demand_rate(avg_daily_demand, seasonality_index) -> Metric:
    """d̄ = average daily demand × seasonality index."""
    formula = "d̄ = average daily demand × seasonality index"
    d, si = _num(avg_daily_demand), _num(seasonality_index)
    inputs = {"Average daily demand": d, "Seasonality index": si}
    if d is None:
        return _na("d̄", "Needs average daily demand.", inputs, formula)
    if d < 0:
        return _na("d̄", "Demand cannot be negative.", inputs, formula)
    steps = []
    if si is None:
        si = 1.0
        steps.append("Seasonality index not given → 1.00 (neutral)")
    if si <= 0:
        return _na("d̄", "Seasonality index must be greater than zero.", inputs, formula)
    steps.append(f"d̄ = {d:g} × {si:g} = {d * si:,.2f} units/day")
    return Metric("d̄ (seasonal daily demand)", value=d * si, inputs=inputs, formula=formula, unit="units/day",
                  steps=steps)


def calculate_inventory(stage: str | None, avg_daily_demand, std_daily_demand, seasonality_index=1.0,
                        local_share=1.0, lt_local=None, lt_cross=None, lt_local_std=None, lt_cross_std=None,
                        z_override=None, review_period_override=None, shelf_life_days=None, unit_cost=None,
                        on_hand=None, network_locations=None, settings: EngineSettings | None = None) -> dict:
    """Final inventory formulas:
        d̄   = average daily demand × seasonality index
        L̄, σ_L from calculate_lead_time()   (local vs cross-city mix)
        SS  = Z · √( L̄·σd² + d̄²·σ_L² )
        CS  = d̄·T ÷ 2            (Introduction / Growth / Maturity; T = review period)
        CS  = √(2DS/H)  (EOQ)    (Decline)
        ROP = d̄·L̄ + SS
    Lifecycle sets Z, T, the cycle-stock method and policy overrides (Layer 1);
    demand, variability, lead times, seasonality, shelf life and cost set the quantity (Layer 2)."""
    settings = settings or EngineSettings()
    res: dict[str, Any] = {"notes": [], "policy": STAGE_POLICY.get(stage) if stage else None}
    d_m = calculate_demand_rate(avg_daily_demand, seasonality_index)
    lt_m, slt_m = calculate_lead_time(local_share, lt_local, lt_cross, lt_local_std, lt_cross_std)
    res["demand_rate"], res["lead_time"], res["lead_time_std"] = d_m, lt_m, slt_m
    base_d, sd = _num(avg_daily_demand), _num(std_daily_demand)
    zo, cost, oh = _num(z_override), _num(unit_cost), _num(on_hand)
    pol = res["policy"]

    keys = ["z", "safety_stock", "order_quantity", "cycle_stock", "reorder_level", "max_stock",
            "avg_inventory", "days_of_cover", "exposure_value", "on_hand_days", "excess_units",
            "stocking_locations", "per_location"]
    reason = None
    if pol is None and zo is None:
        reason = "Lifecycle stage unavailable, so no policy (Z, review period) can be applied."
    elif not d_m.ok:
        reason = d_m.reason
    elif sd is None:
        reason = "Needs the standard deviation of daily demand (σd)."
    elif sd < 0:
        reason = "Demand variability cannot be negative."
    elif not lt_m.ok:
        reason = lt_m.reason
    if reason:
        for k in keys:
            res[k] = _na(k, reason)
        return res
    d, lt, slt = d_m.value, lt_m.value, slt_m.value
    res["d_bar"], res["lead_time_mean"] = d, lt

    # --- Layer 1 levers
    z = zo if zo is not None else pol["z"]
    res["z"] = Metric("Z (service factor)", value=z, formula="Z set by lifecycle policy (Annexure B)"
                      if zo is None else "Z overridden for this SKU",
                      steps=[f"Z = {z:.2f} → cycle service level ≈ {service_level_from_z(z):.1%}"])
    review = _num(review_period_override)
    if review is None and pol is not None:
        review = pol["review_period_days"]
    method = pol["cycle_stock_method"] if pol else "review"

    # --- Safety stock
    var_demand = lt * sd ** 2
    var_supply = d ** 2 * slt ** 2
    sigma = math.sqrt(var_demand + var_supply)
    ss_calc = z * sigma
    supply_share = var_supply / (var_demand + var_supply) if (var_demand + var_supply) > 0 else 0.0
    ss_steps = [f"d̄ = {d:,.2f} units/day;  L̄ = {lt:.2f} days;  σ_L = {slt:.2f} days",
                f"Demand-variability term  L̄ × σd² = {lt:.2f} × {sd:g}² = {var_demand:,.0f}",
                f"Supply-variability term  d̄² × σ_L² = {d:,.2f}² × {slt:.2f}² = {var_supply:,.0f}",
                f"√({var_demand:,.0f} + {var_supply:,.0f}) = {sigma:,.1f}",
                f"SS = {z:.2f} × {sigma:,.1f} = {ss_calc:,.0f} units",
                f"Supply (lead-time) variability drives {supply_share:.0%} of the buffer variance"]
    ss = ss_calc
    if pol and pol["zero_safety_stock"]:
        ss = 0.0
        ss_steps.append(f"Policy override ({stage}): no forward safety stock → {ss_calc:,.0f} overridden to 0")
    res["safety_stock"] = Metric("Safety stock", value=round(ss), unit="units",
                                 formula="SS = Z · √( L̄·σd² + d̄²·σ_L² )",
                                 inputs={"Z": z, "d̄ (units/day)": round(d, 2), "σd (units/day)": sd,
                                         "L̄ (days)": round(lt, 2), "σ_L (days)": round(slt, 2)}, steps=ss_steps)
    res["supply_share"] = supply_share
    res["ss_computed"] = ss_calc

    # --- Order quantity & cycle stock
    q_steps, q, cs_of_q = [], None, None
    if pol and pol["zero_cycle_stock"]:
        q, cs_of_q = 0.0, (lambda x: 0.0)
        q_formula = cs_formula = "Policy: confirmed demand only → no cycle stock"
        q_steps.append(f"{stage}: replenish only against confirmed orders → cycle stock 0")
    elif method == "eoq":
        q_formula = "Q = EOQ = √(2DS/H),  D = avg daily demand × 365,  H = holding rate × unit cost"
        cs_formula = "CS = √(2DS/H)  (EOQ — Decline)"
        if cost is None or cost <= 0:
            for k in ["order_quantity", "cycle_stock", "max_stock", "avg_inventory", "days_of_cover"]:
                res[k] = _na(k, "EOQ (Decline) needs a positive unit cost.")
        else:
            D, S, H = base_d * 365, settings.ordering_cost, settings.holding_rate * cost
            q = math.sqrt(2 * D * S / H) if D > 0 else 0.0
            cs_of_q = lambda x: x
            q_steps += [f"D = {base_d:g} × 365 = {D:,.0f} units/yr;  S = ₹{S:,.0f}/order;  "
                        f"H = {settings.holding_rate:.0%} × ₹{cost:,.2f} = ₹{H:,.2f}/unit/yr",
                        f"EOQ = √(2 × {D:,.0f} × {S:,.0f} ÷ {H:,.2f}) = {q:,.0f} units"]
    else:
        q_formula = "Q = d̄·T  (T = review period)"
        cs_formula = "CS = d̄·T ÷ 2"
        if review is None:
            review = 30
            q_steps.append("No review period set → default 30 days")
        q = d * review
        cs_of_q = lambda x: x / 2
        q_steps.append(f"Q = d̄·T = {d:,.2f} × {review:g} days = {q:,.0f} units")

    if q is not None and q > 0 and settings.shelf_life_cap and _num(shelf_life_days):
        cap = d * _num(shelf_life_days) * settings.shelf_life_max_fraction - ss
        if q > cap:
            new_q = max(0.0, cap)
            q_steps.append(f"Shelf-life guardrail: SS + Q may not exceed {settings.shelf_life_max_fraction:.0%} of "
                           f"{_num(shelf_life_days):g}-day shelf life → Q capped {q:,.0f} → {new_q:,.0f}")
            res["notes"].append(f"Order quantity capped by shelf life ({_num(shelf_life_days):g} days).")
            q = new_q

    cs = None
    if q is not None:
        cs = cs_of_q(q)
        res["order_quantity"] = Metric("Order quantity", value=round(q), unit="units", formula=q_formula,
                                       steps=q_steps)
        res["cycle_stock"] = Metric("Cycle stock", value=round(cs), unit="units", formula=cs_formula,
                                    inputs={"d̄ (units/day)": round(d, 2), "T review period (days)": review,
                                            "Q": round(q)},
                                    steps=q_steps + [f"Cycle stock = {cs:,.0f} units"])

    # --- Reorder level
    lt_demand = d * lt
    rop_val = lt_demand + ss
    rop_steps = [f"Lead-time demand d̄·L̄ = {d:,.2f} × {lt:.2f} = {lt_demand:,.0f}",
                 f"ROP = {lt_demand:,.0f} + {ss:,.0f} = {rop_val:,.0f} units"]
    if pol and pol["no_rop"]:
        rop_steps.append(f"Policy override ({stage}): no proactive ROP (computed {lt_demand + ss_calc:,.0f})")
        res["reorder_level"] = Metric("Reorder level", value=None, status="na", unit="units",
                                      reason="No proactive reorder point - Exit policy (orders only).",
                                      formula="ROP = d̄·L̄ + SS", steps=rop_steps)
        res["rop_computed"] = lt_demand + ss_calc
    else:
        res["reorder_level"] = Metric("Reorder level", value=round(rop_val), unit="units",
                                      formula="ROP = d̄·L̄ + SS",
                                      inputs={"d̄": round(d, 2), "L̄": round(lt, 2), "SS": round(ss)}, steps=rop_steps)
    if q is not None:
        max_stock = ss + q
        avg_inv = ss + cs
        res["max_stock"] = Metric("Max stock (order-up-to)", value=round(max_stock), unit="units",
                                  formula="Max = SS + Q", steps=[f"{ss:,.0f} + {q:,.0f} = {max_stock:,.0f}"])
        res["avg_inventory"] = Metric("Average inventory", value=round(avg_inv), unit="units",
                                      formula="Avg inventory = SS + CS",
                                      steps=[f"{ss:,.0f} + {cs:,.0f} = {avg_inv:,.0f}"])
        res["days_of_cover"] = (Metric("Days of cover", value=round(avg_inv / d, 1), unit="days",
                                       formula="Days of cover = (SS + CS) ÷ d̄",
                                       steps=[f"{avg_inv:,.0f} ÷ {d:,.2f} = {avg_inv / d:.1f} days"])
                                if d > 0 else _na("Days of cover", "Zero demand."))
        res["exposure_value"] = (Metric("Inventory exposure", value=avg_inv * cost, unit="Rs",
                                        formula="Exposure = (SS + CS) × unit cost",
                                        steps=[f"{avg_inv:,.0f} × Rs {cost:,.2f} = Rs {avg_inv * cost:,.0f}"])
                                 if cost is not None else _na("Inventory exposure", "Unit cost not provided."))
    if oh is not None:
        res["on_hand_days"] = (Metric("On-hand cover", value=round(oh / d, 1), unit="days",
                                      formula="On-hand ÷ d̄") if d > 0 else _na("On-hand cover", "Zero demand."))
        if q is not None:
            res["excess_units"] = Metric("Excess vs policy max", value=round(oh - (ss + q)), unit="units",
                                         formula="Excess = On-hand − (SS + Q)",
                                         steps=[f"{oh:,.0f} − {ss + q:,.0f} = {oh - (ss + q):,.0f}"])
    # --- Allocation (network footprint) - basis: assumption
    locs = _num(network_locations)
    if pol and locs and locs > 0 and q is not None:
        n_loc = 1 if pol["network_coverage"] == 0 else max(1, round(locs * pol["network_coverage"]))
        res["stocking_locations"] = Metric("Stocking locations", value=n_loc, basis="Assumption",
                                           formula="Eligible locations × stage network coverage",
                                           steps=[f"{locs:g} × {pol['network_coverage']:.0%} → {n_loc} "
                                                  f"({pol['network_footprint']})"])
        res["per_location"] = Metric("Max stock per location", value=round((ss + q) / n_loc), basis="Assumption",
                                     formula="(SS + Q) ÷ stocking locations",
                                     steps=[f"{ss + q:,.0f} ÷ {n_loc} = {(ss + q) / n_loc:,.0f}"])
    for k in keys:
        res.setdefault(k, _na(k, "Not available for the given inputs."))
    res["review_period"] = review
    res["service_level"] = service_level_from_z(z)
    return res


INVENTORY_FIELDS = {  # product field -> calculate_inventory() keyword
    "avg_daily_demand": "avg_daily_demand", "std_daily_demand": "std_daily_demand",
    "seasonality_index": "seasonality_index", "local_share": "local_share",
    "lead_time_local_days": "lt_local", "lead_time_cross_days": "lt_cross",
    "lead_time_local_std": "lt_local_std", "lead_time_cross_std": "lt_cross_std",
    "z_override": "z_override", "review_period_override": "review_period_override",
    "shelf_life_days": "shelf_life_days", "unit_cost": "unit_cost", "on_hand": "on_hand",
    "network_locations": "network_locations",
}


def inventory_kwargs(product: dict, **overrides) -> dict:
    """Map a product record to calculate_inventory() arguments (single place)."""
    kw = {arg: product.get(fld) for fld, arg in INVENTORY_FIELDS.items()}
    kw.update(overrides)
    return kw


# =============================================================================
# 8. STRATEGY (Layer 1 policy + SKU-specific actions)
# =============================================================================

def generate_strategy(product: dict, signals: dict[str, Metric], lss: Metric, stage_info: dict,
                      inventory: dict) -> dict:
    """Build the recommendation from the policy stage and the SKU's own numbers."""
    stage = stage_info.get("policy_stage")
    if stage is None:
        return {"stage": None, "headline": "N/A — insufficient data",
                "actions": ["Complete the missing inputs so LSS and lifecycle can be calculated."],
                "policy": None, "drivers": {}}
    pol = STAGE_POLICY[stage]
    actions: list[str] = []
    T = ALERT_THRESHOLDS
    scores = {s: signals[s].value for s in SIGNALS}
    valid = {s: v for s, v in scores.items() if v is not None}
    drivers = {}
    if valid:
        contrib = {s: v * LSS_WEIGHTS[s] for s, v in valid.items()}
        drivers = {"strongest": max(valid, key=valid.get), "weakest": min(valid, key=valid.get),
                   "largest_contribution": max(contrib, key=contrib.get)}

    ss, rop, cyc = inventory.get("safety_stock"), inventory.get("reorder_level"), inventory.get("cycle_stock")
    mom, pred, reach, pos = (signals[s].raw for s in SIGNALS)

    # Stage-level action (slide 3 / slide 10)
    stage_actions = {
        "Introduction": "Build availability in priority launch locations and drive trial; keep a controlled launch buffer.",
        "Growth": "Increase production and expand distribution; keep buffers on rolling recent variability.",
        "Maturity": "Maintain availability, fine-tune schemes and run EOQ / MEIO replenishment.",
        "Decline": "Reduce production, limit trade schemes and consolidate supply to Pareto locations.",
        "Exit": "Run down inventory, centralise remaining stock and prepare delisting.",
    }
    actions.append(stage_actions[stage])

    if stage_info.get("pending"):
        actions.append(f"Signal indicates {stage_info['pending']} ({stage_info['pending_count']} of "
                       f"{stage_info['confirm_months']} months). Prepare the {stage_info['pending']} playbook "
                       f"but hold the {stage} policy until confirmed.")
    if stage == "Introduction" and stage_info.get("shadow_stage"):
        actions.append(f"Shadow LSS {stage_info['shadow_lss']:.1f} suggests {stage_info['shadow_stage']} "
                       f"after the launch gate - plan allocation accordingly.")
    if reach is not None and reach < T["low_reach_ratio"]:
        if stage in ("Introduction", "Growth", "Maturity"):
            actions.append(f"Reach is only {reach:.0%} of potential buying points - widen distribution "
                           f"to capture headroom.")
        else:
            actions.append(f"Reach has narrowed to {reach:.0%} - serve remaining buyers from regional hubs (Pull).")
    if pred is not None and pred < T["low_forecast_accuracy"]:
        extra = f" that is why SS is {ss.value:,.0f} units." if ss is not None and ss.ok and ss.value else ""
        actions.append(f"Forecast accuracy is {pred:.0%} - use Sense (scheme flags, tinting data) to strip "
                       f"scheme-led billing before trusting the signal;{extra}")
    if pos is not None and pos < T["low_position_ratio"] and stage not in ("Exit",):
        actions.append(f"Volume is at {pos:.0%} of its own 24-month peak - watch for saturation.")
    if mom is not None and stage == "Growth" and mom < 0:
        actions.append(f"Momentum has turned negative ({mom:+.0%}) despite a Growth score - do not over-scale.")
    share = inventory.get("supply_share")
    if share is not None and share > 0.5 and ss is not None and ss.ok and ss.value:
        actions.append(f"Supply variability drives {share:.0%} of the safety stock - improving lead-time "
                       f"reliability will release buffer faster than better forecasting.")
    exc, ohd = inventory.get("excess_units"), inventory.get("on_hand_days")
    on_hand = _num(product.get("on_hand"))
    if exc is not None and exc.ok and exc.value > 0:
        val = f" (≈ Rs {exc.value * _num(product.get('unit_cost')):,.0f})" if _num(product.get("unit_cost")) else ""
        verb = "Liquidate" if stage in ("Decline", "Exit") else "Rebalance"
        actions.append(f"{verb} {exc.value:,.0f} units{val} held above the policy maximum.")
    if rop is not None and rop.ok and on_hand is not None and on_hand < rop.value:
        actions.append(f"On-hand ({on_hand:,.0f}) is below the reorder level ({rop.value:,.0f}) - release a "
                       f"replenishment order now.")
    near = stage_boundaries_near(lss.value if lss.ok else None, T["transition_band_lss"])
    for name, v in near:
        actions.append(f"LSS {lss.value:.1f} is within {T['transition_band_lss']:g} points of the {name} boundary "
                       f"({v}) - review next month.")
    if product.get("channel") == "B2B":
        actions.append("B2B / project demand: prefer Pull from regional hubs and make-to-order where feasible.")
    for n in inventory.get("notes", []):
        actions.append(n)

    return {"stage": stage, "headline": pol["headline"], "business_action": pol["business_action"],
            "policy": pol, "actions": actions, "drivers": drivers}


# =============================================================================
# 9. ORCHESTRATION - portfolio and month-by-month engines
# =============================================================================

def validate_product(p: dict) -> list[dict]:
    """Data-quality checks. Returns [{'level','field','message'}]."""
    issues = []

    def add(level, fld, msg):
        issues.append({"level": level, "field": fld, "message": msg})

    if not str(p.get("name") or "").strip():
        add("error", "name", "Product name is required.")
    age = _num(p.get("age_months"))
    if age is None:
        add("warning", "age_months", "Age is missing - Introduction gate cannot be checked.")
    elif age < 0:
        add("error", "age_months", "Age cannot be negative.")
    nonneg = ["current_t3m", "previous_t3m", "t3m_y1", "t3m_y2", "current_volume", "peak_volume", "forecast",
              "actual", "active_points", "avg_daily_demand", "std_daily_demand", "on_hand", "unit_cost",
              "lead_time_local_std", "lead_time_cross_std", "shelf_life_days"]
    for f in nonneg:
        v = _num(p.get(f))
        if v is not None and v < 0:
            add("error", f, f"{f.replace('_', ' ').capitalize()} cannot be negative.")
    act, tot = _num(p.get("active_points")), _num(p.get("total_points"))
    if p.get("input_mode") == "history":
        df = history_frame(p.get("history"))
        if df.empty:
            add("error", "history", "Missing historical data - add monthly rows or switch to manual entry.")
        else:
            if (df["sales_volume"] < 0).any() or (df["forecast"] < 0).any():
                add("error", "history", "History contains negative sales or forecast.")
            if len(df) < DATA_RULES["min_months_momentum"]:
                add("warning", "history", f"Only {len(df)} months of history - the Momentum formula needs "
                                          f"{DATA_RULES['min_months_momentum']} (same T3M one and two years back).")
            if df["forecast"].notna().sum() < DATA_RULES["min_months_predictability"]:
                add("warning", "forecast", "Insufficient forecast history for Predictability.")
            ap = df["active_points"].dropna()
            act = float(ap.iloc[-1]) if len(ap) else None
    else:
        if _num(p.get("t3m_y1")) is None or _num(p.get("t3m_y2")) is None:
            add("warning", "t3m_y1", "Missing prior-year T3M sales (y−1 / y−2) - Momentum formula incomplete.")
        if _num(p.get("forecast")) is None:
            add("warning", "forecast", "Missing forecast - Predictability cannot be calculated.")
        if _num(p.get("actual")) is None:
            add("warning", "actual", "Missing actual demand - Predictability cannot be calculated.")
        cv, pk = _num(p.get("current_volume")), _num(p.get("peak_volume"))
        if cv is not None and pk is not None and cv > pk:
            add("warning", "peak_volume", "Current T3M volume exceeds peak - Position capped at 1.0.")
    if tot is not None and tot <= 0:
        add("error", "total_points", "Total potential buying points must be positive.")
    if act is not None and tot is not None and act > tot:
        add("error", "active_points", "Active buying points exceed total potential buying points.")
    for f, lbl in (("lead_time_local_days", "Local lead time"), ("lead_time_cross_days", "Cross-city lead time")):
        v = _num(p.get(f))
        if v is not None and v <= 0:
            add("error", f, f"{lbl} must be greater than zero.")
    share = _num(p.get("local_share"))
    if share is None:
        add("warning", "local_share", "Local fulfilment share (p) missing - lead time cannot be calculated.")
    elif not 0 <= share <= 1:
        add("error", "local_share", "Local fulfilment share (p) must be between 0 and 1.")
    elif share < 1 and _num(p.get("lead_time_cross_days")) is None:
        add("warning", "lead_time_cross_days", "Cross-city lead time missing (p < 1).")
    si = _num(p.get("seasonality_index"))
    if si is not None and si <= 0:
        add("error", "seasonality_index", "Seasonality index must be greater than zero.")
    z = _num(p.get("z_override"))
    if z is not None and not (0 < z <= 4):
        add("error", "z_override", "Z must be between 0 and 4.")
    for f in ["avg_daily_demand", "std_daily_demand", "lead_time_local_days"]:
        if _num(p.get(f)) is None:
            add("warning", f, f"{f.replace('_', ' ').capitalize()} missing - inventory cannot be calculated.")
    return issues


def _inventory_for(product: dict, stage, settings, d_override=None, sd_override=None, products=None):
    d = _num(product.get("avg_daily_demand")) if d_override is None else d_override
    sd = _num(product.get("std_daily_demand")) if sd_override is None else sd_override
    notes = []
    # Introduction with no variability history: borrow the analogue SKU's coefficient of variation
    ref_id = product.get("reference_sku_id")
    if sd is None and stage == "Introduction" and ref_id and products and ref_id in products and d is not None:
        ref = products[ref_id]
        rd, rsd = _num(ref.get("avg_daily_demand")), _num(ref.get("std_daily_demand"))
        if rd and rsd is not None:
            sd = d * rsd / rd
            notes.append(f"σd borrowed from analogue SKU '{ref.get('name')}' (CV {rsd / rd:.0%}).")
    inv = calculate_inventory(stage, **inventory_kwargs(product, avg_daily_demand=d, std_daily_demand=sd),
                              settings=settings)
    inv["notes"] = notes + inv["notes"]
    inv["inputs_used"] = {"d": d, "sd": sd}
    return inv


def evaluate_portfolio(products: dict[str, dict], settings: EngineSettings | None = None) -> dict:
    """Run the whole engine: signals → normalise → LSS → stage → validation rule
    → inventory → strategy → alerts. Returns {'results': {id: ...}, 'table': DataFrame,
    'alerts': [...], 'monthly': DataFrame}."""
    settings = settings or EngineSettings()
    results: dict[str, dict] = {}
    raw = {}
    for pid, p in products.items():
        inputs = resolve_inputs(p)
        results[pid] = {"product": p, "inputs": inputs, "issues": validate_product(p)}
        raw[pid] = calculate_raw_signals(inputs, settings)
    normalise_signals(raw, settings)

    for pid, p in products.items():
        sig = raw[pid]
        scores = {s: sig[s].value for s in SIGNALS}
        lss_full = calculate_lss(scores, settings.weights)
        life = determine_lifecycle(p.get("age_months"), lss_full.value if lss_full.ok else None)
        r = results[pid]
        r["signals"] = sig
        r["scores"] = scores
        r["contributions"] = lss_contributions(lss_full, scores, settings.weights)
        if life["gated"]:
            r["lss"] = _na("LSS", "Age-gated: Introduction SKUs are not classified by LSS.",
                           lss_full.inputs, lss_full.formula)
            r["lss"].steps = lss_full.steps if lss_full.ok else []
            r["shadow_lss"] = lss_full.value if lss_full.ok else None
            shadow = determine_lifecycle(None, r["shadow_lss"]) if r["shadow_lss"] is not None else {}
            r["shadow_stage"] = shadow.get("stage")
        else:
            r["lss"], r["shadow_lss"], r["shadow_stage"] = lss_full, None, None
        r["life"] = life

    monthly = monthly_lifecycle(products, settings)
    for pid, p in products.items():
        r = results[pid]
        hist = monthly[monthly["product_id"] == pid] if not monthly.empty else pd.DataFrame()
        prior = hist["signal_stage"].tolist()[:-1] if len(hist) else []
        seq = prior + [r["life"]["stage"]]
        conf = apply_confirmation_rule(seq, settings.confirm_months, settings.confirm_enabled)[-1]
        if r["life"]["stage"] is None:  # cannot classify this month → no policy, rather than a stale one
            conf = {"policy_stage": None, "pending": None, "pending_count": 0}
        k = settings.confirm_months if settings.confirm_enabled else 1
        r["stage_info"] = {
            "signal_stage": r["life"]["stage"], "policy_stage": conf["policy_stage"],
            "pending": conf["pending"], "pending_count": conf["pending_count"], "confirm_months": k,
            "reason": r["life"]["reason"], "has_history": len(prior) > 0,
            "previous_signal_stage": prior[-1] if prior else None,
            "shadow_lss": r["shadow_lss"], "shadow_stage": r["shadow_stage"],
        }
        r["monthly"] = hist.reset_index(drop=True)
        r["inventory"] = _inventory_for(p, conf["policy_stage"], settings, products=products)
        r["strategy"] = generate_strategy(p, r["signals"], r["lss"], r["stage_info"], r["inventory"])

    table = portfolio_table(results)
    alerts = generate_alerts(results)
    return {"results": results, "table": table, "alerts": alerts, "monthly": monthly, "settings": settings}


def monthly_lifecycle(products: dict[str, dict], settings: EngineSettings) -> pd.DataFrame:
    """Slide 4 monthly loop: for every month refresh the four signals, calculate
    LSS, classify, compare with the previous month and apply the confirmation rule."""
    frames = {pid: history_frame(p.get("history")) for pid, p in products.items()
              if p.get("input_mode") == "history"}
    frames = {k: v for k, v in frames.items() if not v.empty}
    if not frames:
        return pd.DataFrame()
    months = sorted(set().union(*[set(f["month"]) for f in frames.values()]))
    rows = []
    for m in months:
        raw, meta = {}, {}
        for pid, f in frames.items():
            idx = f.index[f["month"] == m]
            if len(idx) == 0:
                continue
            i = int(idx[0])
            p = products[pid]
            inp = derive_inputs_from_history(p["history"], upto=i)
            inp["total_points"] = _num(p.get("total_points"))
            raw[pid] = calculate_raw_signals(inp, settings)
            age = _num(p.get("age_months"))
            meta[pid] = {"age": None if age is None else age - (len(f) - 1 - i), "inp": inp,
                         "sales": f.loc[i, "sales_volume"], "forecast": f.loc[i, "forecast"]}
        normalise_signals(raw, settings)
        for pid, sig in raw.items():
            scores = {s: sig[s].value for s in SIGNALS}
            lss = calculate_lss(scores, settings.weights)
            life = determine_lifecycle(meta[pid]["age"], lss.value if lss.ok else None)
            rows.append({"product_id": pid, "month": m, "age": meta[pid]["age"],
                         "sales_volume": meta[pid]["sales"], "forecast": meta[pid]["forecast"],
                         **{s: scores[s] for s in SIGNALS},
                         "lss": None if life["gated"] or not lss.ok else lss.value,
                         "signal_stage": life["stage"]})
    df = pd.DataFrame(rows).sort_values(["product_id", "month"]).reset_index(drop=True)
    out = []
    for pid, g in df.groupby("product_id", sort=False):
        g = g.copy()
        conf = apply_confirmation_rule(g["signal_stage"].tolist(), settings.confirm_months,
                                       settings.confirm_enabled)
        g["policy_stage"] = [c["policy_stage"] for c in conf]
        g["pending"] = [c["pending"] for c in conf]
        g["prev_signal_stage"] = g["signal_stage"].shift(1)
        g["stage_changed"] = (g["signal_stage"] != g["prev_signal_stage"]) & g["prev_signal_stage"].notna()
        g["policy_changed"] = (g["policy_stage"] != g["policy_stage"].shift(1)) & g["policy_stage"].shift(1).notna()
        # inventory trend: rolling T3M demand, SKU's own CV, same lead time
        p = products[pid]
        d0, sd0 = _num(p.get("avg_daily_demand")), _num(p.get("std_daily_demand"))
        cv = (sd0 / d0) if d0 and sd0 is not None else None
        ss_list, rop_list = [], []
        for _, row in g.iterrows():
            t3 = g.loc[:_, "sales_volume"].tail(3)
            d = float(t3.mean() / 30) if len(t3) == 3 and t3.notna().all() else None
            inv = calculate_inventory(row["policy_stage"], settings=settings, **inventory_kwargs(
                p, avg_daily_demand=d, std_daily_demand=d * cv if (d is not None and cv is not None) else None,
                on_hand=None, network_locations=None))
            ss_list.append(inv["safety_stock"].value)
            rop_list.append(inv["reorder_level"].value)
        g["safety_stock"], g["reorder_level"] = ss_list, rop_list
        out.append(g)
    return pd.concat(out, ignore_index=True)


def portfolio_table(results: dict) -> pd.DataFrame:
    rows = []
    for pid, r in results.items():
        p, inv, si = r["product"], r["inventory"], r["stage_info"]
        rows.append({
            "id": pid, "Product": p.get("name"), "Category": p.get("category"), "Channel": p.get("channel"),
            "Age": _num(p.get("age_months")),
            "Momentum": r["scores"]["momentum"], "Predictability": r["scores"]["predictability"],
            "Reach": r["scores"]["reach"], "Position": r["scores"]["position"],
            "LSS": r["lss"].value if r["lss"].ok else None,
            "Lifecycle Stage": si["policy_stage"] or "N/A",
            "Signal Stage": si["signal_stage"] or "N/A",
            "Safety Stock": inv["safety_stock"].value, "Cycle Stock": inv["cycle_stock"].value,
            "ROP": inv["reorder_level"].value, "Days of Cover": inv["days_of_cover"].value,
            "Exposure (Rs)": inv["exposure_value"].value,
            "On Hand": _num(p.get("on_hand")),
            "On-hand Value (Rs)": (_num(p.get("on_hand")) * _num(p.get("unit_cost"))
                                   if _num(p.get("on_hand")) is not None and _num(p.get("unit_cost")) is not None
                                   else None),
            "Strategy": r["strategy"]["headline"], "Demo": bool(p.get("is_demo")),
        })
    return pd.DataFrame(rows, columns=PORTFOLIO_COLUMNS)


PORTFOLIO_COLUMNS = ["id", "Product", "Category", "Channel", "Age", "Momentum", "Predictability", "Reach",
                     "Position", "LSS", "Lifecycle Stage", "Signal Stage", "Safety Stock", "Cycle Stock", "ROP",
                     "Days of Cover", "Exposure (Rs)", "On Hand", "On-hand Value (Rs)", "Strategy", "Demo"]


def generate_alerts(results: dict) -> list[dict]:
    T = ALERT_THRESHOLDS
    alerts = []

    def add(sev, kind, r, msg):
        alerts.append({"Severity": sev, "Alert": kind, "Product": r["product"].get("name"),
                       "id": r["product"].get("id"), "Detail": msg})

    for pid, r in results.items():
        si, inv, sig = r["stage_info"], r["inventory"], r["signals"]
        stage = si["policy_stage"]
        lss = r["lss"].value if r["lss"].ok else None
        exc = inv.get("excess_units")
        high_inv = exc is not None and exc.ok and exc.value > 0
        if stage == "Decline" and high_inv:
            add("High", "High inventory + Decline", r,
                f"{exc.value:,.0f} units above policy max in a Decline SKU.")
        if lss is not None and lss < T["low_lss"] and high_inv and stage != "Exit":
            add("High", "Low LSS + high inventory", r, f"LSS {lss:.1f} with {exc.value:,.0f} excess units.")
        if stage == "Exit" and _num(r["product"].get("on_hand")):
            add("High", "Exit stock to liquidate", r, f"{_num(r['product']['on_hand']):,.0f} units on hand.")
        if stage == "Introduction":
            age = _num(r["product"].get("age_months")) or 0
            left = LIFECYCLE_THRESHOLDS["introduction_max_age_months"] - age
            extra = f" Shadow LSS {si['shadow_lss']:.1f} → {si['shadow_stage']}." if si.get("shadow_lss") else ""
            add("Info", "Introduction product", r, f"{left:.0f} month(s) left in launch gate.{extra}")
            if left < T["intro_graduation_window"]:
                add("Medium", "Potential lifecycle transition", r,
                    f"Graduates from Introduction within {T['intro_graduation_window']} months.{extra}")
        if sig["reach"].raw is not None and sig["reach"].raw < T["low_reach_ratio"]:
            add("Medium", "Low reach", r, f"Only {sig['reach'].raw:.0%} of potential buying points active.")
        if sig["predictability"].raw is not None and sig["predictability"].raw < T["low_forecast_accuracy"]:
            add("Medium", "Low predictability", r, f"Forecast accuracy {sig['predictability'].raw:.0%}.")
        falling = sig["position"].raw is not None and sig["position"].raw < T["low_position_ratio"]
        mh = r.get("monthly")
        drop_txt = ""
        if mh is not None and len(mh) >= 4 and pd.notna(mh["position"].iloc[-1]) and pd.notna(mh["position"].iloc[-4]):
            drop = mh["position"].iloc[-4] - mh["position"].iloc[-1]
            if drop >= T["position_drop_points"]:
                falling, drop_txt = True, f" Down {drop:.0f} points in 3 months."
        if falling:
            pr = sig["position"].raw
            add("Medium", "Falling position", r,
                (f"At {pr:.0%} of 24-month peak." if pr is not None else "") + drop_txt)
        if si.get("pending"):
            add("Medium", "Potential lifecycle transition", r,
                f"Signal says {si['pending']} ({si['pending_count']}/{si['confirm_months']} months) - "
                f"policy held at {stage}.")
        for name, v in stage_boundaries_near(lss, T["transition_band_lss"]):
            add("Low", "Potential lifecycle transition", r, f"LSS {lss:.1f} within {T['transition_band_lss']:g} "
                                                            f"pts of {name} boundary ({v}).")
        prev = si.get("previous_signal_stage")
        if prev and si["signal_stage"] and prev != si["signal_stage"]:
            add("High" if si["signal_stage"] in ("Decline", "Exit") else "Medium", "Lifecycle change", r,
                f"{prev} → {si['signal_stage']} this month.")
        rop, oh = inv.get("reorder_level"), _num(r["product"].get("on_hand"))
        if rop is not None and rop.ok and oh is not None and oh < rop.value:
            add("Medium", "Below reorder level", r, f"On-hand {oh:,.0f} < ROP {rop.value:,.0f}.")
        if any(i["level"] == "error" for i in r["issues"]):
            add("High", "Data validation", r, "; ".join(i["message"] for i in r["issues"] if i["level"] == "error"))
    order = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}
    return sorted(alerts, key=lambda a: (order[a["Severity"]], a["Product"] or ""))


def project_inventory_profile(inventory: dict, on_hand=None, days: int | None = None) -> pd.DataFrame:
    """Deterministic projection of on-hand stock under the calculated policy
    (constant demand d̄; order Q when inventory position hits ROP; arrives after L̄)."""
    d, lt = inventory.get("d_bar"), inventory.get("lead_time_mean")
    ss_m, q_m, rop_m = inventory.get("safety_stock"), inventory.get("order_quantity"), inventory.get("reorder_level")
    if d is None or lt is None or d <= 0 or ss_m is None or not ss_m.ok:
        return pd.DataFrame()
    ss = ss_m.value
    q = q_m.value if q_m is not None and q_m.ok else 0
    rop = rop_m.value if rop_m is not None and rop_m.ok else None
    level = _num(on_hand) if _num(on_hand) is not None else ss + q
    if days is None:
        days = int(min(365, max(60, 3 * (q / d if q else 0) + 2 * lt)))
    pipeline, rows = [], []
    for day in range(days + 1):
        arrived = sum(qty for t, qty in pipeline if t == day)
        pipeline = [(t, qty) for t, qty in pipeline if t != day]
        if arrived:
            rows.append({"day": day, "on_hand": level, "arrival": 0.0})  # just before receipt
        level += arrived
        rows.append({"day": day, "on_hand": level, "arrival": arrived})
        position = level + sum(qty for _, qty in pipeline)
        if rop is not None and q > 0 and position <= rop:
            pipeline.append((day + int(round(lt)), q))
        level = max(0.0, level - d)
    return pd.DataFrame(rows)


def inventory_by_stage(product: dict, settings: EngineSettings | None = None) -> pd.DataFrame:
    """Same SKU economics under each lifecycle policy - shows Layer 1 vs Layer 2."""
    rows = []
    for stage in STAGES:
        inv = calculate_inventory(stage, settings=settings, **inventory_kwargs(
            product, z_override=None, review_period_override=None, on_hand=None, network_locations=None))
        rows.append({"Policy stage": stage, "Z": inv["z"].value,
                     "Service level": inv.get("service_level"),
                     "Review (days)": inv.get("review_period"),
                     "Safety stock": inv["safety_stock"].value, "Cycle stock": inv["cycle_stock"].value,
                     "ROP": inv["reorder_level"].value, "Days of cover": inv["days_of_cover"].value})
    return pd.DataFrame(rows)
