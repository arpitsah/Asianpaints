"""
demo_data.py - ILLUSTRATIVE demo SKUs (not the actual business portfolio).

The 20 SKUs mirror Annexure A of the deck. Their monthly history is synthetic:
it is generated so that each SKU's raw signals rank in the same order as the
Annexure A percentiles. All scoring is done live by calculations.py - nothing
here stores an LSS, stage, safety stock or recommendation.

Users can delete every demo SKU and enter their own products.
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

AS_OF = "2026-08"
HISTORY_MONTHS = 36

# name, category, age (months), Momentum %ile, Predictability %ile, Distribution %ile,
# peak ratio  - Annexure A inputs
ANNEXURE_A = [
    ("Anti-Fungal Coat LaunchSKU", "Decorative Paints", 5, 100.0, 52.6, 100.0, 1.00),
    ("Premium Wood Care New Range", "Wood Finishes", 7, 89.4, 52.6, 89.4, 0.95),
    ("Fungal-Shield Primer Advanced", "Construction Chemicals", 8, 94.7, 52.6, 94.7, 0.97),
    ("Waterproofing Sealant XT", "Waterproofing", 21, 68.4, 52.6, 68.4, 0.98),
    ("Interior Emulsion Premium+", "Decorative Paints", 18, 84.2, 73.6, 84.2, 0.99),
    ("PU Wood Finish Nova", "Wood Finishes", 15, 78.9, 47.3, 78.9, 0.94),
    ("Exterior Primer FlexCoat", "Construction Chemicals", 12, 73.6, 42.1, 73.6, 0.97),
    ("Exterior Emulsion ApexShield", "Decorative Paints", 62, 63.1, 100.0, 63.1, 0.92),
    ("Enamel Classic 1L", "Decorative Paints", 91, 42.1, 94.7, 42.1, 0.88),
    ("Wall Putty SmoothCoat", "Wall Putty", 77, 57.8, 89.4, 57.8, 0.90),
    ("Adhesive StrongBond Classic", "Adhesives", 98, 47.3, 84.2, 52.6, 0.89),
    ("Wood Primer StainBlock", "Wood Finishes", 83, 52.6, 78.9, 47.3, 0.89),
    ("Distemper EconoWhite", "Decorative Paints", 139, 36.8, 36.8, 36.8, 0.46),
    ("Adhesive ClearBond 500g", "Adhesives", 124, 26.3, 21.0, 26.3, 0.56),
    ("Metal Primer RedOxide", "Construction Chemicals", 146, 21.0, 31.5, 31.5, 0.52),
    ("Textured Base Coat Classic", "Decorative Paints", 155, 31.5, 26.3, 21.0, 0.54),
    ("Wood Finish Legacy Varnish", "Wood Finishes", 211, 10.5, 10.5, 15.7, 0.19),
    ("Textured Coating OldGen", "Decorative Paints", 219, 5.2, 5.2, 5.2, 0.23),
    ("Construction Chemical PhaseOutMix", "Construction Chemicals", 225, 0.0, 0.0, 0.0, 0.14),
    ("Adhesive DiscoGrip Legacy", "Adhesives", 198, 15.7, 15.7, 10.5, 0.17),
]

# Slide 3 "real examples" (illustrative). Extra fields: explicit T3M growth, series mode.
DECK_EXAMPLES = [
    ("Royale Glitz Interior Emulsion", "Decorative Paints", 14, 80.0, 60.0, 55.0, 1.00, 0.28, "normal"),
    ("Apex Ultima Protek Exterior Emulsion", "Decorative Paints", 36, 60.0, 45.0, 35.0, 0.62, 0.02, "normal"),
    ("Emporio Wood Finish PU", "Wood Finishes", 62, 20.0, 35.0, 30.0, 1.00, -0.18, "late_drop"),
    ("Nilaya Architectural Coatings", "Architectural Coatings", 8, 90.0, 50.0, 40.0, 1.00, None, "normal"),
]

# Illustrative seasonality index for the coming period (d̄ = avg daily demand × index).
SEASONALITY = {"Waterproofing": 1.20, "Construction Chemicals": 1.10, "Decorative Paints": 1.05}

# Illustrative lead times: local depot = economics table value, cross-city = +3 days (slide 2: 1 day local,
# 3-5 days from another city); B2C 85% local, B2B 65% local.
B2B_CATEGORIES = {"Construction Chemicals", "Waterproofing", "Architectural Coatings"}

# Illustrative SKU economics (Layer 2). monthly volume, unit cost Rs, lead time, sigma LT,
# shelf life (days), on-hand multiple of monthly volume.
ECONOMICS = {
    "Anti-Fungal Coat LaunchSKU": (1_200, 420, 10, 3.0, 365, 0.6),
    "Premium Wood Care New Range": (900, 520, 10, 3.0, 365, 0.8),
    "Fungal-Shield Primer Advanced": (1_500, 310, 12, 3.0, 540, 0.3),
    "Waterproofing Sealant XT": (5_200, 380, 8, 2.0, 540, 0.5),
    "Interior Emulsion Premium+": (6_600, 290, 8, 2.0, 730, 0.2),
    "PU Wood Finish Nova": (3_800, 460, 14, 3.5, 365, 0.7),
    "Exterior Primer FlexCoat": (4_200, 240, 8, 2.0, 540, 0.9),
    "Exterior Emulsion ApexShield": (15_600, 330, 6, 1.5, 730, 0.9),
    "Enamel Classic 1L": (12_400, 210, 6, 1.5, 730, 0.8),
    "Wall Putty SmoothCoat": (14_800, 55, 5, 1.0, 270, 1.0),
    "Adhesive StrongBond Classic": (9_300, 140, 7, 1.5, 365, 0.7),
    "Wood Primer StainBlock": (7_900, 180, 7, 1.5, 540, 0.9),
    "Distemper EconoWhite": (4_200, 70, 9, 2.5, 540, 3.2),
    "Adhesive ClearBond 500g": (3_300, 120, 9, 2.5, 365, 2.6),
    "Metal Primer RedOxide": (3_900, 160, 9, 2.5, 540, 1.1),
    "Textured Base Coat Classic": (3_600, 240, 10, 2.5, 540, 1.8),
    "Wood Finish Legacy Varnish": (750, 350, 12, 4.0, 365, 2.4),
    "Textured Coating OldGen": (600, 280, 12, 4.0, 365, 3.0),
    "Construction Chemical PhaseOutMix": (450, 190, 12, 4.0, 365, 4.0),
    "Adhesive DiscoGrip Legacy": (800, 110, 12, 4.0, 365, 1.5),
    "Royale Glitz Interior Emulsion": (5_900, 430, 8, 2.0, 730, 0.4),
    "Apex Ultima Protek Exterior Emulsion": (11_000, 360, 6, 1.5, 730, 1.0),
    "Emporio Wood Finish PU": (2_600, 540, 10, 2.5, 365, 2.2),
    "Nilaya Architectural Coatings": (700, 650, 12, 3.0, 540, 0.5),
}


def _months(n: int) -> list[str]:
    end = pd.Period(AS_OF, freq="M")
    return [str(end - (n - 1 - i)) for i in range(n)]


def _series(age: int, m_pct: float, peak_ratio: float, growth: float | None = None, mode: str = "normal"):
    """Monthly volume index (last month = 1.0) over up to 36 months, built so the SKU's
    year-on-year momentum and 24-month peak ratio follow the Annexure A ordering."""
    h = min(age, HISTORY_MONTHS)
    if age <= 9:  # launch ramp
        x = np.arange(1, h + 1) / h
        return list(x ** 0.9)
    g = growth if growth is not None else -0.35 + 0.006 * m_pct  # momentum, monotonic in Momentum %ile
    if mode == "late_drop":  # slow erosion, then a sharp one-month fall (shows 2-month validation)
        v = np.linspace(1.12, 1.0, 36)
        v[35] = 3 * (1 + g) * v[21:24].mean() - v[33:35].sum()
        return list(v[-h:])
    if age < 27:  # too young for y-2: steady compounding ramp
        months_back = 3 if age < 15 else 12   # sequential fallback vs y-1 fallback
        r = (1 + g) ** (1 / months_back)
        return list(r ** (np.arange(h) - (h - 1)))
    # 36 months: y-2 window idx 9-11, y-1 window 21-23, current 33-35, peak within the last 24 (12-35)
    c, Y, P = 1.0, 1.0 / (1 + g), 1.0 / peak_ratio
    kp = 27 if peak_ratio >= 0.93 else (16 if peak_ratio >= 0.8 else 13)
    anchors = {0: Y, 11: Y, kp - 1: P, kp + 1: P, 21: Y, 23: Y, 33: c, 35: c}
    if kp < 21:
        anchors.update({12: min(Y, P)})
    xs = sorted(anchors)
    v = np.interp(np.arange(36), xs, [anchors[x] for x in xs])
    v[kp - 1:kp + 2] = P
    v[21:24], v[9:12], v[33:36] = Y, Y, c
    return list(v[-h:])


def _product(row, idx: int) -> dict:
    name, cat, age, m_pct, p_pct, r_pct, peak, *extra = row
    growth, mode = extra if extra else (None, "normal")
    vol, cost, lt, slt, shelf, oh_mult = ECONOMICS[name]
    rng = np.random.default_rng(idx + 7)
    b2b = cat in B2B_CATEGORIES
    total = 900 if b2b else 12_000
    shape = _series(age, m_pct, peak, growth, mode)
    months = _months(len(shape))
    mape = 0.05 + 0.45 * (1 - p_pct / 100)           # monotonic in Predictability %ile
    reach = 0.08 + 0.80 * r_pct / 100                 # monotonic in Distribution %ile
    late = peak < 0.6                                 # Decline / Exit: used to be predictable
    hist = []
    for i, (m, s) in enumerate(zip(months, shape)):
        actual = round(vol * s)
        e = mape if (not late or i >= len(shape) - 12) else 0.10
        sign = 1 if rng.random() < 0.5 else -1
        forecast = round(actual * (1 + sign * e))
        active = int(min(0.95 * total, total * reach * (s ** 0.6)))
        hist.append({"month": m, "sales_volume": float(actual), "forecast": float(forecast),
                     "active_points": float(active)})
    d = round(np.mean([h["sales_volume"] for h in hist[-3:]]) / 30, 1)
    cv = 0.12 + 0.55 * (1 - p_pct / 100)
    season = SEASONALITY.get(cat, 1.0)
    return {
        "id": uuid.uuid4().hex[:8], "name": name, "category": cat, "channel": "B2B" if b2b else "B2C",
        "age_months": age, "is_demo": True, "input_mode": "history", "history": hist,
        "total_points": total,
        # manual fields (used only if the user switches the SKU to manual entry)
        "current_t3m": None, "previous_t3m": None, "t3m_y1": None, "t3m_y2": None,
        "current_volume": None, "peak_volume": None, "forecast": None, "actual": None, "active_points": None,
        # inventory - Layer 2
        "avg_daily_demand": d, "std_daily_demand": round(d * cv, 1), "seasonality_index": season,
        "local_share": 0.65 if b2b else 0.85,
        "lead_time_local_days": lt, "lead_time_cross_days": lt + 3,
        "lead_time_local_std": slt, "lead_time_cross_std": slt + 1.0,
        "z_override": None, "review_period_override": None,
        "shelf_life_days": shelf, "unit_cost": cost, "on_hand": round(vol * oh_mult),
        "network_locations": 40 if b2b else 160, "reference_sku_id": None,
    }


def demo_products() -> dict[str, dict]:
    prods = [_product(r, i) for i, r in enumerate(ANNEXURE_A + DECK_EXAMPLES)]
    by_name = {p["name"]: p for p in prods}
    # Show the analogue-SKU mechanism: the newest launch borrows variability from a Growth SKU.
    launch = by_name["Anti-Fungal Coat LaunchSKU"]
    launch["std_daily_demand"] = None
    launch["reference_sku_id"] = by_name["Interior Emulsion Premium+"]["id"]
    return {p["id"]: p for p in prods}


def blank_product() -> dict:
    return {
        "id": uuid.uuid4().hex[:8], "name": "", "category": "Decorative Paints", "channel": "B2C",
        "age_months": 12, "is_demo": False, "input_mode": "manual", "history": [],
        "current_t3m": None, "previous_t3m": None, "t3m_y1": None, "t3m_y2": None,
        "current_volume": None, "peak_volume": None,
        "forecast": None, "actual": None, "active_points": None, "total_points": None,
        "avg_daily_demand": None, "std_daily_demand": None, "seasonality_index": 1.0, "local_share": 1.0,
        "lead_time_local_days": None, "lead_time_cross_days": None, "lead_time_local_std": 0.0,
        "lead_time_cross_std": 0.0, "z_override": None, "review_period_override": None,
        "shelf_life_days": None, "unit_cost": None, "on_hand": None, "network_locations": 160,
        "reference_sku_id": None,
    }


CATEGORIES = ["Decorative Paints", "Wood Finishes", "Construction Chemicals", "Waterproofing",
              "Wall Putty", "Adhesives", "Architectural Coatings", "Industrial Coatings", "Other"]
