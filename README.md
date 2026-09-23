---
title: Asian Paints Lifecycle & Inventory
emoji: 🎨
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Product lifecycle (LSS) and inventory decision system
---

# Asian Paints — Product Lifecycle & Inventory Decision System

A Streamlit decision system built on the *Breaker of Chains* deck (IIM Bangalore):
**Lifecycle chooses the playbook, SKU economics choose the stock.**

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest -q tests      # 44 engine + UI tests
```

## Deploy

**Streamlit Community Cloud:** push this folder to GitHub → share.streamlit.io → *Create app* →
pick the repo, branch `main`, main file `app.py` → Advanced settings: Python 3.12 → Deploy.

**Hugging Face Spaces:** create a Space with SDK **Docker** (blank template), then push this repo to it.
The YAML header above and the `Dockerfile` configure the Space.

## Decision flow

`Product → inputs → Momentum / Predictability / Reach / Position → LSS → lifecycle stage
→ 2-month confirmation → Safety stock / Cycle stock / ROP → strategy → portfolio & alerts`

| Tab | What it does |
|---|---|
| LSS Calculator | Add / edit / delete products; the 4 signals, weights, contributions, LSS, stage and *why*; "View calculation" on every metric |
| Inventory Calculator | Layer 1 policy envelope + Layer 2 SKU inputs (editable); SS, cycle stock, ROP, max stock, days of cover, exposure, allocation; projected stock profile; same SKU under all five policies |
| Strategy | Auto-generated recommendation, policy envelope, SKU-specific actions, 7-question decision summary |
| Portfolio | All SKUs; search, lifecycle/category filters, sort, row selection → edit/delete, CSV download |
| Lifecycle Dashboard | KPI cards, 5 charts + monthly stage mix, alerts, this month's stage movements |
| Product Detail | Overview, LSS breakdown, inventory, strategy, sales / LSS / lifecycle / inventory trends, month-by-month table |

## Files

| File | Role |
|---|---|
| `calculations.py` | **The only place business formulas live.** Config (weights, thresholds, stage policy, alert thresholds) at the top. |
| `app.py` | Streamlit UI; calls `calculations.evaluate_portfolio()` and renders its output |
| `components.py` | Styling, cards, badges, chart helpers (no formulas) |
| `demo_data.py` | Illustrative demo SKUs (Annexure A + slide-3 examples) — deletable |
| `tests/` | Engine tests and `streamlit.testing` UI tests |

## Changing a formula

Edit only the function; everything else (tables, charts, alerts, "View calculation" text) follows.

| Change | Edit |
|---|---|
| Momentum | `calculate_momentum()` |
| Seasonal indices for deseasonalising | `estimate_seasonal_indices()` |
| Predictability | `calculate_predictability()` |
| Reach | `calculate_reach()` |
| Position | `calculate_position()` |
| Raw signal → 0–100 score | `normalise_signals()` |
| LSS / weights | `calculate_lss()` / `LSS_WEIGHTS` |
| Stage thresholds | `determine_lifecycle()` / `LIFECYCLE_THRESHOLDS` |
| Confirmation rule | `apply_confirmation_rule()` (months configurable in the sidebar) |
| Safety stock, cycle stock, ROP | `calculate_inventory()` |
| Stage policy (Z, review period, overrides) | `STAGE_POLICY` |
| Recommendations / alerts | `generate_strategy()` / `generate_alerts()` |

Each function returns a `Metric` carrying its own formula text, inputs and working steps, so the UI's explanation always matches the code that ran. Setting `status="pending"` on a Metric renders **"Formula pending"**.

## Formulas and where they come from

**Final formulas (supplied 23 Sep 2026; Momentum revised the same day)**

| Item | Formula |
|---|---|
| Momentum | [ (S′_Qt ÷ S′_Qt−1) + (S′_Qt ÷ S′_Qt−4) ] ÷ 2 − 1 — average of QoQ and YoY growth on deseasonalised sales; S′ = sales ÷ seasonal index; Q_t = latest 3 months, Q_t−1 = the 3 before, Q_t−4 = same quarter last year |
| Seasonal demand | d̄ = average daily demand × seasonality index |
| Lead time (mean) | L̄ = p·L_local + (1−p)·L_cross-city, p = share fulfilled from the local depot |
| Lead time (variance) | σ_L² = p·σ_local² + (1−p)·σ_cross² + p(1−p)·(L_cross − L_local)² |
| Safety stock | SS = Z·√(L̄·σd² + d̄²·σ_L²) |
| Cycle stock — Introduction / Growth / Maturity | CS = d̄·T ÷ 2 (T = review period: 7 / 14 / 30 days) |
| Cycle stock — Decline | CS = √(2DS/H) (EOQ); D = avg daily demand × 365, S = ordering cost, H = holding rate × unit cost |
| Reorder level | ROP = d̄·L̄ + SS |

**From the deck**

| Item | Formula | Source |
|---|---|---|
| Predictability | 1 − MAPE, MAPE = mean(\|A − F\| ÷ A), last 12 months | Slide 5 |
| Reach | Active buying points ÷ Total potential buying points | Slide 5 |
| Position | Current T3M volume ÷ Peak T3M volume (24 months) | Slide 5 |
| Scores 0–100 | Momentum / Predictability / Reach = portfolio percentile rank; Position = peak ratio × 100 | Annexure A |
| LSS | 35% M + 25% P + 20% R + 20% Pos | Slides 3–4 |
| Stage | Intro age ≤ 9 (age-gated); Growth ≥ 60; Maturity 40–60; Decline 20–40; Exit < 20 | Slides 3–4 |
| Z by stage | 1.65 / 1.75 / 1.53 / 1.16 / 0.94 | Annexure B |
| Exit overrides | SS → 0, cycle stock → 0, no proactive ROP (no Exit formula was supplied) | Annexure B |

With a single fulfilment source (p = 1) and seasonality 1.0, the engine still reproduces Annexure B for
Introduction / Growth / Maturity (tested). Decline cycle stock now follows EOQ instead of Annexure B's 6,300.

### Assumptions (labelled "Assumption" in the UI, all configurable)

- **Seasonal indices** (the formula doesn't say how to deseasonalise). `estimate_seasonal_indices()` uses
  classical multiplicative decomposition: each month's sales ÷ its centred 2×12 moving average, averaged per
  calendar month and normalised to 1.0. Ratios are pooled across SKUs with 24+ months of history, and each
  SKU uses its **category** indices, else its **channel**'s, else the portfolio's. A quarter's index is
  Σ sales ÷ Σ(sales ÷ monthly index). Indices are estimated once from full history, so the month-by-month
  replay uses slightly forward-looking indices. Manual-entry SKUs take seasonal indices typed in (blank = 1.0).
- **Momentum for SKUs under 15 months** (no same quarter last year). The sidebar picks *QoQ term only*
  (default) or *strict* (N/A → SKU unclassified).
- **Position is not deseasonalised** (deck formula unchanged), so a SKU in its seasonal low quarter sits
  further from its peak.
- **EOQ annual demand D** uses the unadjusted average daily demand × 365 (a seasonal peak is not annualised).
- **Decline CS** is taken literally as √(2DS/H); the textbook average cycle stock would be EOQ ÷ 2.
- Missing seasonality index → 1.0; missing σ_local / σ_cross → 0 (both noted in the working).

- **Absolute normalisation** (sidebar option, and automatic fallback when fewer than 5 SKUs have a signal): momentum −50%…+50% → 0…100; accuracy and reach × 100.
- **Network coverage by stage** (20% / 60% / 100% / 40% / 1 central location) for stocking locations and per-location allocation.
- **Shelf-life guardrail**: SS + Q may not exceed 50% of shelf life in days of demand.
- **Introduction σd from an analogue SKU**: if σd is blank, the reference SKU's coefficient of variation is used.
- **Inventory trend** (Product Detail) uses rolling T3M demand with the SKU's current CV and lead time.
- **Alert thresholds** in `ALERT_THRESHOLDS`.

### Note on Annexure A

Annexure A labels *Enamel Classic 1L* (LSS 64.3), *Adhesive StrongBond Classic* (65.9) and
*Wood Primer StainBlock* (65.4) as **Maturity**, but the deck's own threshold (LSS ≥ 60 → Growth)
puts them in Growth. The engine applies the thresholds, so they show as Growth. The slide-3 examples
(Royale Glitz, Apex Ultima Protek, Emporio, Nilaya) were added so every stage appears in the demo.
*Emporio* shows the 2-month rule: its signal dropped to Decline this month, so the policy stays at Maturity.

## Demo data

All 24 demo SKUs are **illustrative**. Their monthly histories are synthetic, generated so their signals rank
like Annexure A. Nothing in `demo_data.py` stores a score, stage or stock number; the engine calculates
everything. Use **Remove demo** in the sidebar to start from an empty portfolio.

## Data validation

Negative sales/demand, active > potential buying points, lead time ≤ 0, Z outside (0, 4], and missing
forecast, actuals or history are all flagged. A calculation that can't run shows **"N/A — insufficient
data"**, never a misleading zero.
