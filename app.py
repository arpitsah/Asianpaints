"""
app.py - Streamlit UI for the Asian Paints Product Lifecycle & Inventory Decision System.

Run:  streamlit run app.py

UI only. Every score, stage, stock number and recommendation is produced by
calculations.py (evaluate_portfolio and the calculate_* functions it calls).
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calculations as calc
import components as ui
import demo_data

st.set_page_config(page_title="Lifecycle & Inventory Decision System", page_icon="🎨", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

SS = st.session_state
PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# =============================================================================
# State & engine
# =============================================================================
def init_state():
    if "products" not in SS:
        SS.products = demo_data.demo_products()
        SS.selected_id = next((k for k, v in SS.products.items() if v["name"].startswith("Royale Glitz")),
                              next(iter(SS.products), None))
    SS.setdefault("flash", None)
    for p in ("calc", "pf"):
        SS.setdefault(f"{p}_form", None)
        SS.setdefault(f"{p}_del", None)


def current_settings() -> calc.EngineSettings:
    return calc.EngineSettings(
        normalisation=SS.get("set_norm", "percentile"),
        confirm_enabled=SS.get("set_confirm", True),
        confirm_months=int(SS.get("set_confirm_months", 2)),
        momentum_fallback=SS.get("set_mom_fb2", "qoq"),
        ordering_cost=float(SS.get("set_order_cost", 1000.0)),
        holding_rate=float(SS.get("set_hold", 20.0)) / 100,
        shelf_life_cap=SS.get("set_shelf", True),
    )


def run_engine() -> dict:
    """Recompute whenever products or settings change (memoised on their content)."""
    settings = current_settings()
    sig = json.dumps(SS.products, sort_keys=True, default=str) + repr(settings)
    if SS.get("_engine_sig") != sig:
        SS._engine_out = calc.evaluate_portfolio(SS.products, settings)
        SS._engine_sig = sig
    return SS._engine_out


def flash(msg: str):
    SS.flash = msg


def product_name(pid) -> str:
    p = SS.products.get(pid)
    if not p:
        return "—"
    return p["name"] + ("  · demo" if p.get("is_demo") else "")


def fmt_raw(signal: str, raw) -> str:
    if raw is None:
        return ui.NA_TEXT
    return {"momentum": f"{raw:+.1%} momentum", "predictability": f"{raw:.1%} forecast accuracy",
            "reach": f"{raw:.1%} of buying points", "position": f"{raw:.2f} of peak"}[signal]


# =============================================================================
# Product add / edit / delete
# =============================================================================
def _clean_history(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    df = df.copy()
    df = df[df["month"].astype(str).str.match(r"^\d{4}-\d{2}$", na=False)]
    rows = []
    for _, r in df.iterrows():
        rows.append({"month": str(r["month"])[:7],
                     **{c: (None if pd.isna(r[c]) else float(r[c]))
                        for c in ["sales_volume", "forecast", "active_points"]}})
    return rows


def _num_in(label, value, key, help=None, step=1.0, fmt=None):
    v = None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)
    return st.number_input(label, value=v, step=step, key=key, help=help, placeholder="—", format=fmt)


def inventory_inputs(product: dict, k: str) -> dict:
    """Layer 2 input widgets (shared by the product form and the Inventory tab)."""
    v = {}
    c = st.columns(5)
    v["avg_daily_demand"] = _num_in_col(c[0], "Avg daily demand", product, "avg_daily_demand", k)
    v["seasonality_index"] = _num_in_col(c[1], "Seasonality index", product, "seasonality_index", k, 0.05)
    v["std_daily_demand"] = _num_in_col(c[2], "Demand std dev σd", product, "std_daily_demand", k)
    v["z_override"] = _num_in_col(c[3], "Z override (blank = policy)", product, "z_override", k, 0.05)
    v["review_period_override"] = _num_in_col(c[4], "Review period T override (days)", product,
                                              "review_period_override", k)
    c = st.columns(5)
    v["local_share"] = _num_in_col(c[0], "Local fulfilment share p (0–1)", product, "local_share", k, 0.05)
    v["lead_time_local_days"] = _num_in_col(c[1], "Local lead time (days)", product, "lead_time_local_days", k)
    v["lead_time_cross_days"] = _num_in_col(c[2], "Cross-city lead time (days)", product,
                                            "lead_time_cross_days", k)
    v["lead_time_local_std"] = _num_in_col(c[3], "σ local lead time", product, "lead_time_local_std", k, 0.5)
    v["lead_time_cross_std"] = _num_in_col(c[4], "σ cross-city lead time", product, "lead_time_cross_std", k, 0.5)
    c = st.columns(5)
    v["shelf_life_days"] = _num_in_col(c[0], "Shelf life (days)", product, "shelf_life_days", k)
    v["unit_cost"] = _num_in_col(c[1], "Unit cost (₹)", product, "unit_cost", k)
    v["on_hand"] = _num_in_col(c[2], "Current on-hand (units)", product, "on_hand", k)
    v["network_locations"] = _num_in_col(c[3], "Eligible depots / hubs", product, "network_locations", k)
    return v


def product_form(prefix: str, product: dict, is_new: bool):
    pid = product["id"]
    k = f"{prefix}_{pid}"
    with st.container(border=True):
        st.markdown(f"#### {'Add product' if is_new else 'Edit product — ' + ui.esc(product['name'])}")
        mode = st.radio("Demand data entry", ["history", "manual"], horizontal=True, key=f"{k}_mode",
                        index=0 if product.get("input_mode") == "history" else 1,
                        format_func=lambda m: "Monthly history (signals derived automatically)" if m == "history"
                        else "Manual snapshot (enter T3M values directly)")
        with st.form(f"{k}_form", border=False):
            st.markdown("**Product information**")
            c1, c2, c3, c4 = st.columns([2.2, 1.4, 0.8, 0.8])
            name = c1.text_input("Product name *", value=product.get("name", ""), key=f"{k}_name")
            cats = demo_data.CATEGORIES
            cat = c2.selectbox("Category", cats, key=f"{k}_cat",
                               index=cats.index(product["category"]) if product.get("category") in cats else 0)
            channel = c3.selectbox("Channel", ["B2C", "B2B"], index=0 if product.get("channel") != "B2B" else 1,
                                   key=f"{k}_ch")
            with c4:
                age = _num_in("Age (months)", product.get("age_months"), f"{k}_age", step=1.0, fmt="%.0f")

            vals = {}
            if mode == "manual":
                st.markdown("**Demand information** — Momentum & Predictability")
                c = st.columns(5)
                vals["current_t3m"] = _num_in_col(c[0], "Sales this quarter (Q_t)", product, "current_t3m", k)
                vals["previous_t3m"] = _num_in_col(c[1], "Sales previous quarter (Q_t−1)", product, "previous_t3m", k)
                vals["t3m_y1"] = _num_in_col(c[2], "Same quarter last year (Q_t−4)", product, "t3m_y1", k)
                vals["si_q_t"] = _num_in_col(c[3], "Seasonal index Q_t (also Q_t−4)", product, "si_q_t", k, 0.05)
                vals["si_q_prev"] = _num_in_col(c[4], "Seasonal index Q_t−1", product, "si_q_prev", k, 0.05)
                st.caption("Blank seasonal index = 1.00 (no adjustment). S′ = sales ÷ seasonal index.")
                c = st.columns(4)
                vals["forecast"] = _num_in_col(c[0], "Forecast (period)", product, "forecast", k)
                vals["actual"] = _num_in_col(c[1], "Actual demand (period)", product, "actual", k)
                st.markdown("**Reach & Position information**")
                c = st.columns(4)
                vals["active_points"] = _num_in_col(c[0], "Active buying points", product, "active_points", k)
                vals["total_points"] = _num_in_col(c[1], "Total potential buying points", product, "total_points", k)
                vals["current_volume"] = _num_in_col(c[2], "Current T3M volume", product, "current_volume", k)
                vals["peak_volume"] = _num_in_col(c[3], "Peak T3M volume (24 mo)", product, "peak_volume", k)
                hist_df = None
            else:
                st.markdown("**Monthly sales history** — one row per month (YYYY-MM). Momentum, Predictability, "
                            "Reach and Position are derived from this table by `derive_inputs_from_history()`. "
                            "Momentum needs 15 months (same quarter last year); sales are deseasonalised "
                            "with seasonal indices estimated from the portfolio's history.")
                h = calc.history_frame(product.get("history"))
                if h.empty:
                    h = pd.DataFrame({"month": [str(pd.Period("2026-08", "M") - i) for i in range(5, -1, -1)],
                                      "sales_volume": [None] * 6, "forecast": [None] * 6,
                                      "active_points": [None] * 6})
                hist_df = st.data_editor(
                    h, num_rows="dynamic", key=f"{k}_hist", hide_index=True, height=260,
                    column_config={
                        "month": st.column_config.TextColumn("Month (YYYY-MM)", required=True),
                        "sales_volume": st.column_config.NumberColumn("Sales volume (actual)", format="%.0f"),
                        "forecast": st.column_config.NumberColumn("Forecast", format="%.0f"),
                        "active_points": st.column_config.NumberColumn("Active buying points", format="%.0f"),
                    })
                vals["total_points"] = _num_in("Total potential buying points", product.get("total_points"),
                                               f"{k}_tot")

            st.markdown("**Inventory information** — Layer 2 SKU economics")
            vals.update(inventory_inputs(product, k))
            others = [None] + [i for i in SS.products if i != pid]
            ref = st.selectbox("Analogue / reference SKU (used for Introduction σd when σd is blank)", others,
                               index=others.index(product.get("reference_sku_id"))
                               if product.get("reference_sku_id") in others else 0,
                               format_func={None: "None", **{i: product_name(i) for i in others if i}}.get,
                               key=f"{k}_ref")

            b1, b2, _ = st.columns([1, 1, 5])
            save = b1.form_submit_button("Save product", type="primary", width="stretch")
            cancel = b2.form_submit_button("Cancel", width="stretch")

        if cancel:
            SS[f"{prefix}_form"] = None
            st.rerun()
        if save:
            if not name.strip():
                st.error("Product name is required.")
                return
            new = copy.deepcopy(product)
            new.update(vals)
            new.update({"name": name.strip(), "category": cat, "channel": channel, "age_months": age,
                        "input_mode": mode, "reference_sku_id": ref})
            if mode == "history":
                new["history"] = _clean_history(hist_df)
            SS.products[pid] = new
            SS.selected_id = pid
            SS[f"{prefix}_form"] = None
            issues = calc.validate_product(new)
            errs = [i["message"] for i in issues if i["level"] == "error"]
            flash(f"{'Added' if is_new else 'Updated'} “{new['name']}”." +
                  (f" Data issues: {'; '.join(errs)}" if errs else ""))
            st.rerun()


def _num_in_col(col, label, product, field, k, step=1.0):
    with col:
        return _num_in(label, product.get(field), f"{k}_{field}", step=step)


def product_toolbar(prefix: str, pid: str | None, show_select=False):
    """Add / Edit / Delete controls (shared by the calculator and portfolio tabs)."""
    c = st.columns([1, 1, 1.1, 3])
    if c[0].button("Add product", icon=":material/add:", key=f"{prefix}_add", width="stretch"):
        SS[f"{prefix}_form"] = "add"
        SS[f"{prefix}_new"] = demo_data.blank_product()
        SS[f"{prefix}_del"] = None
    if c[1].button("Edit product", icon=":material/edit:", key=f"{prefix}_edit", width="stretch",
                   disabled=pid is None):
        SS[f"{prefix}_form"] = "edit"
        SS[f"{prefix}_del"] = None
    if c[2].button("Delete product", icon=":material/delete:", key=f"{prefix}_delete", width="stretch",
                   disabled=pid is None):
        SS[f"{prefix}_del"] = pid
        SS[f"{prefix}_form"] = None

    if SS.get(f"{prefix}_del") and SS[f"{prefix}_del"] in SS.products:
        did = SS[f"{prefix}_del"]
        with st.container(border=True):
            st.warning(f"Delete **{SS.products[did]['name']}**? Every dashboard will update immediately.")
            a, b, _ = st.columns([1, 1, 5])
            if a.button("Confirm delete", type="primary", key=f"{prefix}_del_yes"):
                name = SS.products.pop(did)["name"]
                SS[f"{prefix}_del"] = None
                SS.selected_id = next(iter(SS.products), None)
                flash(f"Deleted “{name}”.")
                st.rerun()
            if b.button("Cancel", key=f"{prefix}_del_no"):
                SS[f"{prefix}_del"] = None
                st.rerun()

    mode = SS.get(f"{prefix}_form")
    if mode == "add":
        product_form(prefix, SS[f"{prefix}_new"], is_new=True)
    elif mode == "edit" and pid in SS.products:
        product_form(prefix, SS.products[pid], is_new=False)


# =============================================================================
# Shared blocks
# =============================================================================
def decision_flow(r: dict):
    si, inv = r["stage_info"], r["inventory"]
    lss = r["lss"]
    lss_txt = f"{lss.value:.1f}" if lss.ok else ("Age-gated" if r["life"]["gated"] else "N/A")
    drivers = r["strategy"].get("drivers", {})
    top = calc.SIGNAL_LABELS.get(drivers.get("largest_contribution"), "—")
    ui.flow_strip([
        ("Product", ui.esc(r["product"]["name"])),
        ("Signals", f"Top driver: {ui.esc(top)}"),
        ("LSS", lss_txt),
        ("Lifecycle", ui.badge(si["policy_stage"])),
        ("Safety stock", ui.fmt_num(inv["safety_stock"].value, na="N/A")),
        ("ROP", ui.fmt_num(inv["reorder_level"].value, na="None" if si["policy_stage"] == "Exit" else "N/A")),
        ("Strategy", ui.esc(r["strategy"]["headline"])),
    ])


def stage_status_line(r: dict) -> str:
    si = r["stage_info"]
    if si.get("pending"):
        return (f"Signal stage is **{si['pending']}** for {si['pending_count']} of {si['confirm_months']} "
                f"required months → inventory policy stays **{si['policy_stage']}** until confirmed.")
    if si["signal_stage"] and si["policy_stage"] and si["signal_stage"] != si["policy_stage"]:
        return f"Signal stage {si['signal_stage']} · policy stage {si['policy_stage']}."
    return ""


def show_issues(r: dict):
    for i in r["issues"]:
        (st.error if i["level"] == "error" else st.warning)(i["message"], icon=":material/report:")


def no_product():
    st.info("No product selected. Add a product (or reload the demo SKUs from the sidebar).",
            icon=":material/inventory_2:")


# =============================================================================
# TAB 1 — LSS calculator
# =============================================================================
def tab_lss(out: dict):
    pid = SS.selected_id
    product_toolbar("calc", pid)
    if pid is None or pid not in out["results"]:
        return no_product()
    r = out["results"][pid]
    st.markdown(f"### {ui.esc(r['product']['name'])}")
    decision_flow(r)
    show_issues(r)

    left, right = st.columns([1, 1.55], gap="large")
    with left:
        lss, si = r["lss"], r["stage_info"]
        if lss.ok:
            num = f'<span class="hero-num">{lss.value:.1f}</span><span class="hero-den"> / 100</span>'
        elif r["life"]["gated"]:
            shadow = (f"<div class='kpi-sub'>Shadow LSS (information only): {r['shadow_lss']:.1f} → "
                      f"{r['shadow_stage']}</div>" if r.get("shadow_lss") is not None else "")
            num = f'<span class="hero-na">N/A — age-gated</span>{shadow}'
        else:
            num = f'<span class="hero-na">{ui.NA_TEXT}</span>'
        st.markdown(
            f'<div class="hero"><div class="kpi-label">Lifecycle Signal Score</div>{num}'
            f'<div style="margin-top:.8rem" class="kpi-label">Lifecycle stage</div>'
            f'<div style="margin-top:.25rem">{ui.badge(si["policy_stage"])}</div>'
            f'<div class="why"><b>Why?</b> {ui.esc(si["reason"])}</div></div>', unsafe_allow_html=True)
        line = stage_status_line(r)
        if line:
            st.info(line, icon=":material/hourglass_top:")
        d = r["strategy"].get("drivers") or {}
        if d:
            st.markdown(f"**Driving the score:** {calc.SIGNAL_LABELS[d['largest_contribution']]} "
                        f"(largest weighted contribution) · **Weakest signal:** {calc.SIGNAL_LABELS[d['weakest']]}")
    with right:
        rows = []
        for s in calc.SIGNALS:
            m = r["signals"][s]
            c = r["contributions"][s]
            rows.append({"Signal": calc.SIGNAL_LABELS[s], "Raw value": fmt_raw(s, m.raw),
                         "Score": ui.fmt_num(m.value, "{:.1f}"), "Weight": f"{calc.LSS_WEIGHTS[s]:.0%}",
                         "Contribution": ui.fmt_num(c, "{:.1f}")})
        total = r["lss"].value if r["lss"].ok else r.get("shadow_lss")
        rows.append({"Signal": "LSS" + (" (shadow)" if r["life"]["gated"] and total is not None else ""),
                     "Raw value": "", "Score": ui.fmt_num(total, "{:.1f}"), "Weight": "100%",
                     "Contribution": ui.fmt_num(total, "{:.1f}")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.plotly_chart(ui.contribution_chart(r["contributions"], calc.LSS_WEIGHTS), theme=None,
                        config=PLOT_CFG, key="lss_contrib")

    ui.section("The four demand signals", "Each score is 0–100. Open any card to see its inputs, formula and working.")
    cols = st.columns(4)
    raw_fmt = {"momentum": "{:+.2%}", "predictability": "{:.2%}", "reach": "{:.2%}", "position": "{:.3f}"}
    for col, s in zip(cols, calc.SIGNALS):
        m = r["signals"][s]
        with col:
            ui.metric_kpi(f"{calc.SIGNAL_LABELS[s]} · {calc.LSS_WEIGHTS[s]:.0%}", m, "{:.1f}",
                          sub=calc.SIGNAL_QUESTIONS[s] if m.ok else m.reason, color=ui.ACCENT)
    panels = st.columns(2)
    for i, s in enumerate(calc.SIGNALS):
        with panels[i % 2]:
            ui.calc_panel(r["signals"][s], f"View calculation — {calc.SIGNAL_LABELS[s]}", raw_fmt=raw_fmt[s])
    lss_m = r["lss"]
    show = lss_m if lss_m.ok else calc.calculate_lss(r["scores"], calc.LSS_WEIGHTS)
    ui.calc_panel(show, "View calculation — LSS" + (" (shadow, age-gated)" if r["life"]["gated"] else ""))

    with st.expander("View calculation — Lifecycle classification"):
        t = calc.LIFECYCLE_THRESHOLDS
        rules = pd.DataFrame([
            ["Introduction", f"Age ≤ {t['introduction_max_age_months']} months (age-gated)"],
            ["Growth", f"LSS ≥ {t['growth_min_lss']}"],
            ["Maturity", f"{t['maturity_min_lss']} ≤ LSS < {t['growth_min_lss']}"],
            ["Decline", f"{t['decline_min_lss']} ≤ LSS < {t['maturity_min_lss']}"],
            ["Exit", f"LSS < {t['decline_min_lss']}"]], columns=["Stage", "Rule"])
        rules["This SKU"] = np.where(rules["Stage"] == (r["stage_info"]["signal_stage"] or ""), "◀ signal stage", "")
        st.dataframe(rules, hide_index=True, width="stretch")
        st.markdown(f"**Result:** {r['stage_info']['reason']}")
        k = r["stage_info"]["confirm_months"]
        st.markdown(f"**Confirmation rule:** {'on' if k > 1 else 'off'} — a stage change must persist for "
                    f"{k} consecutive month(s) before the inventory policy changes. Policy stage: "
                    f"**{r['stage_info']['policy_stage'] or 'N/A'}**.")
    with st.expander("Inputs used for this calculation"):
        inp = r["inputs"]
        st.caption(f"Source: {inp.get('source')}" + (f" · as of {inp.get('as_of')}" if inp.get("as_of") else "")
                   + (f" · seasonal indices: {inp.get('seasonality_source')}" if inp.get("seasonality_source") else ""))
        show_inp = {"Sales Q_t": inp.get("current_t3m"), "Sales Q_t−1": inp.get("previous_t3m"),
                    "Sales Q_t−4 (same quarter last year)": inp.get("t3m_y1"),
                    "Seasonal index Q_t": inp.get("si_q_t"), "Seasonal index Q_t−1": inp.get("si_q_prev"),
                    "Seasonal index Q_t−4": inp.get("si_q_yoy"),
                    "Current T3M volume": inp.get("current_volume"), "Peak T3M volume": inp.get("peak_volume"),
                    "Forecast/actual periods": len(inp.get("actuals") or []),
                    "Active buying points": inp.get("active_points"),
                    "Total potential buying points": inp.get("total_points"),
                    "Months of history": inp.get("months_available")}
        st.dataframe(pd.DataFrame({"Input": show_inp.keys(),
                                   "Value": [ui.fmt_num(v, na="—") for v in show_inp.values()]}),
                     hide_index=True, width="stretch")


# =============================================================================
# TAB 2 — Inventory calculator
# =============================================================================
def tab_inventory(out: dict):
    pid = SS.selected_id
    if pid is None or pid not in out["results"]:
        return no_product()
    r = out["results"][pid]
    p, inv, si = r["product"], r["inventory"], r["stage_info"]
    stage = si["policy_stage"]
    pol = calc.STAGE_POLICY.get(stage)
    st.markdown(f"### {ui.esc(p['name'])} &nbsp; {ui.badge(stage)}", unsafe_allow_html=True)
    line = stage_status_line(r)
    if line:
        st.info(line, icon=":material/hourglass_top:")

    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown('<div class="layer" style="background:#16325C"><h4>Layer 1 · Lifecycle control — '
                    'What should we do?</h4><p>The policy stage sets the envelope.</p></div>',
                    unsafe_allow_html=True)
        if pol:
            z_txt = (f"Z {inv['z'].value:.2f} ≈ {inv['service_level']:.1%}" if inv["z"].ok else "—")
            ui.policy_grid([("Service priority", f"{pol['service_priority']} · {z_txt}"),
                            ("Push / Pull", pol["push_pull"]), ("Review cadence", pol["review_cadence"]),
                            ("Network footprint", pol["network_footprint"]),
                            ("Replenishment posture", pol["replenishment_posture"])])
        else:
            st.warning("No lifecycle stage → no policy envelope. " + ui.NA_TEXT)
    with b:
        st.markdown('<div class="layer" style="background:#1E6B5C"><h4>Layer 2 · SKU inventory engine — '
                    'How much should we keep?</h4><p>SKU economics set the quantity. Edit and apply.</p></div>',
                    unsafe_allow_html=True)
    with st.form(f"inv_form_{pid}"):
        f = inventory_inputs(p, f"inv{pid}")
        if st.form_submit_button("Apply inventory inputs", type="primary"):
            SS.products[pid] = {**p, **f}
            flash("Inventory inputs updated — all dashboards recalculated.")
            st.rerun()

    ui.section("Inventory outputs")
    c = st.columns(6)
    with c[0]:
        ui.metric_kpi("Safety stock", inv["safety_stock"], sub="units", color=ui.stage_color(stage))
    with c[1]:
        ui.metric_kpi("Cycle stock", inv["cycle_stock"],
                      sub="units · EOQ" if pol and pol["cycle_stock_method"] == "eoq" else "units · d̄·T/2",
                      color=ui.stage_color(stage))
    with c[2]:
        ui.metric_kpi("Reorder level (ROP)", inv["reorder_level"], sub="units", color=ui.stage_color(stage))
    with c[3]:
        ui.metric_kpi("Total requirement", inv["max_stock"], sub="max = SS + Q", color=ui.stage_color(stage))
    with c[4]:
        ui.metric_kpi("Days of cover", inv["days_of_cover"], "{:,.1f}", sub="(SS + CS) ÷ d̄",
                      color=ui.stage_color(stage))
    with c[5]:
        m = inv["exposure_value"]
        ui.kpi("Inventory exposure", ui.fmt_rs(m.value) if m.ok else ui.NA_TEXT, "(SS + CS) × cost",
               ui.stage_color(stage), na=not m.ok)
    c = st.columns(6)
    with c[0]:
        ui.metric_kpi("Seasonal demand", inv.get("demand_rate"), "{:,.1f}", sub="d̄, units/day")
    with c[1]:
        ui.metric_kpi("Lead time", inv.get("lead_time"), "{:,.2f}", sub="L̄, days (local/cross-city mix)")
    with c[2]:
        ui.metric_kpi("Lead-time variability", inv.get("lead_time_std"), "{:,.2f}", sub="σ_L, days")
    c = st.columns(6)
    with c[0]:
        ui.kpi("Review period", f"{inv['review_period']:g} days" if inv.get("review_period") else "Event-driven",
               "policy" if not p.get("review_period_override") else "SKU override")
    with c[1]:
        ui.metric_kpi("Order quantity Q", inv["order_quantity"], sub="units per replenishment")
    with c[2]:
        ui.metric_kpi("Stocking locations", inv["stocking_locations"], sub="assumption — network coverage")
    with c[3]:
        ui.metric_kpi("Max stock / location", inv["per_location"], sub="allocation (assumption)")
    with c[4]:
        ui.metric_kpi("On-hand cover", inv["on_hand_days"], "{:,.1f}", sub="days of current stock")
    with c[5]:
        ui.metric_kpi("Excess vs policy max", inv["excess_units"], sub="units (+ = above max)")
    for n in inv.get("notes", []):
        st.caption("ℹ️ " + n)

    c1, c2 = st.columns(2)
    with c1:
        if inv.get("demand_rate") is not None:
            ui.calc_panel(inv["demand_rate"], "View calculation — Seasonal demand d̄", "{:,.2f} units/day")
    with c2:
        if inv.get("lead_time") is not None:
            ui.calc_panel(inv["lead_time_std"], "View calculation — Lead time L̄ and σ_L", "σ_L = {:,.2f} days")
            if inv["lead_time"].ok:
                st.caption(f"L̄ = {inv['lead_time'].value:.2f} days — {inv['lead_time'].steps[0]}")
    c1, c2 = st.columns(2)
    with c1:
        ui.calc_panel(inv["safety_stock"], "View calculation — Safety stock", "{:,.0f} units")
        ui.calc_panel(inv["reorder_level"], "View calculation — Reorder level", "{:,.0f} units")
    with c2:
        ui.calc_panel(inv["cycle_stock"], "View calculation — Cycle stock", "{:,.0f} units")
        ui.calc_panel(inv["days_of_cover"], "View calculation — Days of cover", "{:,.1f} days")

    g1, g2 = st.columns([1, 1.1], gap="large")
    with g1:
        prof = calc.project_inventory_profile(inv)
        if prof.empty:
            st.info("Projection unavailable — " + ui.NA_TEXT)
        else:
            fig = go.Figure()
            fig.add_scatter(x=prof["day"], y=prof["on_hand"], mode="lines", name="Projected on-hand",
                            line=dict(color=ui.ACCENT, width=2),
                            hovertemplate="Day %{x}: %{y:,.0f} units<extra></extra>")
            if inv["reorder_level"].ok:
                fig.add_hline(y=inv["reorder_level"].value, line=dict(color=ui.INK_2, dash="dash", width=1),
                              annotation_text="ROP", annotation_position="top left")
            fig.add_hline(y=inv["safety_stock"].value, line=dict(color=ui.stage_color(stage), width=2),
                          annotation_text="Safety stock", annotation_position="bottom left")
            fig.update_xaxes(title="Days")
            fig.update_yaxes(title="Units", rangemode="tozero")
            st.plotly_chart(ui.base_layout(fig, "Projected stock profile under this policy (constant demand)",
                                           320, legend=False), theme=None, config=PLOT_CFG, key="inv_profile")
    with g2:
        st.markdown("**Same SKU economics, five policies**")
        st.caption("Layer 1 changes Z, review period and overrides; Layer 2 inputs stay fixed. "
                   "The lifecycle stage alone never sets the quantity.")
        comp = calc.inventory_by_stage(p, out["settings"])
        comp["Service level"] = comp["Service level"].map(lambda v: f"{v:.1%}" if v is not None else "—")
        comp["Policy stage"] = np.where(comp["Policy stage"] == stage, comp["Policy stage"] + " ◀", comp["Policy stage"])
        st.dataframe(comp[["Policy stage", "Z", "Service level", "Review (days)", "Safety stock",
                           "Cycle stock", "ROP", "Days of cover"]], hide_index=True, width="stretch",
                     column_config={"ROP": st.column_config.NumberColumn(format="%,.0f"),
                                    "Safety stock": st.column_config.NumberColumn(format="%,.0f"),
                                    "Cycle stock": st.column_config.NumberColumn(format="%,.0f")})


# =============================================================================
# TAB 3 — Strategy recommendation
# =============================================================================
def strategy_block(r: dict):
    p, inv, si, strat = r["product"], r["inventory"], r["stage_info"], r["strategy"]
    stage = si["policy_stage"]
    c = st.columns([1.2, 0.8, 0.9, 0.8, 0.8, 0.8])
    with c[0]:
        ui.kpi("Product", p["name"], f"{p.get('category')} · {p.get('channel')} · {ui.fmt_num(p.get('age_months'), na='?')} mo",
               small=True)
    with c[1]:
        lss = r["lss"]
        ui.kpi("LSS", f"{lss.value:.1f} / 100" if lss.ok else ("Age-gated" if r["life"]["gated"] else ui.NA_TEXT),
               "Lifecycle Signal Score", na=not lss.ok)
    with c[2]:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Lifecycle</div><div style="margin-top:.45rem">'
                    f'{ui.badge(stage)}</div><div class="kpi-sub">{ui.esc("pending " + si["pending"] if isinstance(si.get("pending"), str) else "confirmed")}</div></div>',
                    unsafe_allow_html=True)
    with c[3]:
        ui.metric_kpi("Safety stock", inv["safety_stock"], sub="units")
    with c[4]:
        ui.metric_kpi("Cycle stock", inv["cycle_stock"], sub="units")
    with c[5]:
        ui.metric_kpi("ROP", inv["reorder_level"], sub="units")

    if not strat.get("policy"):
        st.warning(strat["headline"])
        return
    pol = strat["policy"]
    st.markdown(f'<div class="hero" style="margin-top:.9rem;border-left:6px solid {ui.stage_color(stage)}">'
                f'<div class="kpi-label">Recommended strategy</div>'
                f'<div style="font-size:1.45rem;font-weight:800;color:{ui.INK};margin-top:.2rem">'
                f'{ui.esc(strat["headline"])}</div>'
                f'<div class="kpi-sub">Business action: {ui.esc(strat["business_action"])} · '
                f'Objective: {ui.esc(pol["objective"])}</div></div>', unsafe_allow_html=True)
    ui.section("Policy envelope (Layer 1)")
    z = inv["z"]
    ui.policy_grid([
        ("Objective", pol["objective"]),
        ("Service level", f"{pol['service_priority']}" + (f" · Z {z.value:.2f} (~{inv['service_level']:.1%})" if z.ok else "")),
        ("Safety stock approach", pol["safety_stock_approach"]),
        ("Cycle stock approach", pol["cycle_stock_approach"]),
        ("Reorder logic", pol["rop_approach"]),
        ("Network strategy", pol["network_footprint"]),
        ("Push / Pull", pol["push_pull"]),
        ("Review cadence", pol["review_cadence"]),
        ("Incentive treatment", pol["incentive"]),
    ])
    ui.section("Recommended actions", "Generated from the policy stage and this SKU's own signals and inventory.")
    for i, a in enumerate(strat["actions"], 1):
        st.markdown(f'<div class="action"><b>{i}.</b> {ui.esc(a)}</div>', unsafe_allow_html=True)


def tab_strategy(out: dict):
    pid = SS.selected_id
    if pid is None or pid not in out["results"]:
        return no_product()
    r = out["results"][pid]
    decision_flow(r)
    strategy_block(r)
    ui.section("Decision summary")
    si, inv, d = r["stage_info"], r["inventory"], r["strategy"].get("drivers") or {}
    lss = r["lss"]
    qa = [
        ("What is the product's lifecycle stage?", f"{si['policy_stage'] or 'N/A'}"
         + (f" (signal: {si['pending']}, awaiting confirmation)" if si.get("pending") else "")),
        ("Why is it in that stage?", si["reason"]),
        ("What is its LSS score?", f"{lss.value:.1f} / 100" if lss.ok else
         ("Not scored — age-gated" if r["life"]["gated"] else ui.NA_TEXT)),
        ("Which demand signal is driving the score?",
         f"{calc.SIGNAL_LABELS[d['largest_contribution']]} contributes most; "
         f"{calc.SIGNAL_LABELS[d['weakest']]} is weakest." if d else ui.NA_TEXT),
        ("How much safety stock should be held?", inv["safety_stock"].display("{:,.0f} units")),
        ("What inventory policy should be used?", (lambda pl: f"{pl['service_priority']} service, "
         f"{pl['push_pull']}, {pl['review_cadence'].lower()} review" if pl else ui.NA_TEXT)(r["strategy"].get("policy"))),
        ("What should the business do next?", r["strategy"]["actions"][0] if r["strategy"]["actions"] else "—"),
    ]
    st.dataframe(pd.DataFrame(qa, columns=["Question", "Answer"]), hide_index=True, width="stretch")


# =============================================================================
# TAB 4 — Portfolio
# =============================================================================
def tab_portfolio(out: dict):
    t = out["table"].copy()
    ui.section("Portfolio", "Every SKU scored by the same engine. Select a row to edit, delete or open it.")
    c = st.columns([2, 1.6, 1.6, 1.2, 0.9])
    q = c[0].text_input("Search", placeholder="Product name…", key="pf_search")
    stages = c[1].multiselect("Lifecycle", calc.STAGES + ["N/A"], key="pf_stage")
    cats = c[2].multiselect("Category", sorted(t["Category"].dropna().unique()) if len(t) else [], key="pf_cat")
    sort_cols = ["LSS", "Product", "Age", "Momentum", "Predictability", "Reach", "Position", "Safety Stock",
                 "ROP", "Exposure (Rs)"]
    sort_by = c[3].selectbox("Sort by", sort_cols, key="pf_sort")
    desc = c[4].toggle("Descending", value=True, key="pf_desc")
    if len(t):
        if q:
            t = t[t["Product"].str.contains(q, case=False, na=False)]
        if stages:
            t = t[t["Lifecycle Stage"].isin(stages)]
        if cats:
            t = t[t["Category"].isin(cats)]
        t = t.sort_values(sort_by, ascending=not desc, na_position="last")
    cols = ["Product", "Category", "Channel", "Age", "Momentum", "Predictability", "Reach", "Position", "LSS",
            "Lifecycle Stage", "Signal Stage", "Safety Stock", "Cycle Stock", "ROP", "Days of Cover",
            "Exposure (Rs)", "Strategy"]
    num = lambda f="%.1f": st.column_config.NumberColumn(format=f)
    ev = st.dataframe(
        t[cols] if len(t) else pd.DataFrame(columns=cols), hide_index=True, width="stretch", height=460,
        on_select=_pick_from_table, selection_mode="single-row", key="pf_table",
        column_config={"Age": num("%d"), "Momentum": num(), "Predictability": num(), "Reach": num(),
                       "Position": num(), "LSS": st.column_config.ProgressColumn("LSS", min_value=0, max_value=100,
                                                                                 format="%.1f"),
                       "Safety Stock": num("%,.0f"), "Cycle Stock": num("%,.0f"), "ROP": num("%,.0f"),
                       "Days of Cover": num(), "Exposure (Rs)": num("₹%,.0f"),
                       "Strategy": st.column_config.TextColumn(width="large")})
    st.caption(f"{len(t)} of {len(out['table'])} SKUs shown · blank LSS = age-gated or insufficient data · "
               "Lifecycle Stage = policy stage after the confirmation rule.")
    SS._pf_ids = t["id"].tolist() if len(t) else []
    sel_pid = SS.selected_id if SS.selected_id in SS.products else None
    b = st.columns([1.3, 1.3, 4])
    export = out["table"].loc[out["table"]["id"].isin(t["id"])] if len(t) else out["table"]
    b[0].download_button("Download CSV", export.drop(columns=["id"]).to_csv(index=False).encode("utf-8"),
                         file_name="lifecycle_portfolio.csv", mime="text/csv", icon=":material/download:",
                         key="pf_csv", width="stretch")
    st.markdown(f"**Selected:** {ui.esc(product_name(sel_pid))} — click a row to change; the same product "
                "opens in every other tab.")
    product_toolbar("pf", sel_pid)


# =============================================================================
# TAB 5 — Lifecycle dashboard
# =============================================================================
def tab_dashboard(out: dict):
    t = out["table"]
    if t.empty:
        return no_product()
    c = st.columns(6)
    with c[0]:
        ui.kpi("Total SKUs", f"{len(t)}", f"{int(t['Demo'].sum())} demo")
    for col, s in zip(c[1:], calc.STAGES):
        with col:
            n = int((t["Lifecycle Stage"] == s).sum())
            ui.kpi(s, f"{n}", f"{n / len(t):.0%} of SKUs", ui.stage_color(s))
    na = int((t["Lifecycle Stage"] == "N/A").sum())
    if na:
        st.caption(f"{na} SKU(s) unclassified — insufficient data.")

    g1, g2 = st.columns(2, gap="medium")
    with g1:
        counts = t["Lifecycle Stage"].value_counts().reindex(calc.STAGES, fill_value=0)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[ui.stage_color(s) for s in counts.index],
                               text=counts.values, textposition="outside",
                               hovertemplate="%{x}: %{y} SKUs<extra></extra>"))
        fig.update_yaxes(title="SKUs")
        st.plotly_chart(ui.base_layout(fig, "1 · Lifecycle distribution", legend=False), theme=None,
                        config=PLOT_CFG, key="d_dist")
    with g2:
        fig = go.Figure()
        scored = t.dropna(subset=["LSS"])
        for s in calc.STAGES:
            v = scored[scored["Lifecycle Stage"] == s]["LSS"]
            if len(v):
                fig.add_histogram(x=v, name=s, marker_color=ui.stage_color(s), xbins=dict(start=0, end=100, size=10),
                                  hovertemplate=f"{s}: %{{y}} SKUs in LSS %{{x}}<extra></extra>")
        for th in (20, 40, 60):
            fig.add_vline(x=th, line=dict(color=ui.MUTED, dash="dot", width=1))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="LSS", range=[0, 100])
        fig.update_yaxes(title="SKUs")
        st.plotly_chart(ui.base_layout(fig, "2 · LSS distribution (age-gated SKUs excluded)"), theme=None,
                        config=PLOT_CFG, key="d_hist")

    g1, g2 = st.columns(2, gap="medium")
    with g1:
        fig = go.Figure()
        ui.add_stage_bands(fig)
        fig.add_vrect(x0=0, x1=calc.LIFECYCLE_THRESHOLDS["introduction_max_age_months"],
                      fillcolor=ui.stage_color("Introduction"), opacity=0.08, line_width=0, layer="below")
        res = out["results"]
        for s in calc.STAGES:
            sub = t[t["Lifecycle Stage"] == s]
            if not len(sub):
                continue
            y = [res[i]["lss"].value if res[i]["lss"].ok else res[i].get("shadow_lss") for i in sub["id"]]
            hollow = s == "Introduction"
            fig.add_scatter(x=sub["Age"], y=y, mode="markers", name=s + (" (shadow LSS)" if hollow else ""),
                            marker=dict(size=11, color="white" if hollow else ui.stage_color(s),
                                        line=dict(color=ui.stage_color(s), width=2)),
                            text=sub["Product"], hovertemplate="%{text}<br>Age %{x} mo · LSS %{y:.1f}<extra></extra>")
        fig.update_xaxes(title="Age (months)", rangemode="tozero")
        fig.update_yaxes(title="LSS", range=[0, 100])
        st.plotly_chart(ui.base_layout(fig, "3 · Age vs LSS"), theme=None, config=PLOT_CFG, key="d_age")
    with g2:
        exp = t.groupby("Lifecycle Stage")[["Exposure (Rs)", "On-hand Value (Rs)"]].sum(min_count=1)
        exp = exp.reindex(calc.STAGES)
        fig = go.Figure()
        fig.add_bar(x=exp.index, y=exp["Exposure (Rs)"] / 1e5, name="Policy avg inventory",
                    marker_color=[ui.stage_color(s) for s in exp.index],
                    hovertemplate="%{x}: ₹%{y:,.1f} L policy inventory<extra></extra>")
        fig.add_bar(x=exp.index, y=exp["On-hand Value (Rs)"] / 1e5, name="Current on-hand",
                    marker=dict(color="white", line=dict(color=[ui.stage_color(s) for s in exp.index], width=2),
                                pattern=dict(shape="/", fgcolor=[ui.stage_color(s) for s in exp.index])),
                    hovertemplate="%{x}: ₹%{y:,.1f} L on hand<extra></extra>")
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="₹ lakh")
        st.plotly_chart(ui.base_layout(fig, "4 · Inventory exposure by lifecycle stage"), theme=None,
                        config=PLOT_CFG, key="d_exp")

    g1, g2 = st.columns(2, gap="medium")
    with g1:
        ssg = t.groupby("Lifecycle Stage").agg(ss=("Safety Stock", "sum"), n=("Safety Stock", "count"))
        ssg = ssg.reindex(calc.STAGES).fillna(0)
        fig = go.Figure(go.Bar(x=ssg.index, y=ssg["ss"], marker_color=[ui.stage_color(s) for s in ssg.index],
                               text=[f"{v:,.0f}" for v in ssg["ss"]], textposition="outside",
                               customdata=ssg["n"], hovertemplate="%{x}: %{y:,.0f} units across %{customdata} SKUs<extra></extra>"))
        fig.update_yaxes(title="Safety stock (units)")
        st.plotly_chart(ui.base_layout(fig, "5 · Safety stock by lifecycle stage", legend=False), theme=None,
                        config=PLOT_CFG, key="d_ss")
    with g2:
        m = out["monthly"]
        if m.empty:
            st.info("Monthly stage mix needs products with monthly history.")
        else:
            mix = m.groupby(["month", "policy_stage"]).size().unstack(fill_value=0).reindex(columns=calc.STAGES,
                                                                                          fill_value=0)
            mix = mix.tail(12)
            fig = go.Figure()
            for s in calc.STAGES:
                fig.add_bar(x=mix.index, y=mix[s], name=s, marker_color=ui.stage_color(s),
                            hovertemplate=f"%{{x}} · {s}: %{{y}} SKUs<extra></extra>")
            fig.update_layout(barmode="stack")
            fig.update_yaxes(title="SKUs")
            st.plotly_chart(ui.base_layout(fig, "Monthly lifecycle mix (policy stage, last 12 months)"), theme=None,
                            config=PLOT_CFG, key="d_mix")

    ui.section("Alerts", "Generated automatically by generate_alerts() from the latest calculation.")
    al = pd.DataFrame(out["alerts"])
    if al.empty:
        st.success("No alerts.")
        return
    c = st.columns([2, 3])
    kinds = c[0].multiselect("Alert type", sorted(al["Alert"].unique()), key="al_kind")
    sev = c[1].segmented_control("Severity", ["High", "Medium", "Low", "Info"], selection_mode="multi",
                                 default=["High", "Medium", "Low", "Info"], key="al_sev")
    if kinds:
        al = al[al["Alert"].isin(kinds)]
    if sev is not None:
        al = al[al["Severity"].isin(sev)]
    summary = pd.DataFrame(out["alerts"]).groupby("Alert").size().sort_values(ascending=False)
    st.caption(" · ".join(f"{k}: {v}" for k, v in summary.items()))
    st.dataframe(al[["Severity", "Alert", "Product", "Detail"]], hide_index=True, width="stretch", height=380)

    if not out["monthly"].empty:
        ui.section("Lifecycle movements in the latest month")
        m = out["monthly"]
        last = m.sort_values("month").groupby("product_id").tail(1)
        mv = last[last["stage_changed"] | last["pending"].notna()]
        if mv.empty:
            st.caption("No stage movement this month.")
        else:
            st.dataframe(pd.DataFrame({
                "Product": [SS.products[i]["name"] for i in mv["product_id"]],
                "Month": mv["month"], "From": mv["prev_signal_stage"], "To (signal)": mv["signal_stage"],
                "Policy stage": mv["policy_stage"],
                "Status": np.where(mv["pending"].notna(), "Awaiting confirmation", "Confirmed")}),
                hide_index=True, width="stretch")


# =============================================================================
# TAB 6 — Product detail
# =============================================================================
def tab_detail(out: dict):
    pid = SS.selected_id
    if pid is None or pid not in out["results"]:
        return no_product()
    r = out["results"][pid]
    p, inv, si = r["product"], r["inventory"], r["stage_info"]
    st.markdown(f"### {ui.esc(p['name'])} &nbsp; {ui.badge(si['policy_stage'])}"
                + (" &nbsp; <span class='demo-pill'>Illustrative demo SKU</span>" if p.get("is_demo") else ""),
                unsafe_allow_html=True)
    ui.section("Product overview")
    c = st.columns(6)
    with c[0]:
        ui.kpi("Category", p.get("category") or "—", small=True)
    with c[1]:
        ui.kpi("Channel", p.get("channel") or "—")
    with c[2]:
        ui.kpi("Age", ui.fmt_num(p.get("age_months"), "{:.0f} mo"))
    with c[3]:
        ui.kpi("LSS", f"{r['lss'].value:.1f}" if r["lss"].ok else ("Age-gated" if r["life"]["gated"] else ui.NA_TEXT),
               na=not r["lss"].ok)
    with c[4]:
        ui.kpi("Signal stage", si["signal_stage"] or "N/A", "this month", ui.stage_color(si["signal_stage"]))
    with c[5]:
        ui.kpi("Policy stage", si["policy_stage"] or "N/A", "drives inventory", ui.stage_color(si["policy_stage"]))

    ui.section("LSS breakdown")
    c = st.columns(4)
    for col, s in zip(c, calc.SIGNALS):
        with col:
            ui.metric_kpi(calc.SIGNAL_LABELS[s], r["signals"][s], "{:.1f}", sub=fmt_raw(s, r["signals"][s].raw),
                          color=ui.ACCENT)

    ui.section("Inventory")
    c = st.columns(6)
    with c[0]:
        ui.kpi("Seasonal demand", ui.fmt_num(inv.get("d_bar"), "{:,.1f}/day"),
               f"avg {ui.fmt_num(p.get('avg_daily_demand'), '{:,.1f}', na='—')} × SI "
               f"{ui.fmt_num(p.get('seasonality_index'), '{:g}', na='1')}")
    with c[1]:
        ui.kpi("Demand variability", ui.fmt_num(inv["inputs_used"]["sd"], "{:,.1f}/day"))
    with c[2]:
        lt, sl = inv.get("lead_time"), inv.get("lead_time_std")
        ui.kpi("Lead time", ui.fmt_num(lt.value if lt is not None else None, "{:.2f} days"),
               f"σ_L {ui.fmt_num(sl.value if sl is not None else None, '{:.2f}', na='—')} days · "
               f"p local {ui.fmt_num(p.get('local_share'), '{:.0%}', na='—')}")
    with c[3]:
        ui.metric_kpi("Safety stock", inv["safety_stock"], sub="units")
    with c[4]:
        ui.metric_kpi("Cycle stock", inv["cycle_stock"], sub="units")
    with c[5]:
        ui.metric_kpi("ROP", inv["reorder_level"], sub="units")

    ui.section("Strategy")
    with st.expander("Full recommendation", expanded=True):
        strategy_block(r)

    seas = r["inputs"].get("seasonal_indices")
    if seas:
        ui.section("Seasonal indices used to deseasonalise Momentum",
                   f"Estimated by estimate_seasonal_indices() · pooled at {r['inputs'].get('seasonality_source')} "
                   "level · S′ = sales ÷ index.")
        mnames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        vals = [seas[m] for m in range(1, 13)]
        fig = go.Figure(go.Bar(x=mnames, y=vals, marker_color=[ui.ACCENT if v >= 1 else "#9DB4DA" for v in vals],
                               text=[f"{v:.2f}" for v in vals], textposition="outside",
                               hovertemplate="%{x}: index %{y:.3f}<extra></extra>"))
        fig.add_hline(y=1, line=dict(color=ui.MUTED, dash="dot", width=1))
        fig.update_yaxes(title="Seasonal index", range=[0, max(vals) * 1.2])
        st.plotly_chart(ui.base_layout(fig, "", 240, legend=False), theme=None, config=PLOT_CFG, key="p_season")

    ui.section("Monthly lifecycle history",
               "Each month: refresh four signals → LSS → classify → compare with last month → confirmation rule.")
    mh = r["monthly"]
    if mh.empty:
        st.info("No monthly history for this SKU (manual snapshot). Switch it to monthly history in Edit product "
                "to see sales, LSS, lifecycle and inventory trends.", icon=":material/timeline:")
        return
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        fig = go.Figure()
        fig.add_scatter(x=mh["month"], y=mh["sales_volume"], name="Actual", mode="lines+markers",
                        line=dict(color=ui.ACCENT, width=2), marker=dict(size=6))
        fig.add_scatter(x=mh["month"], y=mh["forecast"], name="Forecast", mode="lines",
                        line=dict(color=ui.MUTED, width=2, dash="dot"))
        fig.update_yaxes(title="Units / month", rangemode="tozero")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(ui.base_layout(fig, "Sales trend — actual vs forecast"), theme=None, config=PLOT_CFG,
                        key="p_sales")
    with g2:
        fig = go.Figure()
        ui.add_stage_bands(fig)
        fig.add_scatter(x=mh["month"], y=mh["lss"], mode="lines+markers", name="LSS",
                        line=dict(color=ui.INK, width=2),
                        marker=dict(size=9, color=[ui.stage_color(s) for s in mh["signal_stage"]],
                                    line=dict(color="white", width=2)),
                        text=mh["signal_stage"], hovertemplate="%{x}: LSS %{y:.1f} → %{text}<extra></extra>")
        fig.update_yaxes(title="LSS", range=[0, 100])
        st.plotly_chart(ui.base_layout(fig, "LSS trend (blank = age-gated / insufficient data)", legend=False),
                        theme=None, config=PLOT_CFG, key="p_lss")
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        fig = go.Figure()
        order = {s: i for i, s in enumerate(calc.STAGES)}
        for row_name, col in [("Signal stage", "signal_stage"), ("Policy stage", "policy_stage")]:
            fig.add_scatter(x=mh["month"], y=[row_name] * len(mh), mode="markers", name=row_name,
                            marker=dict(symbol="square", size=18, color=[ui.stage_color(s) for s in mh[col]],
                                        line=dict(color="white", width=2)),
                            text=mh[col].fillna("N/A"), hovertemplate="%{x}: %{text}<extra>" + row_name + "</extra>")
        fig.update_yaxes(categoryorder="array", categoryarray=["Policy stage", "Signal stage"])
        st.plotly_chart(ui.base_layout(fig, "Lifecycle trend (colour = stage)", 220, legend=False), theme=None,
                        config=PLOT_CFG, key="p_stage")
        st.markdown(" ".join(ui.badge(s) for s in calc.STAGES), unsafe_allow_html=True)
    with g2:
        fig = go.Figure()
        fig.add_scatter(x=mh["month"], y=mh["reorder_level"], name="ROP", mode="lines",
                        line=dict(color=ui.INK_2, width=2, dash="dash"))
        fig.add_scatter(x=mh["month"], y=mh["safety_stock"], name="Safety stock", mode="lines+markers",
                        line=dict(color=ui.ACCENT, width=2), marker=dict(size=6))
        fig.update_yaxes(title="Units", rangemode="tozero")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(ui.base_layout(fig, "Inventory trend (rolling T3M demand, SKU CV & lead time)"),
                        theme=None, config=PLOT_CFG, key="p_inv")
    tbl = mh[["month", "age", "sales_volume", "forecast"] + calc.SIGNALS +
             ["lss", "signal_stage", "policy_stage", "pending", "stage_changed", "safety_stock", "reorder_level"]].copy()
    tbl.columns = ["Month", "Age", "Sales", "Forecast", "Momentum", "Predictability", "Reach", "Position", "LSS",
                   "Signal stage", "Policy stage", "Pending", "Stage moved", "Safety stock", "ROP"]
    changes = tbl[tbl["Stage moved"]]
    for _, row in changes.tail(3).iterrows():
        prev = mh.loc[mh["month"] == row["Month"], "prev_signal_stage"].iloc[0]
        st.warning(f"Lifecycle change alert · {row['Month']}: {prev} → {row['Signal stage']}"
                   + (f" (policy held at {row['Policy stage']} pending confirmation)" if row["Pending"] else ""),
                   icon=":material/swap_horiz:")
    with st.expander("Month-by-month table"):
        st.dataframe(tbl.iloc[::-1], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f")
                                    for c in ["Momentum", "Predictability", "Reach", "Position", "LSS"]})


# =============================================================================
# Sidebar & layout
# =============================================================================
def _pick_from_table():
    rows = SS.pf_table.selection.rows if "pf_table" in SS else []
    ids = SS.get("_pf_ids", [])
    if rows and rows[0] < len(ids):
        SS.selected_id = ids[rows[0]]


def sidebar(out: dict):
    with st.sidebar:
        st.markdown("### Product")
        ids = list(SS.products)
        if ids:
            if SS.selected_id not in ids:
                SS.selected_id = ids[0]
            labels = {i: product_name(i) for i in ids}
            sel = st.selectbox("Active product", ids, index=ids.index(SS.selected_id), format_func=labels.get,
                               label_visibility="collapsed", key=f"sb_{SS.selected_id}_{len(ids)}")
            if sel != SS.selected_id:
                SS.selected_id = sel
                st.rerun()
            r = out["results"].get(SS.selected_id)
            if r:
                lss = r["lss"]
                st.markdown(f"{ui.badge(r['stage_info']['policy_stage'])} &nbsp; "
                            f"**LSS {lss.value:.1f}**" if lss.ok else ui.badge(r['stage_info']['policy_stage']),
                            unsafe_allow_html=True)
        else:
            st.caption("No products.")

        st.markdown("### Engine settings")
        st.radio("Signal normalisation", ["percentile", "absolute"], key="set_norm",
                 format_func=lambda v: "Portfolio percentile (Annexure A)" if v == "percentile"
                 else "Absolute scale (assumption)")
        st.toggle("Require confirmation before policy change", value=True, key="set_confirm")
        st.slider("Confirmation months", 1, 4, 2, key="set_confirm_months", disabled=not SS.get("set_confirm", True))
        st.radio("Momentum when same-quarter-last-year is missing", ["qoq", "strict"], key="set_mom_fb2",
                 format_func={"qoq": "Fallback: QoQ term only (assumption)",
                              "strict": "Strict: N/A without Q_t−4"}.get,
                 help="Momentum averages QoQ and YoY growth on deseasonalised sales; YoY needs 15 months.")
        st.caption("Decline cycle stock = EOQ √(2DS/H)")
        st.number_input("Ordering cost S (₹/order)", value=1000.0, step=100.0, key="set_order_cost")
        st.number_input("Holding rate H (% of unit cost p.a.)", value=20.0, step=1.0, key="set_hold")
        st.toggle("Shelf-life guardrail (assumption)", value=True, key="set_shelf",
                  help="Caps order quantity so SS + Q ≤ 50% of shelf life in days of demand.")

        st.markdown("### Data")
        c = st.columns(2)
        if c[0].button("Reload demo", width="stretch", help="Replace all products with the illustrative SKUs"):
            SS.products = demo_data.demo_products()
            SS.selected_id = next(iter(SS.products))
            flash("Demo SKUs reloaded.")
            st.rerun()
        if c[1].button("Remove demo", width="stretch", help="Delete every illustrative demo SKU"):
            SS.products = {k: v for k, v in SS.products.items() if not v.get("is_demo")}
            SS.selected_id = next(iter(SS.products), None)
            flash("Demo SKUs removed.")
            st.rerun()
        with st.expander("Formula registry"):
            st.markdown(
                "All formulas live in `calculations.py`:\n"
                "- Momentum → `calculate_momentum()`\n- Predictability → `calculate_predictability()`\n"
                "- Reach → `calculate_reach()`\n- Position → `calculate_position()`\n"
                "- 0–100 scoring → `normalise_signals()`\n- LSS → `calculate_lss()`\n"
                "- Stage → `determine_lifecycle()`\n- Confirmation → `apply_confirmation_rule()`\n"
                "- SS / cycle / ROP → `calculate_inventory()`\n- Strategy → `generate_strategy()`\n"
                "- Alerts → `generate_alerts()`")
            st.caption("Weights: " + ", ".join(f"{calc.SIGNAL_LABELS[s]} {w:.0%}" for s, w in calc.LSS_WEIGHTS.items()))


def header():
    n_demo = sum(1 for p in SS.products.values() if p.get("is_demo"))
    pill = f'<span class="demo-pill">{n_demo} illustrative demo SKUs loaded</span>' if n_demo else ""
    st.markdown(
        f'<div class="app-header"><div><div class="app-eyebrow">Asian Paints · Lifecycle Signal Score</div>'
        f'<div class="app-title">Product Lifecycle &amp; Inventory Decision System</div>'
        f'<div class="app-sub">Lifecycle chooses the playbook. SKU economics choose the stock.</div></div>'
        f'<div>{pill}</div></div>', unsafe_allow_html=True)


def main():
    init_state()
    out = run_engine()
    sidebar(out)
    out = run_engine()  # settings widgets may have changed this run
    header()
    if SS.flash:
        st.toast(SS.flash)
        SS.flash = None
    tabs = st.tabs(["LSS Calculator", "Inventory Calculator", "Strategy", "Portfolio", "Lifecycle Dashboard",
                    "Product Detail"], key="main_tabs")
    with tabs[0]:
        tab_lss(out)
    with tabs[1]:
        tab_inventory(out)
    with tabs[2]:
        tab_strategy(out)
    with tabs[3]:
        tab_portfolio(out)
    with tabs[4]:
        tab_dashboard(out)
    with tabs[5]:
        tab_detail(out)


main()
