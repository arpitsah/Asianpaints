"""Engine tests - formulas, lifecycle, inventory, validation, propagation."""
import copy

import pytest

import calculations as calc
import demo_data


@pytest.fixture
def portfolio():
    return demo_data.demo_products()


def by_name(products, prefix):
    return next(k for k, v in products.items() if v["name"].startswith(prefix))


# ---- final formulas ----------------------------------------------------------
def inv(stage, d, sd, lt, slt, **kw):
    """Single-source lead time (p = 1) reduces the mixture to L̄ = L_local, σ_L = σ_local."""
    return calc.calculate_inventory(stage, d, sd, 1.0, 1.0, lt, None, slt, None, **kw)


@pytest.mark.parametrize("stage,d,sd,lt,slt,ss,cyc,rop", [
    ("Introduction", 40, 18, 10, 3.0, 219, 140, 619),
    ("Growth", 220, 55, 8, 2.0, 817, 1540, 2577),
    ("Maturity", 520, 60, 6, 1.5, 1214, 7800, 4334),
])
def test_ss_cs_rop_match_annexure_b(stage, d, sd, lt, slt, ss, cyc, rop):
    r = inv(stage, d, sd, lt, slt)
    assert r["safety_stock"].value == ss       # SS = Z·√(L̄σd² + d̄²σ_L²)
    assert r["cycle_stock"].value == cyc       # CS = d̄·T/2
    assert r["reorder_level"].value == rop     # ROP = d̄·L̄ + SS


def test_decline_cycle_stock_is_eoq():
    r = inv("Decline", 140, 45, 9, 2.5, unit_cost=200)
    D, S, H = 140 * 365, 1000, 0.20 * 200
    assert r["safety_stock"].value == 435
    assert r["reorder_level"].value == 1695
    assert r["cycle_stock"].value == round((2 * D * S / H) ** 0.5)
    assert not inv("Decline", 140, 45, 9, 2.5)["cycle_stock"].ok     # EOQ needs unit cost


def test_exit_overrides():
    r = inv("Exit", 25, 15, 12, 4.0)
    assert round(r["ss_computed"]) == 106 and r["safety_stock"].value == 0
    assert r["cycle_stock"].value == 0 and not r["reorder_level"].ok


def test_lead_time_mixture():
    mean, std = calc.calculate_lead_time(0.8, 1, 4, 0.5, 1.5)
    assert mean.value == pytest.approx(0.8 * 1 + 0.2 * 4)
    var = 0.8 * 0.5 ** 2 + 0.2 * 1.5 ** 2 + 0.8 * 0.2 * (4 - 1) ** 2
    assert std.value == pytest.approx(var ** 0.5)
    assert not calc.calculate_lead_time(1.5, 1, 4)[0].ok
    assert not calc.calculate_lead_time(0.5, 1, None)[0].ok


def test_mixture_feeds_ss_and_rop():
    r = calc.calculate_inventory("Growth", 100, 20, 1.0, 0.7, 2, 5, 0.5, 1.0)
    lt, slt = r["lead_time"].value, r["lead_time_std"].value
    ss = 1.75 * (lt * 20 ** 2 + 100 ** 2 * slt ** 2) ** 0.5
    assert r["safety_stock"].value == round(ss)
    assert r["reorder_level"].value == round(100 * lt + round(ss))


def test_seasonality_scales_demand():
    base = inv("Maturity", 520, 60, 6, 1.5)
    peak = calc.calculate_inventory("Maturity", 520, 60, 1.2, 1.0, 6, None, 1.5, None)
    assert peak["d_bar"] == pytest.approx(624)
    assert peak["cycle_stock"].value == round(624 * 30 / 2)
    assert peak["safety_stock"].value > base["safety_stock"].value


def test_momentum_final_formula():
    m = calc.calculate_momentum(120, 100, 80)
    assert m.raw == pytest.approx(((120 / 100) + (120 / 80)) / 2 - 1)
    assert m.basis == "Deck"


def test_momentum_fallbacks():
    assert calc.calculate_momentum(120, 100, None, fallback="yoy").raw == pytest.approx(0.2)
    assert calc.calculate_momentum(120, 100, None, fallback="yoy").basis == "Assumption"
    assert not calc.calculate_momentum(120, 100, None, fallback="strict").ok
    assert calc.calculate_momentum(120, None, None, previous_t3m=100, fallback="chain").raw == pytest.approx(0.2)
    assert not calc.calculate_momentum(120, None, None, previous_t3m=100, fallback="yoy").ok


def test_history_derivation_windows():
    hist = [{"month": f"{2024 + (i // 12)}-{i % 12 + 1:02d}", "sales_volume": float(i + 1),
             "forecast": float(i + 1), "active_points": 10.0} for i in range(27)]
    d = calc.derive_inputs_from_history(hist)
    assert d["current_t3m"] == 25 + 26 + 27
    assert d["t3m_y1"] == 13 + 14 + 15
    assert d["t3m_y2"] == 1 + 2 + 3


def test_annexure_a_lss_weights():
    # Waterproofing Sealant XT: 68.4 / 52.6 / 68.4 / 0.98 -> 70.3 (Annexure A)
    lss = calc.calculate_lss({"momentum": 68.4, "predictability": 52.6, "reach": 68.4, "position": 98.0})
    assert lss.value == pytest.approx(70.4, abs=0.1)


def test_raw_signal_formulas():
    assert calc.calculate_momentum(120, 100, 100).raw == pytest.approx(0.20)
    assert calc.calculate_predictability([100, 200], [90, 220]).raw == pytest.approx(0.90)
    assert calc.calculate_reach(300, 1000).raw == pytest.approx(0.30)
    assert calc.calculate_position(80, 100).raw == pytest.approx(0.80)


# ---- lifecycle ---------------------------------------------------------------
@pytest.mark.parametrize("age,lss,stage", [
    (5, 90, "Introduction"), (9, 5, "Introduction"), (10, 60, "Growth"), (30, 59.9, "Maturity"),
    (30, 40, "Maturity"), (30, 39.9, "Decline"), (30, 20, "Decline"), (30, 19.9, "Exit"), (30, None, None)])
def test_determine_lifecycle(age, lss, stage):
    assert calc.determine_lifecycle(age, lss)["stage"] == stage


def test_confirmation_rule_two_months():
    seq = ["Maturity", "Maturity", "Decline", "Decline", "Exit"]
    res = calc.apply_confirmation_rule(seq, 2)
    assert [r["policy_stage"] for r in res] == ["Maturity", "Maturity", "Maturity", "Decline", "Decline"]
    assert res[2]["pending"] == "Decline" and res[4]["pending"] == "Exit"
    off = calc.apply_confirmation_rule(seq, 2, enabled=False)
    assert [r["policy_stage"] for r in off] == seq


# ---- portfolio engine --------------------------------------------------------
def test_demo_covers_all_stages(portfolio):
    out = calc.evaluate_portfolio(portfolio)
    assert set(calc.STAGES) <= set(out["table"]["Lifecycle Stage"])


def test_lss_recalculates_and_stage_changes(portfolio):
    """Req 4, 5, 7: editing inputs changes LSS, stage and strategy."""
    pid = by_name(portfolio, "Apex Ultima")
    base = calc.evaluate_portfolio(portfolio)["results"][pid]
    p = portfolio[pid]
    p["input_mode"] = "manual"
    p.update(current_t3m=50, t3m_y1=100, t3m_y2=110, forecast=100, actual=40, active_points=100, total_points=12000,
             current_volume=50, peak_volume=400)
    new = calc.evaluate_portfolio(portfolio, calc.EngineSettings(confirm_enabled=False))["results"][pid]
    assert new["lss"].value != base["lss"].value
    assert new["stage_info"]["policy_stage"] == "Exit"
    assert new["strategy"]["headline"] != base["strategy"]["headline"]


def test_safety_stock_responds_to_inventory_inputs(portfolio):
    """Req 6."""
    pid = by_name(portfolio, "Royale Glitz")
    a = calc.evaluate_portfolio(portfolio)["results"][pid]["inventory"]["safety_stock"].value
    portfolio[pid]["std_daily_demand"] *= 2
    b = calc.evaluate_portfolio(portfolio)["results"][pid]["inventory"]["safety_stock"].value
    portfolio[pid]["lead_time_local_days"] += 10
    c = calc.evaluate_portfolio(portfolio)["results"][pid]["inventory"]["safety_stock"].value
    assert a < b < c


def test_add_and_delete_update_portfolio(portfolio):
    """Req 1, 3, 8."""
    n = len(calc.evaluate_portfolio(portfolio)["table"])
    new = demo_data.blank_product()
    new.update(name="Test SKU", age_months=30, current_t3m=100, t3m_y1=90, t3m_y2=85, forecast=100, actual=95,
               active_points=500, total_points=1000, current_volume=100, peak_volume=120,
               avg_daily_demand=10, std_daily_demand=3, lead_time_local_days=7)
    portfolio[new["id"]] = new
    out = calc.evaluate_portfolio(portfolio)
    assert len(out["table"]) == n + 1
    assert out["results"][new["id"]]["lss"].ok
    del portfolio[new["id"]]
    assert len(calc.evaluate_portfolio(portfolio)["table"]) == n


def test_invalid_inputs_give_na_not_zero():
    """Req 11."""
    p = demo_data.blank_product()
    p.update(name="Bad", age_months=40, current_t3m=-5, t3m_y1=100, t3m_y2=100, active_points=2000, total_points=1000,
             avg_daily_demand=10, std_daily_demand=2, lead_time_local_days=0)
    out = calc.evaluate_portfolio({p["id"]: p})
    r = out["results"][p["id"]]
    assert not r["signals"]["momentum"].ok and r["signals"]["momentum"].value is None
    assert not r["signals"]["reach"].ok
    assert not r["signals"]["predictability"].ok          # missing forecast/actual
    assert not r["lss"].ok and r["stage_info"]["policy_stage"] is None
    assert not r["inventory"]["safety_stock"].ok          # lead time 0
    msgs = " ".join(i["message"] for i in r["issues"])
    assert "negative" in msgs and "exceed" in msgs and "lead time" in msgs and "forecast" in msgs.lower()


def test_missing_history_flagged():
    p = demo_data.blank_product()
    p.update(name="No history", input_mode="history", history=[])
    out = calc.evaluate_portfolio({p["id"]: p})
    assert any("Missing historical data" in i["message"] for i in out["results"][p["id"]]["issues"])


def test_formula_change_propagates(portfolio, monkeypatch):
    """Req 12: patching one function in calculations.py flows through everything."""
    pid = by_name(portfolio, "Royale Glitz")
    before = calc.evaluate_portfolio(portfolio)
    monkeypatch.setitem(calc.SIGNAL_FUNCTIONS, "momentum",
                        lambda d: calc.Metric("Momentum", raw=0.0, formula="flat"))
    after = calc.evaluate_portfolio(portfolio)
    assert after["results"][pid]["signals"]["momentum"].formula == "flat"
    assert after["results"][pid]["lss"].value != before["results"][pid]["lss"].value
    assert not after["table"]["LSS"].equals(before["table"]["LSS"])


def test_monthly_engine_and_pending_transition(portfolio):
    pid = by_name(portfolio, "Emporio")
    r = calc.evaluate_portfolio(portfolio)["results"][pid]
    assert r["stage_info"]["signal_stage"] == "Decline"
    assert r["stage_info"]["policy_stage"] == "Maturity"   # held by the 2-month rule
    one = calc.evaluate_portfolio(portfolio, calc.EngineSettings(confirm_months=1))["results"][pid]
    assert one["stage_info"]["policy_stage"] == "Decline"
    assert len(r["monthly"]) == 36 and r["monthly"]["stage_changed"].iloc[-1]


def test_percentile_fallback_with_few_products():
    p = demo_data.blank_product()
    p.update(name="Solo", age_months=30, current_t3m=110, t3m_y1=100, t3m_y2=95, forecast=100, actual=90,
             active_points=300, total_points=1000, current_volume=110, peak_volume=130)
    r = calc.evaluate_portfolio({p["id"]: p})["results"][p["id"]]
    assert r["lss"].ok and r["signals"]["momentum"].basis == "Assumption"
