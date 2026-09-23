"""UI tests with Streamlit's AppTest - the real app, driven headlessly."""
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def at():
    return AppTest.from_file("app.py", default_timeout=90).run()


def names(at):
    return {p["name"] for p in at.session_state.products.values()}


def test_app_renders_all_tabs(at):
    assert not at.exception
    assert [t.label for t in at.tabs] == ["LSS Calculator", "Inventory Calculator", "Strategy", "Portfolio",
                                         "Lifecycle Dashboard", "Product Detail"]


def test_add_product_via_form(at):
    at.button(key="calc_add").click().run()
    new_id = at.session_state["calc_new"]["id"]
    k = f"calc_{new_id}"
    at.radio(key=f"{k}_mode").set_value("manual").run()
    at.text_input(key=f"{k}_name").input("Demo Added SKU")
    vals = {"age": 30, "current_t3m": 1200, "t3m_y1": 1000, "t3m_y2": 950, "forecast": 1000, "actual": 950,
            "active_points": 400, "total_points": 1000, "current_volume": 1200, "peak_volume": 1300,
            "avg_daily_demand": 13, "std_daily_demand": 4, "lead_time_local_days": 7}
    for f, v in vals.items():
        at.number_input(key=f"{k}_{f}").set_value(v)
    at.button(key=f"FormSubmitter:{k}_form-Save product").click().run()
    assert not at.exception
    assert "Demo Added SKU" in names(at)
    assert at.session_state.selected_id == new_id
    row = at.session_state._engine_out["table"].set_index("id").loc[new_id]
    assert row["LSS"] > 0 and row["Lifecycle Stage"] in ("Growth", "Maturity", "Decline", "Exit")


def test_edit_product_changes_lss(at):
    pid = at.session_state.selected_id
    before = at.session_state._engine_out["results"][pid]["lss"].value
    at.button(key="calc_edit").click().run()
    k = f"calc_{pid}"
    at.radio(key=f"{k}_mode").set_value("manual").run()
    for f, v in {"current_t3m": 300, "t3m_y1": 1000, "t3m_y2": 1100, "forecast": 1000, "actual": 300,
                 "active_points": 50, "total_points": 12000, "current_volume": 300, "peak_volume": 5000}.items():
        at.number_input(key=f"{k}_{f}").set_value(v)
    at.button(key=f"FormSubmitter:{k}_form-Save product").click().run()
    assert not at.exception
    after = at.session_state._engine_out["results"][pid]
    assert after["lss"].value != before
    assert after["stage_info"]["signal_stage"] in ("Decline", "Exit")


def test_delete_product(at):
    pid = at.session_state.selected_id
    name = at.session_state.products[pid]["name"]
    n = len(at.session_state.products)
    at.button(key="calc_delete").click().run()
    at.button(key="calc_del_yes").click().run()
    assert not at.exception
    assert len(at.session_state.products) == n - 1 and name not in names(at)
    assert len(at.session_state._engine_out["table"]) == n - 1


def test_inventory_form_changes_safety_stock(at):
    pid = at.session_state.selected_id
    before = at.session_state._engine_out["results"][pid]["inventory"]["safety_stock"].value
    at.number_input(key=f"inv{pid}_lead_time_local_days").set_value(30)
    at.button(key=f"FormSubmitter:inv_form_{pid}-Apply inventory inputs").click().run()
    assert not at.exception
    assert at.session_state._engine_out["results"][pid]["inventory"]["safety_stock"].value > before


def test_momentum_strict_mode(at):
    at.radio(key="set_mom_fb").set_value("strict").run()
    assert not at.exception
    res = at.session_state._engine_out["results"]
    royale = next(r for r in res.values() if r["product"]["name"].startswith("Royale"))
    assert not royale["signals"]["momentum"].ok and royale["stage_info"]["policy_stage"] is None


def test_settings_propagate(at):
    at.slider(key="set_confirm_months").set_value(1).run()
    res = at.session_state._engine_out["results"]
    emporio = next(r for r in res.values() if r["product"]["name"].startswith("Emporio"))
    assert emporio["stage_info"]["policy_stage"] == "Decline"
    assert not at.exception


def test_remove_demo_then_empty_state(at):
    at.sidebar.button[1].click().run()   # "Remove demo"
    assert not at.exception
    assert len(at.session_state.products) == 0


def test_csv_download_content(at):
    import calculations as calc
    out = at.session_state._engine_out
    csv = out["table"].drop(columns=["id"]).to_csv(index=False)
    assert csv.splitlines()[0].startswith("Product,Category,Channel,Age,Momentum")
    assert len(csv.splitlines()) == len(out["table"]) + 1
