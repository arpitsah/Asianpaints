"""
components.py - presentation helpers (styling, cards, badges, Plotly charts).
No business formulas live here: every number shown comes from calculations.py.
"""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calculations as calc

INK = "#14213D"
INK_2 = "#4B5563"
MUTED = "#8A94A6"
LINE = "#E5E7EB"
ACCENT = "#1F4E9E"
SURFACE = "#FFFFFF"
NA_TEXT = "N/A — insufficient data"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}
.block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}
.app-header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;
  border-bottom: 1px solid {LINE}; padding-bottom: .8rem; margin-bottom: .6rem; }}
.app-eyebrow {{ color:{ACCENT}; font-weight:700; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; }}
.app-title {{ color:{INK}; font-weight:800; font-size:1.65rem; line-height:1.15; margin:.15rem 0 0; }}
.app-sub {{ color:{INK_2}; font-size:.9rem; margin-top:.2rem; }}
.demo-pill {{ background:#FFF7E6; color:#7A4B00; border:1px solid #F3D19C; border-radius:999px;
  font-size:.72rem; padding:.2rem .6rem; font-weight:600; }}
.kpi {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:12px; padding:.85rem 1rem; min-height:118px; margin-bottom:.5rem; }}
.kpi-label {{ color:{INK_2}; font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }}
.kpi-value {{ color:{INK}; font-size:1.6rem; font-weight:800; line-height:1.2; margin-top:.15rem; }}
.kpi-value.small {{ font-size:1.05rem; line-height:1.3; }}
.kpi-value.na {{ font-size:.95rem; color:{MUTED}; font-weight:600; }}
.kpi-sub {{ color:{MUTED}; font-size:.78rem; margin-top:.15rem; }}
.kpi-bar {{ height:4px; border-radius:4px; margin:-.85rem -1rem .7rem; }}
.badge {{ display:inline-flex; align-items:center; gap:.4rem; border-radius:999px; padding:.18rem .7rem;
  font-weight:700; font-size:.82rem; color:{INK}; border:1px solid; background:#fff; white-space:nowrap; }}
.badge .dot {{ width:.62rem; height:.62rem; border-radius:50%; display:inline-block; }}
.hero {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:14px; padding:1.1rem 1.3rem; }}
.hero-num {{ font-size:3rem; font-weight:800; color:{INK}; line-height:1; }}
.hero-den {{ font-size:1.1rem; color:{MUTED}; font-weight:600; }}
.hero-na {{ font-size:1.3rem; font-weight:700; color:{MUTED}; }}
.why {{ background:#F6F8FB; border-left:3px solid {ACCENT}; border-radius:6px; padding:.6rem .8rem;
  color:{INK}; font-size:.88rem; margin-top:.6rem; }}
.flow {{ display:flex; flex-wrap:wrap; gap:.35rem; align-items:stretch; margin:.3rem 0 1rem; }}
.flow-step {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:10px; padding:.45rem .7rem; min-width:92px; }}
.flow-step .k {{ font-size:.66rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }}
.flow-step .v {{ font-size:.92rem; color:{INK}; font-weight:700; }}
.flow-arrow {{ color:{MUTED}; align-self:center; font-weight:700; }}
.policy-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.6rem; }}
.policy-cell {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:10px; padding:.6rem .75rem; }}
.policy-cell .k {{ font-size:.7rem; color:{MUTED}; text-transform:uppercase; letter-spacing:.07em; font-weight:700; }}
.policy-cell .v {{ font-size:.92rem; color:{INK}; font-weight:600; margin-top:.15rem; }}
.section-title {{ font-size:1.02rem; font-weight:700; color:{INK}; margin:1.1rem 0 .45rem; }}
.section-cap {{ color:{INK_2}; font-size:.84rem; margin-top:-.3rem; margin-bottom:.5rem; }}
.action {{ border:1px solid {LINE}; border-radius:10px; padding:.55rem .75rem; margin-bottom:.4rem; background:{SURFACE};
  color:{INK}; font-size:.9rem; }}
.action b {{ color:{ACCENT}; }}
.layer {{ border-radius:12px; padding:.9rem 1rem; color:#fff; }}
.layer h4 {{ color:#fff; margin:0 0 .2rem; font-size:.95rem; }}
.layer p {{ margin:0; font-size:.82rem; opacity:.9; }}
.assump {{ font-size:.72rem; color:#7A4B00; background:#FFF7E6; border-radius:4px; padding:.05rem .35rem; font-weight:600; }}
div[data-testid="stExpander"] details summary p {{ font-weight:600; color:{INK}; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def fmt_num(v, fmt="{:,.0f}", na=NA_TEXT) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return na
    return fmt.format(v)


def fmt_rs(v, na=NA_TEXT) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return na
    if abs(v) >= 1e7:
        return f"₹{v / 1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v / 1e5:,.1f} L"
    return f"₹{v:,.0f}"


def stage_color(stage) -> str:
    return calc.STAGE_COLORS.get(stage or "N/A", calc.STAGE_COLORS["N/A"])


def badge(stage, suffix: str = "") -> str:
    label = stage or "N/A"
    c = stage_color(stage)
    return (f'<span class="badge" style="border-color:{c}"><span class="dot" style="background:{c}"></span>'
            f'{esc(label)}{esc(suffix)}</span>')


def kpi(label, value, sub="", color=None, na=False, small=False):
    bar = f'<div class="kpi-bar" style="background:{color}"></div>' if color else ""
    cls = "kpi-value na" if na else ("kpi-value small" if small else "kpi-value")
    st.markdown(f'<div class="kpi">{bar}<div class="kpi-label">{esc(label)}</div>'
                f'<div class="{cls}">{esc(value)}</div><div class="kpi-sub">{esc(sub)}</div></div>',
                unsafe_allow_html=True)


def metric_kpi(label, m: calc.Metric, fmt="{:,.0f}", sub="", color=None):
    if m is None or not m.ok:
        kpi(label, NA_TEXT if not (m and m.status == "pending") else "Formula pending",
            sub or (m.reason if m else ""), color, na=True)
    else:
        kpi(label, fmt.format(m.value), sub, color)


def section(title, caption=""):
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-cap">{esc(caption)}</div>', unsafe_allow_html=True)


def flow_strip(steps: list[tuple[str, str]]):
    parts = []
    for i, (k, v) in enumerate(steps):
        if i:
            parts.append('<span class="flow-arrow">→</span>')
        parts.append(f'<div class="flow-step"><div class="k">{esc(k)}</div><div class="v">{v}</div></div>')
    st.markdown(f'<div class="flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def policy_grid(items: list[tuple[str, str]]):
    cells = "".join(f'<div class="policy-cell"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
                    for k, v in items)
    st.markdown(f'<div class="policy-grid">{cells}</div>', unsafe_allow_html=True)


def _fmt_input(v):
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    if isinstance(v, list):
        return ", ".join(_fmt_input(x) for x in v) if v else "—"
    return "—" if v is None else str(v)


def calc_panel(m: calc.Metric, title: str | None = None, result_fmt="{:,.1f}", raw_fmt=None, expanded=False):
    """Standard 'View calculation' expander for any Metric."""
    with st.expander(title or f"View calculation — {m.name}", expanded=expanded):
        if m.status == "pending":
            st.warning("Formula pending — placeholder in calculations.py.")
        st.markdown(f"**Formula** &nbsp; `{m.formula}`" + (
            ' &nbsp; <span class="assump">Assumption</span>' if m.basis == "Assumption" else ""),
            unsafe_allow_html=True)
        if m.inputs:
            st.markdown("**Inputs**")
            st.dataframe(pd.DataFrame({"Input": list(m.inputs.keys()),
                                       "Value": [_fmt_input(v) for v in m.inputs.values()]}),
                         hide_index=True, width="stretch")
        steps = m.steps if isinstance(m.steps, list) else []
        if steps:
            st.markdown("**Working**")
            st.markdown("\n".join(f"{i + 1}. {esc(s)}" for i, s in enumerate(steps)))
        if m.raw is not None and raw_fmt:
            st.markdown(f"**Raw result:** {raw_fmt.format(m.raw)}")
        if m.ok:
            st.markdown(f"**Final:** {result_fmt.format(m.value)}")
        else:
            st.warning(f"{NA_TEXT}: {m.reason}")


# ---------------------------------------------------------------- charts
def base_layout(fig: go.Figure, title: str = "", height: int = 340, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=INK), x=0, xanchor="left"),
        height=height, margin=dict(l=64, r=16, t=48 if title else 12, b=56),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=INK_2, size=12),
        showlegend=legend, legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=11), traceorder="normal"),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=LINE),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=INK_2), automargin=True)
    fig.update_yaxes(gridcolor="#F0F2F5", zeroline=False, linecolor=LINE, tickfont=dict(color=INK_2),
                     automargin=True)
    return fig


def add_stage_bands(fig: go.Figure, axis="y", opacity=0.06):
    t = calc.LIFECYCLE_THRESHOLDS
    bands = [("Growth", t["growth_min_lss"], 100), ("Maturity", t["maturity_min_lss"], t["growth_min_lss"]),
             ("Decline", t["decline_min_lss"], t["maturity_min_lss"]), ("Exit", 0, t["decline_min_lss"])]
    for stage, lo, hi in bands:
        kw = dict(fillcolor=stage_color(stage), opacity=opacity, line_width=0, layer="below")
        if axis == "y":
            fig.add_hrect(y0=lo, y1=hi, **kw)
            fig.add_annotation(xref="paper", x=1, y=(lo + hi) / 2, text=stage, showarrow=False,
                               xanchor="right", font=dict(size=10, color=MUTED))
        else:
            fig.add_vrect(x0=lo, x1=hi, **kw)
    return fig


def contribution_chart(contribs: dict, weights: dict):
    labels = [calc.SIGNAL_LABELS[s] for s in calc.SIGNALS]
    vals = [contribs.get(s) for s in calc.SIGNALS]
    maxes = [weights[s] * 100 for s in calc.SIGNALS]
    fig = go.Figure()
    fig.add_bar(y=labels, x=maxes, orientation="h", marker_color="#EEF1F6", name="Max possible",
                hovertemplate="%{y}: max %{x:.0f} pts<extra></extra>")
    fig.add_bar(y=labels, x=[v or 0 for v in vals], orientation="h", marker_color=ACCENT, name="Contribution",
                text=[f"{v:.1f}" if v is not None else "N/A" for v in vals], textposition="outside",
                hovertemplate="%{y}: %{x:.1f} LSS pts<extra></extra>")
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 40])
    base_layout(fig, "Weighted contribution to LSS (points of 100)", 260, legend=True)
    fig.update_yaxes(automargin=True, ticklabelstandoff=6)
    fig.update_layout(margin=dict(l=120, r=16, t=48, b=40))
    return fig
