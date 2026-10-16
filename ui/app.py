"""AT-FedBN cardiovascular research prototype dashboard.

Run with ``streamlit run ui/app.py`` from the project root.
"""

import json
import html
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import yaml

st.set_page_config(page_title="Cardiovascular Risk Assessment | AT-FedBN", page_icon="+",
                   layout="wide", initial_sidebar_state="collapsed")

theme = st.session_state.get("ui_theme", "Dark")
if theme == "Dark":
    palette = {"app": "#0b1220", "surface": "#111c2e", "surface2": "#172438", "text": "#f1f5f9",
               "muted": "#9aacc0", "line": "#293a50", "accent": "#4fd1c5", "accent_text": "#08211f",
               "info": "#102b34", "warn": "#352918", "error": "#351f27", "success": "#183128",
               "grid": "#293a50", "bar1": "#4fd1c5", "bar2": "#7da5e8", "bar3": "#e4a66a", "bar4": "#b39ad7"}
else:
    palette = {"app": "#f7f9fb", "surface": "#ffffff", "surface2": "#edf3f6", "text": "#1d2b36",
               "muted": "#536675", "line": "#d5e0e7", "accent": "#087f8c", "accent_text": "#ffffff",
               "info": "#eaf3f6", "warn": "#fff3dc", "error": "#fbeaec", "success": "#e8f4ed",
               "grid": "#d9e2e8", "bar1": "#087f8c", "bar2": "#4d6f9e", "bar3": "#9a6335", "bar4": "#73568f"}

st.markdown(f"""
<style>
:root {{ --app:{palette['app']}; --surface:{palette['surface']}; --surface2:{palette['surface2']};
--text:{palette['text']}; --muted:{palette['muted']}; --line:{palette['line']}; --accent:{palette['accent']};
--accent-text:{palette['accent_text']}; --info:{palette['info']}; --warn:{palette['warn']};
--error:{palette['error']}; --success:{palette['success']}; }}
.stApp, [data-testid="stAppViewContainer"] {{ background:var(--app); color:var(--text); }}
.main .block-container {{ max-width:1240px; padding-top:.65rem; padding-bottom:2.5rem; }}
.stApp, .stApp p, .stApp label, .stApp li, .stApp small {{ color:var(--text); }}
h1,h2,h3,h4 {{ color:var(--text) !important; letter-spacing:-.02em; }}
h1 {{ font-size:2.15rem; }} h2 {{ font-size:1.4rem; margin-top:1.1rem; }}
.eyebrow {{ color:var(--accent); font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }}
.subtle {{ color:var(--muted) !important; font-size:1rem; line-height:1.55; }}
.notice,.result-card,.step-detail {{ background:var(--surface); color:var(--text); border:1px solid var(--line); border-radius:10px; }}
.notice {{ background:var(--info); border-left:4px solid var(--accent); padding:.7rem .9rem; margin:.55rem 0 1rem; }}
.status-message {{ border:1px solid var(--line); border-left:4px solid; border-radius:8px; padding:.65rem .85rem; margin:.45rem 0 .8rem; color:var(--text); }}
.status-info {{ background:var(--info); border-left-color:var(--accent); }}
.status-warning {{ background:var(--warn); border-left-color:#ad792c; }}
.status-error {{ background:var(--error); border-left-color:#b34e58; }}
.status-success {{ background:var(--success); border-left-color:#31805c; }}
.result-card {{ padding:1.05rem 1.2rem; margin:.5rem 0; }}
.step-detail {{ background:var(--surface2); padding:.65rem .85rem; margin:.45rem 0 .9rem; }}
.step-detail p {{ margin:0; color:var(--text); }}
section[data-testid="stSidebar"] {{ display:none; }}
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--line); padding:.65rem .85rem; border-radius:9px; }}
[data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {{ color:var(--muted) !important; }}
[data-testid="stWidgetLabel"] p {{ color:var(--text) !important; }}
pre, code {{ color:var(--text); background:var(--surface2); }}
[data-testid="stHeader"] {{ background:var(--app); }}
input[type="radio"] {{ accent-color:var(--accent); }}
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div {{ background:var(--surface) !important; color:var(--text) !important; border-color:var(--line) !important; }}
[data-baseweb="popover"], [role="listbox"], [role="option"] {{ background:var(--surface) !important; color:var(--text) !important; }}
[role="option"]:hover {{ background:var(--surface2) !important; }}
.stButton > button {{ background:var(--surface); color:var(--text); border:1px solid var(--line); border-radius:8px; min-height:2.45rem; }}
.stButton > button:hover {{ border-color:var(--accent); color:var(--accent); }}
.stButton > button[kind="primary"] {{ background:var(--accent); color:var(--accent-text); border-color:var(--accent); }}
.stButton > button[kind="primary"]:hover {{ background:var(--accent); color:var(--accent-text); filter:brightness(.94); }}
.stFormSubmitButton > button {{ background:var(--accent); color:var(--accent-text); border-color:var(--accent); font-weight:700; }}
.stFormSubmitButton > button:hover {{ background:var(--accent); color:var(--accent-text); filter:brightness(.94); }}
[data-testid="stAlert"] {{ color:var(--text); background:var(--surface2); border-color:var(--line); }}
[data-testid="stAlert"] p {{ color:var(--text) !important; }}
div[data-testid="stAlert"]:has(svg[viewBox]) {{ border-radius:9px; }}
.flow-lane {{ color:var(--muted); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin:.5rem 0 .25rem; }}
.flow-arrow {{ color:var(--accent); font-size:1.2rem; font-weight:700; text-align:center; padding-top:1.05rem; }}
.flow-turn {{ color:var(--accent); font-size:1.1rem; font-weight:700; text-align:right; padding-right:8%; margin:-.4rem 0; }}
.architecture-label {{ color:var(--muted); font-size:.78rem; }}
.stButton > button {{ white-space:pre-line; line-height:1.25; }}
.site-brand {{ color:var(--text); font-size:1.05rem; font-weight:800; letter-spacing:-.03em; padding:.38rem 0; }}
.site-brand span {{ display:block; color:var(--muted); font-size:.68rem; font-weight:500; letter-spacing:.02em; margin-top:2px; }}
.stForm {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; }}
.data-table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; margin:.3rem 0 .7rem; }}
table.data-table {{ width:100%; border-collapse:collapse; color:var(--text); background:var(--surface); font-size:.86rem; }}
.data-table th {{ text-align:left; color:var(--text); background:var(--surface2); font-weight:700; }}
.data-table th,.data-table td {{ padding:.55rem .7rem; border-bottom:1px solid var(--line); }}
.data-table tr:last-child td {{ border-bottom:0; }}
.chart-card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.55rem .7rem; margin:.4rem 0 .8rem; }}
.chart-card svg {{ display:block; width:100%; height:auto; }}
.chart-title {{ fill:var(--text); font:600 14px sans-serif; }}
.chart-label {{ fill:var(--text); font:12px sans-serif; }}
.chart-muted {{ fill:var(--muted); font:11px sans-serif; }}
.chart-grid {{ stroke:{palette['grid']}; stroke-width:1; }}
.bar-0 {{ fill:{palette['bar1']}; }} .bar-1 {{ fill:{palette['bar2']}; }} .bar-2 {{ fill:{palette['bar3']}; }} .bar-3 {{ fill:{palette['bar4']}; }}
.prob-track {{ background:var(--surface2); border-radius:99px; height:12px; overflow:hidden; margin:.4rem 0; }}
.prob-fill {{ background:var(--accent); height:100%; border-radius:99px; }}
@media (max-width:700px) {{ .main .block-container {{padding-left:1rem;padding-right:1rem;}} h1 {{font-size:1.75rem;}} }}
</style>
""", unsafe_allow_html=True)


def project_path(path):
    return PROJECT_ROOT / path


@st.cache_data(show_spinner=False)
def read_json(name):
    path = project_path("outputs/results") / name
    try:
        with path.open() as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def read_config():
    try:
        with project_path("config.yaml").open() as stream:
            return yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return {}


@st.cache_data(show_spinner="Reading dataset summaries…")
def dataset_summary(num_clients):
    rows, total_train, total_test = [], 0, 0
    root = project_path("data/processed")
    for client in range(num_clients):
        train_path, test_path = root / f"hospital_{client}_train.csv", root / f"hospital_{client}_test.csv"
        if not train_path.exists() or not test_path.exists():
            continue
        train, test = pd.read_csv(train_path), pd.read_csv(test_path)
        if "cardio" not in train or "cardio" not in test:
            continue
        rows.append({"Hospital": f"Hospital {client + 1}", "Training samples": len(train),
                     "Test samples": len(test), "Training positive rate": train.cardio.mean(),
                     "Test positive rate": test.cardio.mean()})
        total_train += len(train)
        total_test += len(test)
    return rows, total_train, total_test


def metric_value(result, location, name):
    node = result or {}
    for part in location.split("."):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    value = node.get(name) if isinstance(node, dict) else None
    return value if isinstance(value, (int, float)) else None


def pct(value):
    return "Unavailable" if value is None else f"{value:.1%}"


def status_message(kind, message):
    if kind not in {"info", "warning", "error", "success"}:
        kind = "info"
    st.markdown(f'<div class="status-message status-{kind}">{html.escape(str(message))}</div>', unsafe_allow_html=True)


def select_button_value(state_key, value):
    st.session_state[state_key] = value


def button_choices(options, state_key, columns_per_row=3):
    """Small persistent button set that changes the displayed chart."""
    st.session_state.setdefault(state_key, options[0])
    if st.session_state[state_key] not in options:
        st.session_state[state_key] = options[0]
    selected = st.session_state[state_key]
    for start in range(0, len(options), columns_per_row):
        row = options[start:start + columns_per_row]
        columns = st.columns(len(row), gap="small")
        for offset, option in enumerate(row):
            index = start + offset
            with columns[offset]:
                st.button(option, key=f"{state_key}_choice_{index}",
                          type="primary" if option == selected else "secondary",
                          use_container_width=True, on_click=select_button_value,
                          args=(state_key, option))
    return st.session_state[state_key]


def render_table(frame, percent_columns=(), decimal_columns=()):
    """Render a compact static table that inherits the active dashboard theme."""
    view = frame.copy()
    for column in percent_columns:
        if column in view:
            view[column] = view[column].map(lambda value: "—" if pd.isna(value) else f"{value:.1%}")
    for column in decimal_columns:
        if column in view:
            view[column] = view[column].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    markup = view.to_html(index=False, escape=True, classes="data-table", border=0)
    st.markdown(f'<div class="data-table-wrap">{markup}</div>', unsafe_allow_html=True)


def render_grouped_chart(title, categories, series, maximum=None, value_format="number"):
    """Draw a responsive, theme-aware grouped bar chart from saved/data-derived values."""
    categories = list(categories)
    series = [(str(name), list(values)) for name, values in series]
    valid = [float(value) for _, values in series for value in values if value is not None and pd.notna(value)]
    if not categories or not series or not valid:
        status_message("info", "No saved values are available for this chart.")
        return
    maximum = max(1.0, max(valid)) if maximum is None else float(maximum)
    maximum = maximum or 1.0
    left, right, top = 145, 28, 56
    group_height = max(55, 20 * len(series) + 20)
    height = top + len(categories) * group_height + 40
    plot_width = 900 - left - right
    pieces = [f'<div class="chart-card"><svg viewBox="0 0 900 {height}" role="img" aria-label="{html.escape(title)}">',
              f'<text x="{left}" y="24" class="chart-title">{html.escape(title)}</text>']
    for tick in range(5):
        value = maximum * tick / 4
        x = left + plot_width * tick / 4
        label = f"{value:.0%}" if value_format == "percent" else f"{value:,.0f}"
        pieces.extend([f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-28}" class="chart-grid"/>',
                       f'<text x="{x:.1f}" y="{height-8}" text-anchor="middle" class="chart-muted">{label}</text>'])
    for index, (name, _) in enumerate(series):
        x = left + index * 180
        pieces.extend([f'<rect x="{x}" y="38" width="10" height="10" class="bar-{index % 4}"/>',
                       f'<text x="{x+15}" y="47" class="chart-muted">{html.escape(name)}</text>'])
    for cat_index, category in enumerate(categories):
        y0 = top + cat_index * group_height + 7
        pieces.append(f'<text x="{left-10}" y="{y0 + len(series)*16/2 + 4:.1f}" text-anchor="end" class="chart-label">{html.escape(str(category))}</text>')
        for serie_index, (name, values) in enumerate(series):
            value = values[cat_index] if cat_index < len(values) else None
            if value is None or pd.isna(value):
                continue
            value = max(0.0, float(value))
            width = plot_width * min(value, maximum) / maximum
            y = y0 + serie_index * 16
            display_value = f"{value:.1%}" if value_format == "percent" else f"{value:,.0f}"
            pieces.append(f'<rect x="{left}" y="{y}" width="{width:.2f}" height="11" rx="3" class="bar-{serie_index % 4}"><title>{html.escape(str(category))} · {html.escape(name)}: {display_value}</title></rect>')
    pieces.append("</svg></div>")
    st.markdown("".join(pieces), unsafe_allow_html=True)


def node_flow(title, stages, descriptions, key, zones=None):
    st.markdown(f'<div class="flow-lane">{html.escape(title)} · select a step</div>', unsafe_allow_html=True)
    selected_index = st.session_state.get(f"{key}_selected", 0)
    route = list(range(len(stages)))
    if len(stages) > 4:
        route = list(range(4)) + list(range(len(stages) - 1, 3, -1))
    for row_start in range(0, len(route), 4):
        row = route[row_start:row_start + 4]
        widths = []
        for position in range(len(row)):
            widths.append(1)
            if position < len(row) - 1:
                widths.append(.13)
        columns = st.columns(widths, gap="small")
        for position, stage_index in enumerate(row):
            stage, subtitle = stages[stage_index]
            zone = zones[stage_index] if zones else "Process"
            label = f"{stage_index + 1:02d} · {stage}\n{zone} · {subtitle}"
            with columns[position * 2]:
                st.button(label, key=f"{key}_node_{stage_index}", use_container_width=True,
                          type="primary" if stage_index == selected_index else "secondary",
                          on_click=lambda selected=stage_index, state_key=f"{key}_selected": st.session_state.__setitem__(state_key, selected))
                if row_start < 4 and position == len(row) - 1 and row_start + 4 < len(route):
                    st.markdown('<div class="flow-turn">↓</div>', unsafe_allow_html=True)
            if position < len(row) - 1:
                if row_start >= 4 and position > 0:
                    arrow = "←"
                elif row_start < 4:
                    arrow = "→"
                else:
                    continue
                with columns[position * 2 + 1]:
                    st.markdown(f'<div class="flow-arrow">{arrow}</div>', unsafe_allow_html=True)
    selected_index = st.session_state.get(f"{key}_selected", 0)
    selected = stages[selected_index][0]
    explanation = descriptions.get(selected, "")
    st.markdown(f'<div class="step-detail"><strong>{html.escape(selected)}</strong><p>{html.escape(explanation)}</p></div>', unsafe_allow_html=True)


cfg = read_config()
data_cfg = cfg.get("data", {})
model_cfg = cfg.get("model", {})
training_cfg = cfg.get("training", {})
fed_cfg = cfg.get("federated", {})
privacy_cfg = cfg.get("privacy", {})
num_clients = int(data_cfg.get("num_clients", 0))
hospital_rows, total_train, total_test = dataset_summary(num_clients)

st.session_state.setdefault("active_page", "Risk Assessment")
st.session_state.setdefault("ui_theme", "Dark")


def choose_page(name):
    st.session_state["active_page"] = name


def toggle_theme():
    st.session_state["ui_theme"] = "Light" if st.session_state.get("ui_theme") == "Dark" else "Dark"


page = st.session_state["active_page"]
brand_col, risk_col, data_col, method_col, theme_col = st.columns([1.75, 1.2, 1.4, 1.7, 1.15], gap="small")
with brand_col:
    st.markdown('<div class="site-brand">AT-FedBN<span>Cardiovascular research</span></div>', unsafe_allow_html=True)
with risk_col:
    st.button("Risk Assessment", key="top_nav_risk", type="primary" if page == "Risk Assessment" else "secondary",
              use_container_width=True, on_click=choose_page, args=("Risk Assessment",))
with data_col:
    st.button("Model & Dataset", key="top_nav_data", type="primary" if page == "Model & Dataset" else "secondary",
              use_container_width=True, on_click=choose_page, args=("Model & Dataset",))
with method_col:
    st.button("AT-FedBN Methodology", key="top_nav_method", type="primary" if page == "AT-FedBN Methodology" else "secondary",
              use_container_width=True, on_click=choose_page, args=("AT-FedBN Methodology",))
with theme_col:
    st.button("Switch to light" if theme == "Dark" else "Switch to dark", key="theme_toggle",
              use_container_width=True, on_click=toggle_theme)
st.markdown("<div style='height:1px;background:var(--line);margin:.25rem 0 1.3rem'></div>", unsafe_allow_html=True)


def model_performance():
    st.markdown("### Model information")
    at = read_json("at_fedbn.json")
    checkpoint_candidates = ["at_fedbn_final.pth", "at_fedbn_ui_live_demo.pth", "fedavg_baseline_final.pth"]
    checkpoint = next((name for name in checkpoint_candidates if (project_path("outputs/models") / name).is_file()), None)
    model_label = Path(checkpoint).stem if checkpoint else "No supported checkpoint found"
    a, b, c, d = st.columns(4)
    a.metric("Prediction checkpoint", model_label)
    b.metric("Simulated hospitals", str(num_clients) if num_clients else "Unavailable")
    c.metric("Saved accuracy", pct(metric_value(at, "local.average", "accuracy")))
    d.metric("Saved ROC-AUC", pct(metric_value(at, "local.average", "roc_auc")))
    st.caption("AT-FedBN averages of client-local evaluation results. These experiment metrics are separate from the estimated probability for an individual assessment.")


if page == "Risk Assessment":
    st.markdown("<div class='eyebrow'>Patient assessment</div>", unsafe_allow_html=True)
    st.title("Cardiovascular Risk Assessment")
    st.markdown("<p class='subtle'>Enter health indicators to receive a model-based cardiovascular disease prediction.</p>", unsafe_allow_html=True)
    st.markdown("<div class='notice'><strong>Research prototype:</strong> This tool is for education and research. It has not been clinically validated and is not suitable for diagnosis or treatment decisions.</div>", unsafe_allow_html=True)

    with st.form("risk_assessment_form"):
        st.markdown("#### Personal information")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 18, 100, 50, 1)
        with c2:
            gender = st.selectbox("Sex recorded in dataset", [1, 2], format_func=lambda x: {1: "Female", 2: "Male"}[x])
        st.markdown("#### Physical measurements")
        c1, c2, c3 = st.columns(3)
        with c1:
            height = st.number_input("Height (cm)", 100, 250, 170, 1)
        with c2:
            weight = st.number_input("Weight (kg)", 30, 300, 75, 1)
        with c3:
            st.metric("Calculated BMI", f"{weight / ((height / 100) ** 2):.1f}")
        st.markdown("#### Blood pressure")
        c1, c2 = st.columns(2)
        with c1:
            ap_hi = st.number_input("Systolic pressure (mmHg)", 50, 300, 120, 1)
        with c2:
            ap_lo = st.number_input("Diastolic pressure (mmHg)", 30, 200, 80, 1)
        st.markdown("#### Clinical indicators")
        c1, c2 = st.columns(2)
        level = {1: "Normal", 2: "Above normal", 3: "Well above normal"}
        with c1:
            cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=level.get)
        with c2:
            glucose = st.selectbox("Glucose", [1, 2, 3], format_func=level.get)
        st.markdown("#### Lifestyle")
        c1, c2, c3 = st.columns(3)
        with c1:
            smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with c2:
            alco = st.selectbox("Alcohol consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with c3:
            active = st.selectbox("Physically active", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        submitted = st.form_submit_button("Assess Cardiovascular Risk", type="primary", use_container_width=True)

    if submitted:
        if ap_lo >= ap_hi:
            status_message("error", "Please check the blood pressure values: systolic pressure should be greater than diastolic pressure.")
        else:
            try:
                import numpy as np
                import torch
                from src.models.cardio_mlp import CardioMLP
                from src.utils.scaler_utils import transform_raw_input

                features = np.asarray([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, glucose, smoke, alco, active]], dtype=np.float32)
                scaled = transform_raw_input(features, use_hospital_scaler=False, num_hospitals=num_clients)
                candidates = [project_path("outputs/models") / name for name in (
                    "at_fedbn_final.pth", "at_fedbn_ui_live_demo.pth", "fedavg_baseline_final.pth")]
                checkpoint_path = next((p for p in candidates if p.is_file()), None)
                if checkpoint_path is None:
                    status_message("error", "No prediction checkpoint is available. The dashboard needs a trained checkpoint in outputs/models/.")
                else:
                    model = CardioMLP(input_dim=model_cfg.get("input_dim", 11),
                                      hidden_dims=tuple(model_cfg.get("hidden_dims", [64, 32])),
                                      dropout=model_cfg.get("dropout", 0.2))
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"])
                    elif isinstance(checkpoint, dict) and any(k.startswith(("fc", "bn")) for k in checkpoint):
                        model.load_state_dict(checkpoint)
                    else:
                        raise ValueError("The selected checkpoint has an unsupported format.")
                    model.eval()
                    with torch.no_grad():
                        probability = float(torch.sigmoid(model(torch.from_numpy(np.asarray(scaled, dtype=np.float32)))).item())
                    threshold = 0.5
                    st.session_state["last_prediction"] = {"probability": probability,
                        "prediction": int(probability >= threshold), "threshold": threshold,
                        "checkpoint": checkpoint_path.stem}
            except Exception as exc:
                status_message("error", f"Prediction could not be completed: {exc}")

    result = st.session_state.get("last_prediction")
    if result:
        st.markdown("---")
        st.markdown("### Assessment result")
        positive = result["prediction"] == 1
        outcome = "Model predicts cardiovascular disease" if positive else "Model does not predict cardiovascular disease"
        probability = result["probability"]
        st.markdown(f"""<div class="result-card"><strong>{outcome}</strong>
<p>Estimated model probability <strong>{probability:.1%}</strong></p>
<div class="prob-track" role="img" aria-label="Estimated model probability {probability:.1%}"><div class="prob-fill" style="width:{probability * 100:.2f}%"></div></div>
<small>Classification threshold: {result['threshold']:.0%}. A model output, not a clinically calibrated risk estimate or diagnosis.</small></div>""", unsafe_allow_html=True)
        st.caption(f"Prediction checkpoint: {result['checkpoint']}")
    st.markdown("---")
    model_performance()
    st.caption("Discuss health concerns and decisions with a qualified healthcare professional. This prototype does not provide medical advice.")


elif page == "Model & Dataset":
    st.markdown("<div class='eyebrow'>How the system is built</div>", unsafe_allow_html=True)
    st.title("Model & Dataset")
    st.markdown("<p class='subtle'>A shared prediction network is trained across simulated hospital partitions; local Batch Normalization values remain client-specific.</p>", unsafe_allow_html=True)

    st.markdown("### Prediction and training pipeline")
    stages = [("Source", "Cardiovascular records"), ("Preparation", "Clean, order and standardize features"),
              ("Local", "Partition records by simulated hospital"), ("Client", "Train local model on-site"),
              ("Exchange", "Shared model parameters / updates"), ("Server", "Aggregate shared parameters"),
              ("Shared model", "Broadcast updated shared weights"), ("Inference", "Scale inputs, sigmoid logit")]
    descriptions = {
        "Source": "The project uses the cardiovascular dataset stored at data/raw/cardio_train.csv. Patient-level records are not displayed in this dashboard.",
        "Preparation": "The preprocessing pipeline removes the target from the feature matrix and fits StandardScaler on training data. The existing prediction utility applies its documented raw-feature normalization.",
        "Local": f"Training partitions are configured for {num_clients} simulated clients. The configured Dirichlet alpha is {data_cfg.get('dirichlet_alpha', 'unavailable')}.",
        "Client": f"Each client runs local training for {training_cfg.get('local_epochs', 'unavailable')} epochs per round on its partition.",
        "Exchange": "For FedBN, non-BatchNorm trainable parameters are shared. Raw records and local BatchNorm parameters/statistics are omitted.",
        "Server": "The server aggregates shared parameters according to the selected strategy. AT-FedBN combines sample-volume and update-alignment weights.",
        "Shared model": "The server returns updated shared parameters; each client merges them while preserving its local BatchNorm state.",
        "Inference": "The app orders 11 inputs as age, gender, height, weight, systolic/diastolic pressure, cholesterol, glucose, smoking, alcohol and activity; it applies transform_raw_input, sigmoid, then the existing 0.5 threshold."
    }
    node_flow("Training and inference · client to server and back", stages, descriptions,
              "model_pipeline_step", ["Source", "Preparation", "Client", "Client", "Transfer", "Server", "Return", "Inference"])

    st.markdown("### Prediction network")
    hidden = model_cfg.get("hidden_dims", [64, 32])
    architecture = [
        ("Input", f"{model_cfg.get('input_dim', 11)} ordered features"),
        ("Dense block 1", f"Linear {hidden[0]} · BatchNorm · ReLU · Dropout {model_cfg.get('dropout', 0.2)}"),
        ("Dense block 2", f"Linear {hidden[1]} · BatchNorm · ReLU · Dropout {model_cfg.get('dropout', 0.2)}"),
        ("Output", "Linear 1 logit · sigmoid · threshold 0.5"),
    ]
    architecture_details = {
        "Input": "The network receives the ordered, standardized 11 patient features used by the prediction pipeline.",
        "Dense block 1": "The first learned projection has the configured width. Batch Normalization, ReLU and Dropout follow the linear layer.",
        "Dense block 2": "A second linear block narrows the representation and uses the same BatchNorm, ReLU and Dropout pattern.",
        "Output": "The model returns one raw logit. Inference applies sigmoid, then the existing 0.5 decision threshold.",
    }
    node_flow("CardioMLP · select a layer for details", architecture, architecture_details,
              "architecture_step", ["Input", "Shared", "Shared", "Prediction"])
    with st.expander("What the layers do"):
        st.write("Linear layers combine input signals. Batch Normalization stabilizes activations; ReLU adds nonlinearity; Dropout reduces co-adaptation during training. The output is a raw logit trained with BCEWithLogitsLoss, converted to a probability with sigmoid during evaluation.")

    st.markdown("### Dataset overview")
    if hospital_rows:
        m1, m2, m3 = st.columns(3)
        m1.metric("Training records", f"{total_train:,}")
        m2.metric("Test records", f"{total_test:,}")
        m3.metric("Model features", str(model_cfg.get("input_dim", 11)))
        train_df, test_df = pd.read_csv(project_path("data/processed/train.csv")), pd.read_csv(project_path("data/processed/test.csv"))
        hospital = pd.DataFrame(hospital_rows).set_index("Hospital")
        chart_view = button_choices(["Target classes", "Hospital samples", "Positive rate"], "dataset_chart_view")
        if chart_view == "Target classes" and "cardio" in train_df and "cardio" in test_df:
            negative = [(train_df.cardio == 0).sum(), (test_df.cardio == 0).sum()]
            positive = [(train_df.cardio == 1).sum(), (test_df.cardio == 1).sum()]
            render_grouped_chart("Records by target class · cardio", ["Negative (0)", "Positive (1)"],
                                 [("Training", [negative[0], positive[0]]), ("Test", [negative[1], positive[1]])])
        elif chart_view == "Hospital samples":
            render_grouped_chart("Samples by hospital", hospital.index, [
                ("Training", hospital["Training samples"].tolist()), ("Test", hospital["Test samples"].tolist())])
        elif chart_view == "Positive rate":
            render_grouped_chart("Positive target rate", hospital.index, [
                ("Training", hospital["Training positive rate"].tolist()), ("Test", hospital["Test positive rate"].tolist())],
                maximum=1, value_format="percent")
        elif chart_view == "Target classes":
            status_message("info", "Target class counts are unavailable in the combined train/test data.")
        render_table(hospital.reset_index(), percent_columns=("Training positive rate", "Test positive rate"))
        st.caption(f"The processed dataset contains {total_train + total_test:,} split records across {len(hospital_rows)} available hospital partitions. Client distributions differ; the configured Dirichlet partition parameter α={data_cfg.get('dirichlet_alpha', 'unavailable')} controls partition heterogeneity.")
    else:
        status_message("info", "Processed hospital train/test CSVs are unavailable, so dataset-derived sample and class charts cannot be shown.")
    st.caption("Dataset provenance: project data files identify the source as the cardiovascular disease dataset; no patient records are exposed here.")


else:
    st.markdown("<div class='eyebrow'>Federated learning approach</div>", unsafe_allow_html=True)
    st.title("AT-FedBN Methodology")
    st.markdown("<p class='subtle'>Hospitals train collaboratively while retaining records and Batch Normalization state locally. AT-FedBN adjusts shared-parameter aggregation using update alignment and data volume.</p>", unsafe_allow_html=True)
    at = read_json("at_fedbn.json")
    beta = float((at or {}).get("beta", fed_cfg.get("beta", 0.7)))

    st.markdown("### One training round")
    stages = [("Hospital", "Records remain local"), ("Local training", "Update model on local data"),
              ("Split", "Keep BN local; select shared weights"), ("Optional noise", "Clip/noise shared update"),
              ("Server", "Compare update directions"), ("Trust", "ReLU cosine alignment"),
              ("Aggregate", "Blend trust + data volume"), ("Broadcast", "Return shared weights")]
    descriptions = {
        "Hospital": "Each simulated client keeps its own partition of patient records. Raw records are not transmitted in the federated training flow.",
        "Local training": "Clients train the CardioMLP locally for the configured number of epochs.",
        "Split": "The client code shares non-BatchNorm trainable parameters. BatchNorm affine parameters and running statistics remain local.",
        "Optional noise": f"The configured client routine clips the aggregated shared weight update and adds Gaussian noise (enabled={privacy_cfg.get('enabled', False)}, clip norm={privacy_cfg.get('clip_norm', 'unavailable')}, noise scale={privacy_cfg.get('noise_scale', 'unavailable')}). This is update-level noise, not per-example DP-SGD.",
        "Server": "The server compares each shared client delta with the volume-weighted reference update for that round.",
        "Trust": "The score is max(trust_floor, cosine similarity). It measures directional alignment with the reference update, not data quality or trustworthiness.",
        "Aggregate": f"Final weight = β × volume share + (1−β) × normalized trust share, with β={beta:.2f}. If all trust scores are approximately zero, the implementation falls back to volume weights; final weights are normalized.",
        "Broadcast": "Only aggregated shared weights return to the clients; each client preserves its own BatchNorm state before the next round."
    }
    node_flow("One round · client operations then server aggregation", stages, descriptions,
              "method_pipeline_step", ["Client", "Client", "Client", "Client", "Server", "Server", "Server", "Return"])

    st.markdown("### Trust and aggregation")
    st.code("trustᵢ = max(trust_floor, cosine_similarity(ΔWᵢ, ΔW_reference))\nweightᵢ = β · (nᵢ / N) + (1 − β) · normalized_trustᵢ", language="text")
    st.caption("ΔWᵢ is a client's shared-parameter update, ΔW_reference is the volume-weighted update direction, and nᵢ/N is its share of participating training samples. β controls the blend. Trust scores describe update alignment only; they do not establish client reliability.")

    weights = (at or {}).get("weights", {}) if at else {}
    trust_history = (at or {}).get("trust_scores", {}) if at else {}
    if trust_history:
        trust_rounds = sorted(trust_history, key=lambda value: int(value))
        client_ids = sorted({client_id for values in trust_history.values() for client_id in values})
        if len(trust_rounds) > 1 and client_ids:
            st.markdown("### Trust history")
            client_labels = [f"Client {index + 1}" for index in range(len(client_ids))]
            selected_client = button_choices(client_labels, "trust_history_client", columns_per_row=3)
            client_id = client_ids[client_labels.index(selected_client)]
            history_values = [trust_history[round_id].get(client_id) for round_id in trust_rounds]
            render_grouped_chart(f"Update-alignment trust · {selected_client}",
                                 [f"Round {round_id}" for round_id in trust_rounds],
                                 [(selected_client, history_values)], maximum=1, value_format="percent")
            st.caption("Saved per-round trust scores are shown for the selected simulated client. The score measures update alignment, not data quality.")

    if weights:
        rounds = sorted(weights, key=lambda x: int(x))
        if len(rounds) > 1:
            selected = st.selectbox("Inspect aggregation round", rounds, index=len(rounds) - 1, format_func=lambda x: f"Round {x}")
        else:
            selected = rounds[-1]
            st.caption(f"Saved aggregation weights · Round {selected}")
        round_data = weights[selected]
        rows = []
        for idx, (client, item) in enumerate(round_data.items(), 1):
            cosine = item.get("cosine", item.get("trust_score"))
            rows.append({"Hospital": f"Client {idx}", "Cosine alignment": cosine,
                         "Trust score (ReLU cosine)": max(float(fed_cfg.get("trust_floor", 0.0)), cosine) if cosine is not None else None,
                         "Volume weight": item.get("volume"), "Trust weight": item.get("trust"),
                         "Final aggregation weight": item.get("final")})
        chart = pd.DataFrame(rows).set_index("Hospital")
        st.markdown("**Saved client scores and weights**")
        render_grouped_chart("Trust and final aggregation weight", chart.index,
                             [("Trust score", chart["Trust score (ReLU cosine)"].tolist()),
                              ("Final weight", chart["Final aggregation weight"].tolist())],
                             maximum=1, value_format="percent")
        render_table(chart.reset_index(), decimal_columns=("Cosine alignment", "Trust score (ReLU cosine)",
                     "Volume weight", "Trust weight", "Final aggregation weight"))
        st.caption(f"Round {selected} values are read from outputs/results/at_fedbn.json. Raw cosine alignment is clipped by the configured trust floor to obtain the trust score; aggregation uses normalized trust weights. Client sample counts and volume weights are zero in the saved round where recorded, and are shown as stored.")
    else:
        status_message("info", "No saved AT-FedBN client aggregation weights are available.")

    st.markdown("### What remains local")
    st.markdown("**Shared:** Linear-layer trainable parameters are exchanged and aggregated.  **Local:** `bn1` and `bn2` parameters and running statistics remain on each client. Local normalization lets the model adapt to differing hospital activation distributions.")

    st.markdown("### Privacy scope")
    status_message("warning", "The code's optional noise mechanism clips and perturbs the full shared weight update after local training. This is not per-example gradient DP-SGD, and the repository's simplified epsilon estimate is not a rigorous privacy accountant; no epsilon guarantee is presented here. Federated learning and update noise do not eliminate all information leakage. This simulated research prototype is not a production healthcare deployment.")

    st.markdown("### Saved experiment comparison")
    experiments = [
        ("Centralized", "central_baseline.json", "best", "Test-set best checkpoint metrics"),
        ("FedBN", "fedbn_only.json", "local.average", "Average client-local evaluation"),
        ("AT-FedBN", "at_fedbn.json", "local.average", "Average client-local evaluation"),
    ]
    comparison = []
    for name, filename, loc, context in experiments:
        result = read_json(filename)
        if result:
            comparison.append({"Model": name, "Evaluation context": context,
                "Accuracy": metric_value(result, loc, "accuracy"),
                "Balanced accuracy": metric_value(result, loc, "balanced_accuracy"),
                "Precision": metric_value(result, loc, "precision"),
                "Recall / sensitivity": metric_value(result, loc, "recall"),
                "F1": metric_value(result, loc, "f1"),
                "ROC-AUC": metric_value(result, loc, "roc_auc")})
    if comparison:
        comp = pd.DataFrame(comparison).set_index("Model")
        metric_columns = ["Accuracy", "Balanced accuracy", "Precision", "Recall / sensitivity", "F1", "ROC-AUC"]
        available_metrics = [column for column in metric_columns if comp[column].notna().any()]
        if available_metrics:
            chosen_metric = button_choices(available_metrics, "comparison_metric")
            render_grouped_chart(f"{chosen_metric} by model", [chosen_metric],
                [(name, [comp.loc[name, chosen_metric]]) for name in comp.index],
                maximum=1, value_format="percent")
        render_table(comp.reset_index(), decimal_columns=metric_columns)
        st.caption("Results are displayed in each experiment's saved evaluation context. Centralized test-set results and averages of client-local evaluations are not identical evaluation procedures; interpret comparisons accordingly.")
    else:
        status_message("info", "No comparable saved model metrics are available.")
