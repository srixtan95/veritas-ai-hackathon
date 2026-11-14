"""Immortals AI research prototype dashboard. Launch with ``streamlit run app.py``."""

import html
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.affordability import DEFAULT_AFFORDABILITY_PERCENTAGE, assess_affordability
from src.application import (
    FEATURE_HELP,
    FEATURE_LABELS,
    MODEL_B_PATH,
    PROJECT_ROOT,
    assess_credit_risk,
    load_credit_model,
    load_research_results,
    model_input_schema,
)


st.set_page_config(
    page_title="Immortals AI | Fairness-Aware Credit Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }
h1, h2, h3 { font-family: 'Segoe UI', Arial, sans-serif; letter-spacing: -0.025em; }
.hero { padding: 1.35rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#102b3b,#20566b); color:white; margin:.25rem 0 1.3rem; }
.hero p { color:#d7e6ea; margin-bottom:0; }
.eyebrow { color:#8bd1bf; text-transform:uppercase; letter-spacing:.12em; font-size:.76rem; font-weight:700; }
.panel-note { padding:.85rem 1rem; border-left:4px solid #d09a45; background:#fff7e8; border-radius:6px; color:#57461f; }
.fl-card { border:1px solid #e3e9ed; border-radius:14px; padding:1rem 1.1rem; background:rgba(255,255,255,.65); min-height:125px; margin:.3rem 0; }
.fl-card h3 { margin:.15rem 0 .55rem; font-size:1.05rem; color:#1a3a4b; }
.fl-card p { margin:0; color:#4d606a; font-size:.94rem; }
.fl-good { border-left:4px solid #2d7667; }
.fl-bad { border-left:4px solid #b66060; }
div[data-testid="stMetric"] { border:1px solid rgba(120,135,150,.22); padding:.9rem; border-radius:12px; background:rgba(255,255,255,.62); }
.fl-chart { border:1px solid #e5eaee; border-radius:12px; padding:1rem; margin:.4rem 0 .6rem; overflow:hidden; }
.fl-chart-title { font-weight:700; margin:0 0 .65rem; color:#1c3444; }
.fl-chart svg { display:block; }
.fl-chart .axis { font-size:12px; fill:#75838d; }
.fl-chart .category { font-size:13px; fill:#344b59; }
.fl-chart .value { font-size:12px; fill:#344b59; }
.fl-legend { display:flex; flex-wrap:wrap; gap:.5rem 1rem; padding:.3rem 0 0 16rem; font-size:.78rem; color:#536571; }
.fl-legend-item { display:flex; align-items:center; gap:.35rem; }
.fl-legend-item i { display:inline-block; width:.7rem; height:.7rem; border-radius:50%; }
.fl-table-wrap { overflow-x:auto; margin:.4rem 0 1rem; }
.fl-table { border-collapse:collapse; width:100%; font-size:.88rem; }
.fl-table th { text-align:left; padding:.65rem .75rem; color:#294254; background:#f0f4f6; border-bottom:1px solid #dce4e9; }
.fl-table td { padding:.6rem .75rem; border-bottom:1px solid #e9eef1; color:#354b58; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading saved age-excluded Model B…")
def cached_model():
    return load_credit_model(MODEL_B_PATH)


@st.cache_data(show_spinner=False)
def cached_research_results():
    return load_research_results()


@st.cache_data(show_spinner=False)
def cached_global_importance():
    path = PROJECT_ROOT / "outputs" / "explainability" / "global_feature_importance.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Saved global explanation report not found: {path}")
    return pd.read_csv(path)


def _money(value):
    return f"{value:,.2f}"


def _render_table(frame, title=None, percent_columns=()):
    """Render a compact accessible table without Streamlit's Arrow dependency."""
    if title:
        st.markdown(f"**{html.escape(str(title))}**", unsafe_allow_html=True)
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column, value in row.items():
            if pd.isna(value):
                shown = "N/A"
            elif column in percent_columns:
                shown = f"{float(value):.1f}%"
            elif isinstance(value, (float, np.floating)):
                shown = f"{float(value):.4f}"
            else:
                shown = str(value)
            cells.append(f"<td>{html.escape(shown)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown('<div class="fl-table-wrap"><table class="fl-table"><thead><tr>' + header
                + "</tr></thead><tbody>" + "".join(rows)
                + "</tbody></table></div>", unsafe_allow_html=True)


def _render_bar_chart(frame, title, percent=False, axis_label=None):
    """Responsive SVG bars; labels and table remain usable on small screens."""
    data = frame.copy().apply(lambda column: pd.to_numeric(column, errors="coerce"))
    categories = [str(value) for value in data.index]
    series = [str(value) for value in data.columns]
    values = data.to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = 0
    max_value = max(float(values.max()) if values.size else 0, 1e-9)
    width, left, right, top = 900, 245, 105, 45
    group_height = max(54, 17 * len(series) + 12)
    height = top + group_height * len(categories) + 24
    plot_width = width - left - right
    colors = ["#2d7667", "#557b9e", "#c78c3d", "#8a6da9", "#b66060", "#6b8790", "#9a9c55", "#7581ba"]
    svg = [f'<svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    for tick in range(5):
        x = left + plot_width * tick / 4
        label = max_value * tick / 4
        label_text = f"{label:.0f}%" if percent else (f"{label:.2f}" if max_value <= 1.5 else f"{label:.0f}")
        svg.append(f'<line x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{height-16}" stroke="#e3e9ed"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-2}" text-anchor="middle" class="axis">{label_text}</text>')
    for row_index, category in enumerate(categories):
        y0 = top + row_index * group_height
        svg.append(f'<text x="{left-12}" y="{y0 + group_height/2 + 4:.1f}" text-anchor="end" class="category">{html.escape(category[:36])}</text>')
        for series_index, series_name in enumerate(series):
            value = float(values[row_index, series_index])
            bar_height = min(15, (group_height - 8) / max(len(series), 1))
            y = y0 + 3 + series_index * (bar_height + 2)
            bar_width = max(0, plot_width * value / max_value)
            color = colors[series_index % len(colors)]
            svg.append(f'<rect x="{left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="4" fill="{color}"/>')
            shown = f"{value:.1f}%" if percent else (f"{value:.3f}" if max_value <= 1.5 else f"{value:.2f}")
            svg.append(f'<text x="{min(left+bar_width+7,width-4):.1f}" y="{y+bar_height-2:.1f}" class="value">{html.escape(shown)}</text>')
    svg.append("</svg>")
    legend = "".join(f'<span class="fl-legend-item"><i style="background:{colors[i % len(colors)]}"></i>{html.escape(name[:45])}</span>' for i, name in enumerate(series))
    axis_caption = axis_label or ("Rate (%)" if percent else "Metric value")
    st.markdown(f"<div class='fl-chart'><div class='fl-chart-title'>{html.escape(title)}</div>{''.join(svg)}<div class='fl-legend'>{legend}</div><div class='axis-caption'>Horizontal axis: {html.escape(axis_caption)} · Vertical axis: {html.escape(' / '.join(categories))}</div></div>", unsafe_allow_html=True)


def _render_hero():
    st.markdown("""
    <div class="hero"><div class="eyebrow">THE IMMORTALS · FAIRNESS-AWARE CREDIT INTELLIGENCE</div>
      <h1>Immortals AI</h1><p><strong>Fairness-Aware Credit Intelligence</strong><br>
      Intelligence. Transparency. Fairness.</p></div>
    """, unsafe_allow_html=True)


def _render_overview():
    st.subheader("Credit decisions deserve careful scrutiny")
    st.write("Credit-risk tools learn patterns from past lending records. Those patterns can help organise evidence, but old labels can also carry historical limitations. Immortals AI makes the model's estimates and research visible so people can inspect how the system behaves.")
    st.markdown("**Why fairness matters:** access to credit affects people's options. A model should be examined across groups and error types, while recognising that a small historical dataset cannot settle whether a system is fair.")
    st.markdown("#### Four connected parts")
    columns = st.columns(4, gap="medium")
    cards = [
        ("Credit-risk model", "A saved Logistic Regression estimates Good-risk and Bad-risk probabilities from historical German Credit data."),
        ("Affordability", "A separate budgeting calculator estimates EMI and principal using income, obligations, rate and tenure."),
        ("Fairness research", "Saved experiments compare group rates, errors, feature ablations and repeated validation."),
        ("Explainability", "Global influence and applicant-level log-odds contributions show how the fitted model formed a score."),
    ]
    for column, (title, description) in zip(columns, cards):
        with column:
            st.markdown(f'<div class="fl-card"><h3>{html.escape(title)}</h3><p>{html.escape(description)}</p></div>', unsafe_allow_html=True)
    st.markdown("#### How to use this application")
    st.markdown("1. Open **Applicant Assessment** and enter the required financial and historical credit attributes.  2. Read the separate affordability estimate and Model B risk probabilities.  3. Inspect which transformed inputs moved the model score in either direction.  4. Use **Fairness & Research** to review the saved group and experiment results.")
    st.info("Research prototype based on historical German Credit data. It is not a real lending decision system. **This project is a prototype and is not financial advice.**")


def _render_financial_inputs():
    st.markdown("#### 1 · Financial information")
    income = st.number_input("Monthly income", min_value=0.0, value=None, step=1000.0, placeholder="Enter monthly income", key="finance_income")
    st.markdown("#### 2 · Existing financial obligations")
    a, b = st.columns(2)
    with a:
        repayments = st.number_input("Existing monthly loan repayments", min_value=0.0, value=None, step=500.0, placeholder="Enter monthly repayments", key="finance_repayments")
    with b:
        other = st.number_input("Other recurring monthly obligations", min_value=0.0, value=None, step=500.0, placeholder="Enter monthly obligations", key="finance_other")
    st.markdown("#### 3 · Proposed loan information")
    a, b, c = st.columns(3)
    with a:
        rate = st.number_input("Proposed annual interest rate (%)", min_value=0.0, max_value=100.0, value=None, step=0.25, placeholder="Enter 0–100%", key="finance_rate")
    with b:
        tenure = st.number_input("Proposed loan tenure (months)", min_value=1, max_value=1200, value=None, step=1, placeholder="Enter months", key="finance_tenure")
    with c:
        amount = st.number_input("Proposed loan amount (optional)", min_value=0.01, value=None, step=5000.0, placeholder="Leave blank to estimate a limit", key="finance_amount")
    percentage = st.slider("Illustrative affordability share of disposable income", min_value=0, max_value=100, value=int(DEFAULT_AFFORDABILITY_PERCENTAGE), step=1, format="%d%%", key="finance_percentage")
    st.caption("30% is the prototype default, configurable here. It is an illustrative assumption, not a validated lending standard. Keep all monetary amounts in the same currency unit.")
    return income, repayments, other, rate, tenure, amount, percentage


def _model_inputs(schema):
    values = {}
    complete = True
    st.markdown("#### 4 · Credit-history and applicant characteristics")
    st.caption("These are the exact 19 inputs expected by saved age-excluded Model B. Age (Attribute13) is not collected or sent to the model.")
    for field in schema:
        with st.container(border=True):
            if field["kind"] == "categorical":
                choices = field["choices"]
                chosen = st.selectbox(
                    field["label"], options=choices, index=None,
                    format_func=lambda value, mapping=field["help"]: f"{value} · {mapping[value]}" if value in mapping else value,
                    placeholder="Choose a dataset category", key=f"applicant_{field['name']}",
                    help="Only categories present in the fitted model's training data are offered.",
                )
                values[field["name"]] = chosen
                complete &= chosen is not None
            else:
                chosen = st.number_input(
                    field["label"], min_value=0, max_value=2_000_000_000,
                    value=None, step=1, placeholder="Enter a nonnegative whole number",
                    key=f"applicant_{field['name']}",
                    help="Whole-number model input. Values beyond historical training ranges may be extrapolations.",
                )
                values[field["name"]] = chosen
                complete &= chosen is not None
    return values, complete


def _render_affordability_explanation(financial):
    income, repayments, other, rate, tenure, amount, percentage = financial
    st.markdown("#### Affordability explanation")
    missing = [name for name, value in (("monthly income", income), ("existing repayments", repayments), ("other obligations", other), ("interest rate", rate), ("tenure", tenure)) if value is None]
    if missing:
        st.info("Enter " + ", ".join(missing) + " to calculate the affordability estimate. The calculation remains separate from the credit-risk model.")
        return None
    try:
        result = assess_affordability(
            monthly_income=income,
            existing_monthly_loan_repayments=repayments,
            other_recurring_monthly_obligations=other,
            annual_interest_rate=rate,
            tenure_months=int(tenure),
            proposed_loan_amount=amount,
            affordability_percentage=float(percentage),
        )
    except (TypeError, ValueError) as error:
        st.error(f"Check the affordability inputs: {error}")
        return None
    st.session_state["live_affordability_result"] = result
    one, two = st.columns(2)
    one.metric("Total monthly obligations", _money(result["total_monthly_obligations"]))
    two.metric("Disposable after existing obligations", _money(result["disposable_income_after_existing_obligations"]))
    one, two = st.columns(2)
    one.metric(f"Maximum illustrative EMI ({percentage}%)", _money(result["maximum_illustrative_affordable_emi"]))
    two.metric("Indicative maximum principal", _money(result["indicative_maximum_loan_principal"]))
    st.caption(f"Income {_money(income)} − existing repayments {_money(repayments)} − other obligations {_money(other)} = disposable income {_money(result['disposable_income_after_existing_obligations'])} per month.")
    if amount is None:
        st.info("Add a proposed loan amount to compare its EMI with the illustrative affordability limit.")
    else:
        one, two = st.columns(2)
        one.metric("Proposed monthly EMI", _money(result["proposed_monthly_emi"]))
        two.metric("Remaining income after obligations + EMI", _money(result["remaining_monthly_income_after_proposed_loan"]))
        one, two = st.columns(2)
        one.metric("Total repayment", _money(result["proposed_total_repayment"]))
        two.metric("Total interest payable", _money(result["proposed_total_interest"]))
        if result["proposed_emi_exceeds_illustrative_affordable_emi"]:
            st.warning("The proposed EMI exceeds the illustrative affordability limit. This is a budgeting signal, not an automatic credit rejection.")
        else:
            st.success("The proposed EMI is within the illustrative limit. This does not guarantee repayment ability or imply credit approval.")
    st.caption("A higher interest rate generally raises EMI for a fixed principal. A longer tenure generally lowers the scheduled EMI but can increase total interest. These calculations do not guarantee repayment ability.")
    return result


def _contribution_rows(result, schema):
    """Map the existing transformed contributions back to all 19 source inputs."""
    explanation = result["explanation"]
    by_source = {name: [] for name in result["features"]}
    for item in explanation["contributions"]:
        by_source[item["feature"]].append(item)
    schema_by_name = {field["name"]: field for field in schema}
    rows = []
    for name in result["features"]:
        field = schema_by_name[name]
        raw_value = result["features"][name]
        display_value = str(raw_value)
        if field["kind"] == "categorical" and raw_value in FEATURE_HELP.get(name, {}):
            display_value += f" · {FEATURE_HELP[name][raw_value]}"
        active = by_source[name]
        if not active:
            rows.append({"feature": name, "label": FEATURE_LABELS.get(name, name), "value": display_value,
                         "transformed": "No active contribution", "contribution": 0.0})
        else:
            for item in active:
                rows.append({"feature": name, "label": FEATURE_LABELS.get(name, name), "value": display_value,
                             "transformed": item["transformed_feature"],
                             "contribution": item["good_log_odds_contribution"]})
    return rows


def _render_contribution_group(rows, good_direction):
    selected = [row for row in rows if (row["contribution"] > 0 if good_direction else row["contribution"] < 0)]
    selected.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    if not selected:
        st.caption("No active contributions in this direction for this applicant.")
        return
    for row in selected:
        direction = "towards Good risk" if good_direction else "towards Bad risk"
        explanation = (f"Adds {abs(row['contribution']):.3f} to Good-versus-Bad log odds" if good_direction
                      else f"Subtracts {abs(row['contribution']):.3f} from Good-versus-Bad log odds")
        st.markdown(f"**{html.escape(row['label'])}** · {html.escape(row['value'])}  \n`{html.escape(row['transformed'])}` · {direction} · **{row['contribution']:+.3f}**  \n{explanation}.")


def _render_credit_explanation(model, schema, values, complete):
    st.markdown("#### Credit-risk model explanation")
    if not complete:
        st.info("Choose a category or enter a value for every Model B feature to see the live prediction. Required inputs are identified on the left.")
        st.session_state.pop("live_credit_result", None)
        return
    try:
        result = assess_credit_risk(model, values)
    except (TypeError, ValueError) as error:
        st.error(f"Check the Model B inputs: {error}")
        return
    st.session_state["live_credit_result"] = result
    st.markdown(f"**Predicted class · {result['predicted_class_name']}**")
    good, bad = st.columns(2)
    good.metric("Good-risk probability", f"{result['probabilities']['Good risk']:.1%}")
    bad.metric("Bad-risk probability", f"{result['probabilities']['Bad risk']:.1%}")
    if result["near_threshold"]:
        st.warning("The probabilities are within 10 percentage points of 50%. This model estimate is relatively uncertain; probability is not a guarantee.")
    else:
        st.caption("Probabilities are model estimates, not certainty or guarantees.")
    st.write(result["explanation"]["explanation"])
    rows = _contribution_rows(result, schema)
    good_col, bad_col = st.columns(2, gap="large")
    with good_col:
        st.markdown("**Supporting Good-risk score**")
        with st.expander("Show contributions", expanded=True):
            _render_contribution_group(rows, True)
    with bad_col:
        st.markdown("**Supporting Bad-risk score**")
        with st.expander("Show contributions", expanded=True):
            _render_contribution_group(rows, False)
    with st.expander("All 19 model features", expanded=False):
        neutral = [row for row in rows if row["contribution"] == 0]
        if neutral:
            st.markdown("Features with zero contribution for this applicant")
            for row in neutral:
                st.write(f"{row['label']} · {row['value']} · 0.000 (no active transformed contribution)")
        st.caption("One-hot encoded categories are shown as transformed terms. Each number is the fitted transformed value multiplied by its Logistic Regression coefficient, in Good-versus-Bad log-odds units. These mathematical contributions do not prove causation, fairness, or correctness, and no feature independently determines a result.")
    explanation = result["explanation"]
    good_count = sum(row["contribution"] > 0 for row in rows)
    bad_count = sum(row["contribution"] < 0 for row in rows)
    st.markdown("**Combined explanation**")
    st.write(f"Across the active transformed terms, {good_count} contribution(s) move the score towards Good risk and {bad_count} towards Bad risk. Together with the model intercept, they produce Good-risk log odds of {explanation['good_log_odds']:.3f} and a Good-risk probability of {result['probabilities']['Good risk']:.1%}. The classifier therefore predicts **{result['predicted_class_name']}**. This describes how the model combines its inputs, not why an applicant is inherently creditworthy.")
    with st.expander("Responsible AI notice"):
        st.write("Immortals AI does not approve or reject loans. This is a research-oriented estimate, not a repayment guarantee. The saved model was trained using historical German Credit data. Age is excluded from live Model B prediction and may be used separately for aggregate fairness analysis. Removing age does not guarantee that correlated proxy information is absent.")


def _render_applicant_assessment(model):
    st.subheader("Applicant assessment")
    st.write("Explore a budgeting estimate and a separate historical credit-risk estimate. Neither is an approval or rejection.")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        financial = _render_financial_inputs()
        st.caption("Financial inputs are used only by the affordability calculator and are kept separate from credit-risk prediction.")
        schema = model_input_schema(model)
        values, complete = _model_inputs(schema)
    with right:
        with st.container(border=True):
            _render_affordability_explanation(financial)
        with st.container(border=True):
            _render_credit_explanation(model, schema, values, complete)


def _rate_chart(frame, group_column, series_column, rate_column, title):
    wide = frame.pivot(index=group_column, columns=series_column, values=rate_column) * 100
    _render_bar_chart(wide, title, percent=True, axis_label="Rate (%)")


def _render_fairness_research():
    st.subheader("Fairness & Research")
    st.write("This page presents the project's saved evaluation results. It does not retrain models or rerun experiments.")
    st.warning("Removing a feature does not prove fairness. These results describe one historical dataset and several model evaluations; no single metric is sufficient evidence of fairness.")
    try:
        research = cached_research_results()
        stage1 = json.loads((PROJECT_ROOT / "models" / "stage1_metrics.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"Saved research results could not be loaded: {error}")
        return

    st.markdown("### 1 · Baseline comparison")
    baseline = stage1["metrics"]
    compare = pd.DataFrame({
        "Always predict Good": {"Accuracy": baseline["dummy_all_good"]["accuracy"], "Balanced accuracy": baseline["dummy_all_good"]["balanced_accuracy"], "False approvals": baseline["dummy_all_good"]["false_approvals"], "False rejections": baseline["dummy_all_good"]["false_rejections"]},
        "Logistic Regression": {"Accuracy": baseline["logistic_regression"]["accuracy"], "Balanced accuracy": baseline["logistic_regression"]["balanced_accuracy"], "False approvals": baseline["logistic_regression"]["false_approvals"], "False rejections": baseline["logistic_regression"]["false_rejections"]},
    })
    _render_bar_chart(compare.loc[["Accuracy", "Balanced accuracy"]] * 100, "Held-out predictive performance", percent=True, axis_label="Score (%)")
    _render_table(compare.reset_index(names="Held-out metric"))
    st.caption("On the saved 200-row test split, always predicting Good reaches 70% accuracy but marks all 60 Bad-risk cases as Good. Logistic Regression reaches 78% accuracy with 28 such errors and 16 false rejections. This illustrates why accuracy alone can mislead.")

    st.markdown("### 2 · Age-included versus age-excluded model")
    age = research["age_fairness"]
    performance = pd.read_csv(PROJECT_ROOT / "outputs" / "age_ablation" / "model_comparison.csv")
    _render_bar_chart(performance.set_index("model")[["accuracy", "balanced_accuracy", "roc_auc"]] * 100, "Model A and Model B held-out performance", percent=True, axis_label="Score (%)")
    st.caption("Model A includes age; Model B excludes Attribute13. Accuracy changes from 78.0% to 77.5% on this split. Age is still retained separately for aggregate fairness auditing.")

    st.markdown("### 3 · Age-group approval-rate differences")
    _rate_chart(age, "group", "model", "approval_rate", "Predicted Good-risk rate by age group")
    st.caption("This is a descriptive predicted-Good rate comparison for Under 25 and 25 or older on one held-out split.")

    st.markdown("### 4 · False approval and false rejection gaps")
    gaps_path = PROJECT_ROOT / "outputs" / "age_ablation" / "fairness_gaps.csv"
    gaps = pd.read_csv(gaps_path)
    gap_wide = gaps.pivot(index="model", columns="metric", values="absolute_gap_percentage_points")
    _render_bar_chart(gap_wide, "Absolute held-out age-group error and approval gaps", axis_label="Percentage-point gap")
    st.caption("False approval means an actual Bad-risk case predicted Good; false rejection means actual Good predicted Bad. The rates use different class-specific denominators.")

    st.markdown("### 5 · Proxy-feature ablation experiments")
    proxy_perf = research["proxy_performance"]
    metric = st.selectbox("Proxy ablation performance measure", ["accuracy", "balanced_accuracy", "roc_auc", "error_cost"], key="proxy_metric")
    proxy_values = proxy_perf.set_index("experiment")[[metric]].copy()
    if metric != "error_cost":
        proxy_values *= 100
    _render_bar_chart(proxy_values, f"Saved proxy ablation comparison: {metric.replace('_', ' ')}", percent=metric != "error_cost", axis_label="Error cost" if metric == "error_cost" else "Score (%)")
    _render_table(research["proxy_performance_changes"], "Performance change after removing candidate features")
    st.caption("Removing checking account status changed accuracy by −5.5 percentage points in its experiment; removing savings, employment, and housing together changed it by +3.0 points. The outcomes vary by feature and metric; neither result proves causation or eliminates proxy effects.")

    st.markdown("### 6 · Repeated validation across eight configurations")
    repeated_perf = research["repeated_performance"]
    perf_metric = st.selectbox("Repeated validation metric", sorted(repeated_perf["metric"].unique()), index=list(sorted(repeated_perf["metric"].unique())).index("accuracy"), key="repeated_metric")
    perf_subset = repeated_perf[repeated_perf["metric"] == perf_metric]
    percent_metric = perf_metric in {"accuracy", "balanced_accuracy", "roc_auc"}
    perf_values = perf_subset.set_index("model")[["mean"]] * (100 if percent_metric else 1)
    _render_bar_chart(perf_values, f"Mean {perf_metric.replace('_', ' ')} across 10 repeats", percent=percent_metric, axis_label=perf_metric)
    _render_table(perf_subset[["model", "mean", "empirical_2_5_percentile", "empirical_97_5_percentile", "repeats"]])
    fairness = research["repeated_fairness"]
    age_metric = st.selectbox("Repeated age-group measure", sorted(fairness["metric"].unique()), key="repeat_fair_metric")
    fair_subset = fairness[fairness["metric"] == age_metric].copy()
    fair_subset["mean"] *= 100
    _render_bar_chart(fair_subset.pivot(index="model", columns="group", values="mean"), f"Repeated mean {age_metric.replace('_', ' ')} by age group", percent=True, axis_label="Rate (%)")
    gaps_repeat = research["repeated_fairness_gaps"]
    _render_table(gaps_repeat[gaps_repeat["metric"].str.contains("approval_rate_gap|false_approval_rate_gap|false_rejection_rate_gap", regex=True)], "Repeated age-group gaps (mean and empirical range)")
    st.caption("Validation uses five stratified folds repeated ten times. Ranges are empirical 2.5th–97.5th percentiles, not formal population confidence intervals; repeats reuse the same dataset and overlapping training sets.")

    st.markdown("### 7 · Performance and fairness trade-offs")
    changes = research["proxy_performance_changes"].copy()
    st.write("Across these saved experiments, changes to predictive performance and group error gaps do not move together in one consistent direction. Review the individual rows and denominators; a better value on one metric is not a complete fairness judgment.")
    _render_table(changes, "Predictive metric changes for feature-removal experiments")
    _render_table(research["proxy_fairness_gaps"], "Age-group rate gaps for feature-removal experiments")
    st.caption("Small subgroup counts make rates unstable, and the historical German Credit labels are limited proxies for real repayment outcomes.")


def _render_feature_technology():
    st.subheader("Features & Technology")
    sections = [
        ("Credit-risk engine", "Reuses the saved scikit-learn Logistic Regression pipeline trained on Statlog German Credit records. It estimates class 1 (Good risk) and class 2 (Bad risk); those probabilities are not loan decisions."),
        ("Affordability engine", "A separate rule-based calculator uses income, obligations, proposed interest rate and tenure to calculate disposable income, EMI and an indicative principal. Its configurable default uses 30% of nonnegative disposable income as an illustrative EMI limit; this assumption is not a validated standard."),
        ("Fairness audit", "Compares predicted Good rates, false approvals and false rejections between Under 25 and 25 or older in saved outputs. Age is available to aggregate auditing while excluded from live Model B inputs."),
        ("Proxy-feature investigation", "Tests whether removing selected features changes performance and age-group outcomes. A remaining input may carry information associated with another attribute; removal alone cannot establish fairness."),
        ("Repeated validation", "Compares eight model configurations across five stratified folds repeated ten times. This reveals how results vary across partitions of the same dataset, not new independent populations."),
        ("Explainability", "Global importance groups one-hot encoded levels by original feature and ranks mean absolute per-applicant contributions to Good-versus-Bad log odds. Individual views multiply the actual transformed inputs by saved Logistic Regression coefficients and include the intercept."),
        ("Baseline comparison", "The dashboard compares Logistic Regression with a simple baseline that predicts Good for everyone. This shows why a single accuracy score can hide mistakes on the Bad-risk class."),
        ("Responsible AI", "The dataset is historical and limited. Probabilities are estimates, explanations describe fitted model arithmetic, and neither a score nor an affordability estimate proves fairness, correctness or repayment ability. The application is a research prototype, not a production lending system."),
    ]
    for title, description in sections:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(description)
    st.markdown("#### Global feature influence in Model B")
    try:
        importance = cached_global_importance()
        shown = importance.head(10).copy()
        shown["label"] = shown["feature"].map(FEATURE_LABELS).fillna(shown["feature"])
        st.markdown("Rank uses mean absolute per-row Good-versus-Bad log-odds contribution; categorical one-hot terms are grouped under their source feature. This accounts for fitted scaling and observed encoded activations rather than comparing raw coefficient sizes.")
        chart = shown.set_index("label")[["mean_absolute_log_odds_contribution"]]
        _render_bar_chart(chart, "Top Model B global feature influence", axis_label="Mean absolute log-odds contribution")
        _render_table(shown[["label", "mean_absolute_log_odds_contribution", "mean_log_odds_contribution", "direction"]].rename(columns={"label": "Feature", "mean_absolute_log_odds_contribution": "Mean absolute contribution", "mean_log_odds_contribution": "Mean signed contribution"}))
        st.caption("Largest global influences in the saved report are checking-account status (Attribute1), foreign-worker status (Attribute20), savings (Attribute6), other installment plans (Attribute14), and installment rate (Attribute8). Direction is the average signed effect; actual direction varies by applicant category/value.")
    except (FileNotFoundError, OSError, pd.errors.ParserError) as error:
        st.info(f"The saved global influence report is unavailable: {error}")
    st.markdown("#### Technology stack")
    requirements = PROJECT_ROOT / "requirements.txt"
    packages = [line.strip() for line in requirements.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    st.write("Python runs the analysis; Streamlit provides the interface; pandas and NumPy handle tabular/numeric work; scikit-learn provides preprocessing and Logistic Regression; joblib loads the saved model; SciPy supports numerical routines; Matplotlib and seaborn support research plots; ucimlrepo is used by dataset-loading workflows. Dashboard predictions load the saved pipeline and do not download or retrain the training dataset.")
    st.code("\n".join(packages), language="text")


def main():
    _render_hero()
    st.sidebar.markdown("## Immortals AI")
    st.sidebar.caption("The Immortals · Fairness-Aware Credit Intelligence")
    st.sidebar.info("Historical research estimates and a separate budgeting calculator. Neither makes a loan decision.")
    try:
        model = cached_model()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"The saved age-excluded Model B could not be loaded from {MODEL_B_PATH}: {error}")
        st.stop()
    overview, assessment, fairness, technology = st.tabs([
        "Overview", "Applicant Assessment", "Fairness & Research", "Features & Technology",
    ])
    with overview:
        _render_overview()
    with assessment:
        _render_applicant_assessment(model)
    with fairness:
        _render_fairness_research()
    with technology:
        _render_feature_technology()
    st.divider()
    st.caption("This project is a prototype and is not financial advice.")


if __name__ == "__main__":
    main()
