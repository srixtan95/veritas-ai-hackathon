"""Immortals AI research prototype dashboard. Launch with ``streamlit run app.py``."""

import html

import numpy as np
import pandas as pd
import streamlit as st

from src.affordability import DEFAULT_AFFORDABILITY_PERCENTAGE, assess_affordability
from src.application import (
    FEATURE_LABELS,
    MODEL_B_PATH,
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
.hero { padding: 1.35rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#112b3f,#20566b); color:white; margin:.25rem 0 1.3rem; }
.hero p { color:#d7e6ea; margin-bottom:0; }
.eyebrow { color:#8bd1bf; text-transform:uppercase; letter-spacing:.12em; font-size:.76rem; font-weight:700; }
.panel-note { padding:.85rem 1rem; border-left:4px solid #e2a447; background:#fff7e8; border-radius:6px; color:#57461f; }
div[data-testid="stMetric"] { border:1px solid rgba(120,135,150,.22); padding:1rem; border-radius:12px; background:rgba(255,255,255,.62); }
.fl-chart { border:1px solid #e5eaee; border-radius:12px; padding:1rem; margin:.4rem 0 1rem; overflow:hidden; }
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


def _money(value):
    return f"{value:,.2f}"


def _render_table(frame, title=None, percent_columns=()):
    """Render a compact accessible HTML table without Streamlit's Arrow path."""
    if title:
        st.markdown(f"**{title}**")
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
    table = (
        '<div class="fl-table-wrap"><table class="fl-table"><thead><tr>' + header
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)


def _render_bar_chart(frame, title, percent=False):
    """Render a responsive grouped bar chart as inline SVG without chart dependencies."""
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
            value_text = f"{value:.1f}%" if percent else (f"{value:.3f}" if max_value <= 1.5 else f"{value:.2f}")
            svg.append(f'<text x="{min(left+bar_width+7,width-4):.1f}" y="{y+bar_height-2:.1f}" class="value">{html.escape(value_text)}</text>')
    svg.append("</svg>")
    legend = "".join(
        f'<span class="fl-legend-item"><i style="background:{colors[index % len(colors)]}"></i>{html.escape(name[:45])}</span>'
        for index, name in enumerate(series)
    )
    st.markdown(
        f"<div class='fl-chart'><div class='fl-chart-title'>{html.escape(title)}</div>{''.join(svg)}<div class='fl-legend'>{legend}</div></div>",
        unsafe_allow_html=True,
    )


def _render_risk_form(model):
    schema = model_input_schema(model)
    values = {}
    with st.form("applicant_assessment_form", clear_on_submit=False):
        st.markdown("#### Applicant features")
        st.caption("Enter the 19 features expected by age-excluded Model B. Age is not requested or passed to the prediction model.")
        columns = st.columns(2, gap="large")
        for index, field in enumerate(schema):
            target = columns[index % 2]
            with target:
                label = field["label"]
                if field["kind"] == "categorical":
                    choices = field["choices"]
                    chosen = st.selectbox(
                        label,
                        options=choices,
                        format_func=lambda value, mapping=field["help"]: (
                            f"{value} · {mapping[value]}" if value in mapping else value
                        ),
                        key=f"applicant_{field['name']}",
                    )
                    values[field["name"]] = chosen
                else:
                    value = st.number_input(
                        label,
                        min_value=0,
                        max_value=2_000_000_000,
                        value=int(field["default"]),
                        step=1,
                        key=f"applicant_{field['name']}",
                        help="Whole-number model input. Values outside the historical dataset range may be extrapolations.",
                    )
                    values[field["name"]] = value
        submitted = st.form_submit_button("Run credit-risk assessment", type="primary", use_container_width=True)
    if submitted:
        st.session_state.pop("credit_result", None)
        try:
            st.session_state["credit_result"] = assess_credit_risk(model, values)
        except (TypeError, ValueError) as error:
            st.error(f"Check the applicant inputs: {error}")
        else:
            st.success("Assessment calculated with the saved Model B pipeline.")
    return st.session_state.get("credit_result")


def _render_risk_result(result):
    if not result:
        st.info("Complete the applicant form and submit it to see the risk estimate and explanation.")
        return
    st.markdown("#### Credit-risk estimate")
    left, right, note = st.columns([1, 1, 2])
    left.metric("Predicted class", result["predicted_class_name"])
    right.metric("Prediction threshold status", "Near 50%" if result["near_threshold"] else "Outside near-50% band")
    with note:
        st.markdown("**Predicted probabilities**")
        probability_cols = st.columns(2)
        probability_cols[0].metric("Good risk", f"{result['probabilities']['Good risk']:.1%}")
        probability_cols[1].metric("Bad risk", f"{result['probabilities']['Bad risk']:.1%}")
    if result["near_threshold"]:
        st.warning("The model probabilities are close to 50%. This estimate is relatively uncertain; probabilities are not guarantees.")
    else:
        st.caption("Probabilities are model estimates, not guarantees.")
    st.info("A credit-risk prediction is not a loan approval decision.")

    explanation = result["explanation"]
    st.markdown("#### Individual explanation")
    st.write(explanation["explanation"])
    good, bad = st.columns(2, gap="large")
    with good:
        st.markdown("**Factors supporting Good risk**")
        if explanation["strongest_towards_good"]:
            for item in explanation["strongest_towards_good"]:
                st.write(f"+{item['good_log_odds_contribution']:.3f} · {FEATURE_LABELS.get(item['feature'], item['feature'])} — {item['transformed_feature']}")
        else:
            st.caption("No transformed feature contributions in this direction.")
    with bad:
        st.markdown("**Factors supporting Bad risk**")
        if explanation["strongest_towards_bad"]:
            for item in explanation["strongest_towards_bad"]:
                st.write(f"{item['good_log_odds_contribution']:.3f} · {FEATURE_LABELS.get(item['feature'], item['feature'])} — {item['transformed_feature']}")
        else:
            st.caption("No transformed feature contributions in this direction.")
    st.caption("Contribution values are additive changes to Good-versus-Bad log odds from the fitted pipeline; they describe model behavior and do not establish fairness or correctness.")


def _render_credit_section(model):
    st.subheader("Applicant assessment")
    st.write("Use the applicant's financial and credit-history attributes to inspect Model B's estimate. This tool supports analysis; it does not make a lending decision.")
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        result = _render_risk_form(model)
    with right:
        _render_risk_result(result)


def _render_affordability_section():
    st.subheader("Affordability assessment")
    st.write("A standalone budgeting calculation. It is not merged with the credit-risk estimate and is not a measure of creditworthiness.")
    st.markdown('<div class="panel-note"><b>Illustrative assumption:</b> maximum EMI defaults to 30% of nonnegative disposable income. This percentage is configurable and has not been validated as a lending standard. Keep all amounts in the same currency unit.</div>', unsafe_allow_html=True)
    with st.form("affordability_form"):
        rate = st.slider("Illustrative affordability percentage", min_value=0, max_value=100,
                         value=int(DEFAULT_AFFORDABILITY_PERCENTAGE), step=1, format="%d%%")
        columns = st.columns(2, gap="large")
        income = columns[0].number_input("Monthly income", min_value=0.0, value=60000.0, step=1000.0)
        repayments = columns[1].number_input("Existing monthly loan repayments", min_value=0.0, value=8000.0, step=500.0)
        other = columns[0].number_input("Other recurring monthly obligations", min_value=0.0, value=12000.0, step=500.0)
        annual_rate = columns[1].number_input("Proposed annual interest rate (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.25)
        tenure = columns[0].number_input("Proposed loan tenure (months)", min_value=1, max_value=1200, value=36, step=1)
        amount = columns[1].number_input("Proposed loan amount", min_value=0.01, value=300000.0, step=5000.0)
        submitted = st.form_submit_button("Calculate affordability", type="primary", use_container_width=True)
    if submitted:
        try:
            result = assess_affordability(
                monthly_income=income,
                existing_monthly_loan_repayments=repayments,
                other_recurring_monthly_obligations=other,
                annual_interest_rate=annual_rate,
                tenure_months=int(tenure),
                proposed_loan_amount=amount,
                affordability_percentage=float(rate),
            )
        except (TypeError, ValueError) as error:
            st.error(f"Check the affordability inputs: {error}")
        else:
            st.session_state["affordability_result"] = result
    result = st.session_state.get("affordability_result")
    if result:
        st.markdown("#### Budget estimate")
        first = st.columns(3)
        first[0].metric("Disposable income", _money(result["disposable_income_after_existing_obligations"]))
        first[1].metric(f"Illustrative affordable EMI ({result['affordability_percentage']:.0f}%)", _money(result["maximum_illustrative_affordable_emi"]))
        first[2].metric("Indicative maximum principal", _money(result["indicative_maximum_loan_principal"]))
        second = st.columns(4)
        second[0].metric("Proposed monthly EMI", _money(result["proposed_monthly_emi"]))
        second[1].metric("Total repayment", _money(result["proposed_total_repayment"]))
        second[2].metric("Total interest", _money(result["proposed_total_interest"]))
        second[3].metric("Income after obligations + EMI", _money(result["remaining_monthly_income_after_proposed_loan"]))
        if result["proposed_emi_exceeds_illustrative_affordable_emi"]:
            st.warning(result["assessment_message"])
        else:
            st.success(result["assessment_message"])
        st.caption(result["affordability_assumption"])
        st.caption("Affordability is not creditworthiness. This estimate does not guarantee repayment ability or approve/reject the applicant.")


def _render_rate_chart(frame, group_column, model_column, rate_column, metric_name, title):
    chart_frame = frame[[group_column, model_column, rate_column]].copy()
    chart_frame[rate_column] = chart_frame[rate_column] * 100
    wide = chart_frame.pivot(index=group_column, columns=model_column, values=rate_column)
    _render_bar_chart(wide, title, percent=True)
    _render_table(chart_frame.rename(columns={group_column: "Age group", model_column: "Model / experiment", rate_column: f"{metric_name} (%)"}), percent_columns=(f"{metric_name} (%)",))


def _render_fairness_section():
    st.subheader("Fairness and validation research")
    st.write("Saved research outputs only. The dashboard does not rerun the fairness experiments or cross-validation.")
    st.warning("Removing age does not prove fairness. Remaining features may act as proxies, and these research metrics do not determine what is suitable for present-day lending.")
    try:
        research = cached_research_results()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"Saved research results could not be loaded: {error}")
        return

    with st.expander("Single held-out split: age included vs excluded", expanded=True):
        age = research["age_fairness"].copy()
        metric_labels = {
            "approval_rate": "Approval rate",
            "false_approval_rate": "False approval rate",
            "false_rejection_rate": "False rejection rate",
        }
        selected_metric = st.selectbox("Age comparison metric", list(metric_labels), format_func=metric_labels.get, key="age_metric")
        _render_rate_chart(age, "group", "model", selected_metric, metric_labels[selected_metric], metric_labels[selected_metric] + " by age group")
        st.caption("Model A uses age; Model B excludes age. Rates are from one 20% held-out test split; subgroup sizes are small.")

    with st.expander("Proxy-feature ablation"):
        proxy_perf = research["proxy_performance"]
        performance_cols = ["experiment", "accuracy", "balanced_accuracy", "roc_auc", "false_approvals", "false_rejections", "error_cost"]
        st.markdown("**Prediction performance when features are removed**")
        perf_metric = st.selectbox("Performance metric", ["accuracy", "balanced_accuracy", "roc_auc", "error_cost"], key="proxy_perf_metric")
        _render_bar_chart(proxy_perf.set_index("experiment")[[perf_metric]], perf_metric.replace("_", " ").title())
        _render_table(proxy_perf[performance_cols])
        proxy_fairness = research["proxy_fairness"]
        proxy_metric = st.selectbox("Age-group fairness rate", list(metric_labels), format_func=metric_labels.get, key="proxy_metric")
        _render_rate_chart(proxy_fairness, "group", "experiment", proxy_metric, metric_labels[proxy_metric], metric_labels[proxy_metric] + " by age group and ablation")
        st.caption("A feature's predictive association with age or a change in these rates does not establish causation or eliminate proxy effects.")

    with st.expander("Repeated stratified validation (5 folds × 10 repeats)"):
        repeated = research["repeated_fairness"]
        repeated_metric = st.selectbox("Repeated age-group rate", sorted(repeated["metric"].unique()), key="repeat_metric")
        subset = repeated[repeated["metric"] == repeated_metric].copy()
        chart = subset.pivot(index="model", columns="group", values="mean") * 100
        st.markdown(f"**Mean {repeated_metric.replace('_', ' ')} across 10 repeats**")
        _render_bar_chart(chart, f"Mean {repeated_metric.replace('_', ' ')} across 10 repeats", percent=True)
        interval_view = subset[["model", "group", "mean", "empirical_2_5_percentile", "empirical_97_5_percentile", "repeats"]].copy()
        for column in ["mean", "empirical_2_5_percentile", "empirical_97_5_percentile"]:
            interval_view[column] = interval_view[column] * 100
        _render_table(interval_view, percent_columns=("mean", "empirical_2_5_percentile", "empirical_97_5_percentile"))
        gaps = research["repeated_fairness_gaps"]
        st.markdown("**Absolute age-group gaps, percentage points**")
        gap_metric = st.selectbox("Gap measure", sorted(gaps["metric"].unique()), key="repeat_gap_metric")
        gap_subset = gaps[gaps["metric"] == gap_metric].set_index("model")[["mean"]]
        _render_bar_chart(gap_subset, f"Mean {gap_metric.replace('_', ' ')} across repeats")
        _render_table(gaps[gaps["metric"] == gap_metric])
        performance = research["repeated_performance"]
        st.markdown("**Repeated predictive performance**")
        repeat_perf_metric = st.selectbox("Repeated performance measure", sorted(performance["metric"].unique()), key="repeat_perf_metric")
        perf_subset = performance[performance["metric"] == repeat_perf_metric].set_index("model")[["mean"]]
        _render_bar_chart(perf_subset, f"Mean {repeat_perf_metric.replace('_', ' ')} across repeats")
        _render_table(performance[performance["metric"] == repeat_perf_metric])
        st.caption("Ranges shown are empirical 2.5th–97.5th percentiles across repeated evaluations, not formal population confidence intervals. Repeats reuse this dataset and overlapping training sets.")


def main():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">THE IMMORTALS · FAIRNESS-AWARE CREDIT INTELLIGENCE</div>
      <h1>Immortals AI</h1>
      <p><strong>Fairness-Aware Credit Intelligence</strong><br>Intelligence. Transparency. Fairness.</p>
    </div>
    """, unsafe_allow_html=True)
    st.warning("Research prototype. Historical German Credit data does not establish present-day lending suitability. Removing age does not guarantee fairness.")
    try:
        model = cached_model()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"The saved age-excluded Model B could not be loaded from {MODEL_B_PATH}: {error}")
        st.stop()
    st.sidebar.markdown("## Immortals AI")
    st.sidebar.caption("The Immortals · Model B excludes age")
    st.sidebar.info("Credit-risk prediction and affordability are separate analyses. Neither is a loan approval decision.")
    risk_tab, affordability_tab, fairness_tab = st.tabs([
        "Applicant assessment", "Affordability", "Fairness & validation",
    ])
    with risk_tab:
        _render_credit_section(model)
    with affordability_tab:
        _render_affordability_section()
    with fairness_tab:
        _render_fairness_section()
    st.divider()
    st.caption("This project is a prototype and is not financial advice.")


if __name__ == "__main__":
    main()
