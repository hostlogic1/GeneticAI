"""
GeneticAI Elite Dashboard
=========================
Streamlit dashboard for running and viewing genetic indicator optimization results.
Dark-themed, inspired by the Wolfpack Elite Dashboard design.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path

from genetic_indicator_engine import (
    run_evolution, backtest_strategy, bootstrap_validation,
    load_data_yfinance, load_data_csv, split_data,
    IndicatorCache, INDICATOR_NAMES, INDICATOR_REGISTRY,
    save_results, load_results, TradingStrategy, compute_fitness_score,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeneticAI Elite Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS - Dark Theme matching Wolfpack Elite style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0e17;
        color: #e0e0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1520;
    }

    /* Cards / metric boxes */
    .metric-card {
        background: linear-gradient(135deg, #0f1520 0%, #141e30 100%);
        border: 1px solid #1a2744;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.2em;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85em;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .positive { color: #00e676; }
    .negative { color: #ff5252; }
    .neutral { color: #00bcd4; }
    .warning { color: #ffc107; }

    /* Title styling */
    .dashboard-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #00bcd4;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 5px;
        text-shadow: 0 0 20px rgba(0,188,212,0.3);
    }
    .dashboard-subtitle {
        text-align: center;
        font-size: 0.9em;
        color: #8892a4;
        margin-bottom: 20px;
    }

    /* Section headers */
    .section-header {
        color: #00e676;
        font-size: 1.1em;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #1a2744;
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }

    /* Parameter box */
    .param-box {
        background: #0f1520;
        border: 1px solid #00e676;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 0.85em;
    }

    /* Validated badge */
    .badge-validated {
        background: #00e676;
        color: #000;
        padding: 8px 20px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-failed {
        background: #ff5252;
        color: #fff;
        padding: 8px 20px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }

    /* Table styling */
    .results-table {
        background: #0f1520;
        border-radius: 8px;
        padding: 15px;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #00bcd4;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #141e30;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: #8892a4;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00bcd4;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def format_pct(val, decimals=2):
    """Format percentage with color."""
    color = "positive" if val > 0 else "negative" if val < 0 else "neutral"
    return f'<span class="{color}">{val:+.{decimals}f}%</span>'


def metric_card(label, value, color_class="neutral"):
    """Render a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """


def create_equity_chart(equity_curve, title="Equity Curve"):
    """Create a styled equity curve chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        line=dict(color='#00e676', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 230, 118, 0.1)',
        name='Equity',
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color='#e0e0e0', size=16)),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#8892a4'),
        xaxis=dict(
            gridcolor='#1a2744',
            title='Timestamp',
        ),
        yaxis=dict(
            gridcolor='#1a2744',
            title='Equity ($)',
        ),
        hovermode='x unified',
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def create_drawdown_chart(portfolio):
    """Create a drawdown chart."""
    dd = portfolio.drawdown() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode='lines',
        line=dict(color='#ff5252', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.2)',
        name='Drawdown %',
    ))
    fig.update_layout(
        title=dict(text='Drawdown', font=dict(color='#e0e0e0', size=16)),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#8892a4'),
        xaxis=dict(gridcolor='#1a2744'),
        yaxis=dict(gridcolor='#1a2744', title='Drawdown %'),
        hovermode='x unified',
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def create_generation_chart(gen_stats):
    """Create generation progress chart."""
    df = pd.DataFrame(gen_stats)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Best Fitness Score", "Best Sharpe Ratio"),
                        vertical_spacing=0.12)
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["best_score"],
        mode='lines+markers', name='Best Score',
        line=dict(color='#00bcd4', width=2),
        marker=dict(size=6),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["avg_score"],
        mode='lines', name='Avg Score',
        line=dict(color='#8892a4', width=1, dash='dot'),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["best_sharpe"],
        mode='lines+markers', name='Best Sharpe',
        line=dict(color='#00e676', width=2),
        marker=dict(size=6),
    ), row=2, col=1)
    fig.update_layout(
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#8892a4'),
        hovermode='x unified',
        margin=dict(l=60, r=20, t=40, b=40),
        height=500,
    )
    fig.update_xaxes(gridcolor='#1a2744')
    fig.update_yaxes(gridcolor='#1a2744')
    return fig


def create_bootstrap_chart(bootstrap_results):
    """Create bootstrap distribution chart."""
    returns = bootstrap_results["returns"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        marker_color='#00bcd4',
        opacity=0.7,
        name='Return Distribution',
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#ff5252", line_width=2)
    fig.add_vline(x=np.mean(returns), line_dash="dash", line_color="#00e676", line_width=2)
    fig.update_layout(
        title=dict(text='Bootstrap Return Distribution', font=dict(color='#e0e0e0', size=16)),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#8892a4'),
        xaxis=dict(gridcolor='#1a2744', title='Return (%)'),
        yaxis=dict(gridcolor='#1a2744', title='Frequency'),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def strategy_description(strat: TradingStrategy) -> str:
    """Human-readable strategy description."""
    lines = []
    lines.append(f"Direction: {strat.direction.upper()}")
    lines.append(f"Stop Loss: {strat.stop_loss_pct*100:.1f}% | Take Profit: {strat.take_profit_pct*100:.1f}%")
    lines.append("")
    lines.append("ENTRY CONDITIONS (all must be true):")
    for i, c in enumerate(strat.entry_conditions):
        params_str = ", ".join(f"{k}={v}" for k, v in c.params.items())
        line = f"  {i+1}. {c.indicator_name}({params_str}) [{c.output_key}] {c.condition_type}"
        if c.condition_type in ("is_above", "is_below", "crosses_above", "crosses_below"):
            if c.compare_indicator:
                line += f" {c.compare_indicator}[{c.compare_output_key}]"
            else:
                line += f" {c.threshold:.2f}"
        lines.append(line)
    lines.append("")
    lines.append("EXIT CONDITIONS (any triggers exit):")
    for i, c in enumerate(strat.exit_conditions):
        params_str = ", ".join(f"{k}={v}" for k, v in c.params.items())
        line = f"  {i+1}. {c.indicator_name}({params_str}) [{c.output_key}] {c.condition_type}"
        if c.condition_type in ("is_above", "is_below", "crosses_above", "crosses_below"):
            if c.compare_indicator:
                line += f" {c.compare_indicator}[{c.compare_output_key}]"
            else:
                line += f" {c.threshold:.2f}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "data" not in st.session_state:
    st.session_state.data = None
if "train" not in st.session_state:
    st.session_state.train = None
if "val" not in st.session_state:
    st.session_state.val = None
if "test" not in st.session_state:
    st.session_state.test = None
if "running" not in st.session_state:
    st.session_state.running = False
if "selected_strategy_idx" not in st.session_state:
    st.session_state.selected_strategy_idx = 0


# ---------------------------------------------------------------------------
# Sidebar - Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuration")

    st.markdown("**Data Source**")
    data_source = st.selectbox("Source", ["Yahoo Finance", "CSV File"], key="data_source")

    if data_source == "Yahoo Finance":
        symbol = st.text_input("Symbol", value="SOL-USD")
        interval = st.selectbox("Timeframe", ["1h", "1d", "15m", "30m", "5m"], index=0)
        period = st.selectbox("Period", ["2y", "1y", "6mo", "3mo", "5y"], index=0)
    else:
        csv_file = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.markdown("**Split Ratios**")
    col1, col2 = st.columns(2)
    train_pct = col1.slider("Train %", 40, 80, 60) / 100
    val_pct = col2.slider("Val %", 10, 30, 20) / 100

    st.markdown("---")
    st.markdown("**GA Parameters**")
    pop_size = st.slider("Population Size", 20, 500, 100, step=10)
    n_generations = st.slider("Generations", 5, 100, 25)
    p_crossover = st.slider("Crossover Rate", 0.1, 1.0, 0.7)
    p_mutation = st.slider("Mutation Rate", 0.05, 0.5, 0.3)
    tournament_size = st.slider("Tournament Size", 2, 7, 3)
    elite_size = st.slider("Elite Size", 1, 20, 5)

    st.markdown("---")
    bootstrap_samples = st.slider("Bootstrap Samples", 10, 200, 50)
    bootstrap_threshold = st.number_input("Bootstrap Threshold (%)", value=0.0)

    st.markdown("---")

    # Load data button
    if st.button("Load Data", use_container_width=True):
        with st.spinner("Loading data..."):
            try:
                if data_source == "Yahoo Finance":
                    st.session_state.data = load_data_yfinance(symbol, interval, period)
                else:
                    if csv_file is not None:
                        import io
                        st.session_state.data = load_data_csv(io.BytesIO(csv_file.read()))
                    else:
                        st.error("Please upload a CSV file")

                if st.session_state.data is not None:
                    train, val, test = split_data(st.session_state.data, train_pct, val_pct)
                    st.session_state.train = train
                    st.session_state.val = val
                    st.session_state.test = test
                    st.success(f"Loaded {len(st.session_state.data)} bars")
            except Exception as e:
                st.error(f"Error loading data: {e}")

    # Load previous results
    if st.button("Load Saved Results", use_container_width=True):
        try:
            st.session_state.results = load_results()
            st.success("Results loaded!")
        except Exception as e:
            st.error(f"No saved results found: {e}")


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------

# Title
st.markdown('<div class="dashboard-title">GeneticAI Elite Dashboard</div>', unsafe_allow_html=True)

if st.session_state.data is not None:
    data = st.session_state.data
    info_text = (
        f"SYMBOL: {data_source == 'Yahoo Finance' and symbol or 'CSV'} | "
        f"TIMEFRAME: {data_source == 'Yahoo Finance' and interval or 'auto'} | "
        f"BARS: {len(data):,} | "
        f"SPLIT: {train_pct*100:.0f}% IS / {val_pct*100:.0f}% OOS | "
        f"THRESHOLD: {bootstrap_threshold}%"
    )
    st.markdown(f'<div class="dashboard-subtitle">{info_text}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="dashboard-subtitle">Load data to begin</div>', unsafe_allow_html=True)


# Control bar
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    if st.session_state.data is not None:
        if st.button("RUN EVOLUTION", use_container_width=True, type="primary"):
            st.session_state.running = True

with col2:
    analysis_mode = st.selectbox("Analysis Mode", [
        "Normal Training", "In-Sample", "Out-of-Sample", "Bootstrap"
    ])

with col3:
    sort_metric = st.selectbox("Sort Metric", [
        "fitness_score", "fitness_sharpe", "fitness_return",
        "fitness_winrate", "fitness_profit_factor", "fitness_sortino"
    ])


# ---------------------------------------------------------------------------
# Run Evolution
# ---------------------------------------------------------------------------
if st.session_state.running and st.session_state.train is not None:
    st.markdown('<div class="section-header">Evolution Progress</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    gen_chart_placeholder = st.empty()

    gen_stats_live = []

    def progress_callback(gen, total, stat):
        progress_bar.progress(gen / total)
        status_text.markdown(
            f"**Generation {gen}/{total}** | "
            f"Best Score: {stat['best_score']:.3f} | "
            f"Sharpe: {stat['best_sharpe']:.3f} | "
            f"Return: {stat['best_return']:.1f}% | "
            f"Time: {stat['total_time']:.1f}s"
        )
        gen_stats_live.append(stat)
        if gen % 3 == 0 or gen == total:
            gen_chart_placeholder.plotly_chart(
                create_generation_chart(gen_stats_live), use_container_width=True
            )

    results = run_evolution(
        df_train=st.session_state.train,
        df_val=st.session_state.val,
        pop_size=pop_size,
        n_generations=n_generations,
        p_crossover=p_crossover,
        p_mutation=p_mutation,
        tournament_size=tournament_size,
        elite_size=elite_size,
        progress_callback=progress_callback,
    )

    st.session_state.results = results
    st.session_state.running = False
    save_results(results)
    st.success("Evolution complete! Results saved.")
    st.rerun()


# ---------------------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------------------
if st.session_state.results is not None:
    results = st.session_state.results
    strategies = results["best_strategies"]
    gen_stats = results["generation_stats"]

    # Sort strategies by selected metric
    strategies_sorted = sorted(strategies, key=lambda s: getattr(s, sort_metric, 0), reverse=True)

    # Pick current strategy
    if st.session_state.selected_strategy_idx >= len(strategies_sorted):
        st.session_state.selected_strategy_idx = 0
    current_strat = strategies_sorted[st.session_state.selected_strategy_idx]

    # Determine which dataset to evaluate on
    if analysis_mode == "Normal Training" and st.session_state.train is not None:
        eval_df = st.session_state.train
        eval_label = "IN-SAMPLE (Training)"
    elif analysis_mode == "In-Sample" and st.session_state.train is not None:
        eval_df = st.session_state.train
        eval_label = "IN-SAMPLE"
    elif analysis_mode == "Out-of-Sample" and st.session_state.test is not None:
        eval_df = st.session_state.test
        eval_label = "OUT-OF-SAMPLE (Test)"
    else:
        eval_df = st.session_state.val if st.session_state.val is not None else st.session_state.train
        eval_label = "VALIDATION"

    # Evaluate current strategy
    if eval_df is not None:
        cache = IndicatorCache(eval_df)
        metrics = backtest_strategy(current_strat, eval_df, cache)

        # ---------------------------------------------------------------
        # Parameters section
        # ---------------------------------------------------------------
        st.markdown('<div class="section-header">Parameters</div>', unsafe_allow_html=True)

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        n_combos = len(strategies)
        n_survivors = sum(1 for s in strategies if s.fitness_trades > 10)

        with pcol1:
            st.markdown(metric_card("Combos", f"{n_combos}", "neutral"), unsafe_allow_html=True)
        with pcol2:
            st.markdown(metric_card("IS Survivors", f"{n_survivors}", "neutral"), unsafe_allow_html=True)
        with pcol3:
            bs_result = None
            if analysis_mode == "Bootstrap":
                bs_result = bootstrap_validation(
                    current_strat, st.session_state.data,
                    n_samples=bootstrap_samples,
                    threshold=bootstrap_threshold,
                )
                st.markdown(metric_card("Bootstrap", f"{len(strategies)}", "neutral"), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Generations", f"{len(gen_stats)}", "neutral"), unsafe_allow_html=True)
        with pcol4:
            st.markdown(metric_card("Final", f"{n_survivors}", "neutral"), unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # Key metrics row
        # ---------------------------------------------------------------
        st.markdown(f'<div class="section-header">{eval_label}</div>', unsafe_allow_html=True)

        mcol1, mcol2, mcol3, mcol4, mcol5, mcol6, mcol7 = st.columns(7)

        ret_color = "positive" if metrics["total_return"] > 0 else "negative"
        with mcol1:
            st.markdown(metric_card("Net Profit", f"{metrics['total_return']:+.2f}%", ret_color),
                        unsafe_allow_html=True)
        with mcol2:
            dd_color = "warning" if abs(metrics["max_drawdown"]) > 20 else "neutral"
            st.markdown(metric_card("Max Equity DD", f"{metrics['max_drawdown']:.2f}%", dd_color),
                        unsafe_allow_html=True)
        with mcol3:
            st.markdown(metric_card("Total Trades", f"{metrics['total_trades']}", "neutral"),
                        unsafe_allow_html=True)
        with mcol4:
            wr_color = "positive" if metrics["win_rate"] > 50 else "warning"
            st.markdown(metric_card("Win Rate", f"{metrics['win_rate']:.2f}%", wr_color),
                        unsafe_allow_html=True)
        with mcol5:
            sh_color = "positive" if metrics["sharpe"] > 1 else "neutral"
            st.markdown(metric_card("Sharpe Ratio", f"{metrics['sharpe']:.3f}", sh_color),
                        unsafe_allow_html=True)
        with mcol6:
            st.markdown(metric_card("Sortino Ratio", f"{metrics['sortino']:.3f}",
                                     "positive" if metrics["sortino"] > 1 else "neutral"),
                        unsafe_allow_html=True)
        with mcol7:
            if bs_result:
                bp_color = "positive" if bs_result["pass_rate"] > 70 else "warning"
                st.markdown(metric_card("Bootstrap Pass %", f"{bs_result['pass_rate']:.0f}%", bp_color),
                            unsafe_allow_html=True)
            else:
                pf_color = "positive" if metrics["profit_factor"] > 1.5 else "neutral"
                st.markdown(metric_card("Profit Factor", f"{metrics['profit_factor']:.3f}", pf_color),
                            unsafe_allow_html=True)

        # Simulation result badge
        is_validated = (
            metrics["sharpe"] > 0.3
            and metrics["total_return"] > 0
            and metrics["total_trades"] > 10
            and metrics["profit_factor"] > 1.0
        )
        badge_class = "badge-validated" if is_validated else "badge-failed"
        badge_text = "VALIDATED" if is_validated else "NOT VALIDATED"
        st.markdown(f'<div style="text-align: right;"><span class="{badge_class}">{badge_text}</span></div>',
                    unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # Strategy selector
        # ---------------------------------------------------------------
        st.markdown(f'<div class="section-header">Ranked Results: {len(strategies_sorted)} Strategies</div>',
                    unsafe_allow_html=True)

        strat_options = [
            f"#{i+1} | Score: {s.fitness_score:.2f} | Sharpe: {s.fitness_sharpe:.3f} | "
            f"Return: {s.fitness_return:.1f}% | WR: {s.fitness_winrate:.1f}%"
            for i, s in enumerate(strategies_sorted)
        ]

        selected_idx = st.selectbox("Select Strategy", range(len(strat_options)),
                                     format_func=lambda i: strat_options[i])
        if selected_idx != st.session_state.selected_strategy_idx:
            st.session_state.selected_strategy_idx = selected_idx
            st.rerun()

        # ---------------------------------------------------------------
        # Tabs: Charts / Details / All Strategies
        # ---------------------------------------------------------------
        tab_oos, tab_is, tab_bootstrap, tab_details, tab_all = st.tabs([
            "OUT-OF-SAMPLE", "IN-SAMPLE", "BOOTSTRAP", "Strategy Details", "All Strategies"
        ])

        with tab_is:
            if st.session_state.train is not None:
                cache_is = IndicatorCache(st.session_state.train)
                metrics_is = backtest_strategy(current_strat, st.session_state.train, cache_is)
                if metrics_is["equity_curve"] is not None:
                    st.plotly_chart(create_equity_chart(metrics_is["equity_curve"],
                                                        "In-Sample Equity Curve"),
                                   use_container_width=True)
                    if metrics_is["portfolio"] is not None:
                        st.plotly_chart(create_drawdown_chart(metrics_is["portfolio"]),
                                       use_container_width=True)

                    # Detailed metrics table
                    st.markdown('<div class="section-header">Normal Training</div>', unsafe_allow_html=True)
                    stats_data = {
                        "Metric": ["Net Profit", "Net Profit %", "Avg Bars In Trades",
                                   "Avg PnL", "Profit Factor", "Win Rate", "Sharpe",
                                   "Sortino", "Total Closed Trades", "Max Drawdown %"],
                        "ALL": [
                            f"{metrics_is['total_return']:.2f}",
                            f"{metrics_is['total_return']:.2f}%",
                            "N/A",
                            f"{metrics_is['total_return']/max(metrics_is['total_trades'],1):.2f}",
                            f"{metrics_is['profit_factor']:.2f}",
                            f"{metrics_is['win_rate']:.2f}%",
                            f"{metrics_is['sharpe']:.2f}",
                            f"{metrics_is['sortino']:.2f}",
                            f"{metrics_is['total_trades']}",
                            f"{metrics_is['max_drawdown']:.2f}%",
                        ],
                    }
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                else:
                    st.warning("No trades generated in-sample")

        with tab_oos:
            test_df = st.session_state.test
            if test_df is not None:
                cache_oos = IndicatorCache(test_df)
                metrics_oos = backtest_strategy(current_strat, test_df, cache_oos)
                if metrics_oos["equity_curve"] is not None:
                    st.plotly_chart(create_equity_chart(metrics_oos["equity_curve"],
                                                        "Out-of-Sample Equity Curve"),
                                   use_container_width=True)
                    if metrics_oos["portfolio"] is not None:
                        st.plotly_chart(create_drawdown_chart(metrics_oos["portfolio"]),
                                       use_container_width=True)

                    # OOS metrics
                    oos_cols = st.columns(5)
                    with oos_cols[0]:
                        st.metric("OOS Return", f"{metrics_oos['total_return']:.2f}%")
                    with oos_cols[1]:
                        st.metric("OOS Sharpe", f"{metrics_oos['sharpe']:.3f}")
                    with oos_cols[2]:
                        st.metric("OOS Win Rate", f"{metrics_oos['win_rate']:.1f}%")
                    with oos_cols[3]:
                        st.metric("OOS Max DD", f"{metrics_oos['max_drawdown']:.2f}%")
                    with oos_cols[4]:
                        st.metric("OOS Trades", f"{metrics_oos['total_trades']}")
                else:
                    st.warning("No trades generated out-of-sample")
            else:
                st.info("Load data first to see out-of-sample results")

        with tab_bootstrap:
            if st.session_state.data is not None:
                if st.button("Run Bootstrap Validation"):
                    with st.spinner("Running bootstrap..."):
                        bs = bootstrap_validation(
                            current_strat, st.session_state.data,
                            n_samples=bootstrap_samples,
                            threshold=bootstrap_threshold,
                        )
                        st.session_state.bootstrap_result = bs

                if hasattr(st.session_state, 'bootstrap_result') and st.session_state.bootstrap_result:
                    bs = st.session_state.bootstrap_result
                    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                    with bcol1:
                        st.markdown(metric_card("Pass Rate", f"{bs['pass_rate']:.1f}%",
                                                 "positive" if bs['pass_rate'] > 70 else "warning"),
                                    unsafe_allow_html=True)
                    with bcol2:
                        st.markdown(metric_card("Mean Return", f"{bs['mean_return']:.2f}%",
                                                 "positive" if bs['mean_return'] > 0 else "negative"),
                                    unsafe_allow_html=True)
                    with bcol3:
                        st.markdown(metric_card("Std Return", f"{bs['std_return']:.2f}%", "neutral"),
                                    unsafe_allow_html=True)
                    with bcol4:
                        st.markdown(metric_card("Mean Sharpe", f"{bs['mean_sharpe']:.3f}",
                                                 "positive" if bs['mean_sharpe'] > 0 else "negative"),
                                    unsafe_allow_html=True)
                    st.plotly_chart(create_bootstrap_chart(bs), use_container_width=True)

        with tab_details:
            st.markdown('<div class="section-header">Strategy Parameters</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="param-box"><pre>{strategy_description(current_strat)}</pre></div>',
                        unsafe_allow_html=True)

            # Evolution progress chart
            if gen_stats:
                st.markdown('<div class="section-header">Evolution Progress</div>', unsafe_allow_html=True)
                st.plotly_chart(create_generation_chart(gen_stats), use_container_width=True)

        with tab_all:
            st.markdown('<div class="section-header">All Top Strategies</div>', unsafe_allow_html=True)

            all_strats_data = []
            for i, s in enumerate(strategies_sorted):
                all_strats_data.append({
                    "Rank": i + 1,
                    "Score": f"{s.fitness_score:.3f}",
                    "Sharpe": f"{s.fitness_sharpe:.3f}",
                    "Return %": f"{s.fitness_return:.2f}",
                    "Win Rate %": f"{s.fitness_winrate:.1f}",
                    "Max DD %": f"{s.fitness_max_dd:.2f}",
                    "Profit Factor": f"{s.fitness_profit_factor:.3f}",
                    "Sortino": f"{s.fitness_sortino:.3f}",
                    "Trades": s.fitness_trades,
                    "Direction": s.direction,
                    "# Entry Rules": len(s.entry_conditions),
                    "# Exit Rules": len(s.exit_conditions),
                    "SL %": f"{s.stop_loss_pct*100:.1f}",
                    "TP %": f"{s.take_profit_pct*100:.1f}",
                })
            st.dataframe(pd.DataFrame(all_strats_data), use_container_width=True, hide_index=True)

            # Scatter plot: Return vs Sharpe
            scatter_df = pd.DataFrame(all_strats_data)
            fig = px.scatter(
                scatter_df, x="Return %", y="Sharpe",
                color="Win Rate %", size="Trades",
                hover_data=["Rank", "Profit Factor", "Max DD %"],
                title="Strategy Return vs Sharpe Ratio",
            )
            fig.update_layout(
                paper_bgcolor='#0a0e17',
                plot_bgcolor='#0a0e17',
                font=dict(color='#8892a4'),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------
    # Evolution statistics
    # ---------------------------------------------------------------
    if gen_stats:
        st.markdown("---")
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        with ecol1:
            st.metric("Total Time", f"{results['total_time']:.1f}s")
        with ecol2:
            st.metric("Generations", len(gen_stats))
        with ecol3:
            st.metric("Population", pop_size)
        with ecol4:
            total_evals = len(results.get("all_evaluated", []))
            st.metric("Total Evaluations", f"{total_evals:,}")

else:
    # No results yet - show instructions
    st.markdown("---")
    st.markdown("""
    ### Getting Started

    1. **Configure** your data source and GA parameters in the sidebar
    2. **Load Data** to download/import price data
    3. **Run Evolution** to start the genetic optimization
    4. **Analyze** the results across In-Sample, Out-of-Sample, and Bootstrap tabs

    The genetic algorithm will evolve combinations of 25 technical indicators including:

    """)

    # Show available indicators
    ind_cols = st.columns(4)
    for i, name in enumerate(INDICATOR_NAMES):
        with ind_cols[i % 4]:
            desc = INDICATOR_REGISTRY[name]["description"]
            st.markdown(f"- **{name}**: {desc}")

    st.markdown("""
    ---
    Each strategy combines multiple indicator conditions for entry/exit signals,
    with stop-loss and take-profit levels, optimized through genetic evolution.
    """)
