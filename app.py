"""
GeneticAI Elite Dashboard
=========================
Streamlit dashboard for running and viewing genetic indicator optimization results.
Dark-themed, inspired by the Wolfpack Elite Dashboard design.
Features: Walk-forward analysis, configurable strategy complexity,
favorites system with strategy export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import datetime
from pathlib import Path

from genetic_indicator_engine import (
    run_evolution, backtest_strategy, bootstrap_validation,
    load_data_yfinance, load_data_csv, split_data,
    IndicatorCache, INDICATOR_NAMES, INDICATOR_REGISTRY,
    save_results, load_results, TradingStrategy, compute_fitness_score,
    run_walk_forward, generate_walk_forward_folds,
    save_favorites, load_favorites, strategy_from_dict, export_strategy_script,
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

    /* Validated badges */
    .badge-strong {
        background: #00e676;
        color: #000;
        padding: 8px 20px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-validated {
        background: #00bcd4;
        color: #000;
        padding: 8px 20px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-weak {
        background: #ffc107;
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
        xaxis=dict(gridcolor='#1a2744', title=''),
        yaxis=dict(gridcolor='#1a2744', title='Equity ($)'),
        hovermode='x unified',
        margin=dict(l=60, r=20, t=40, b=40),
        height=400,
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
        title=dict(text='Drawdown', font=dict(color='#e0e0e0', size=14)),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#8892a4'),
        xaxis=dict(gridcolor='#1a2744'),
        yaxis=dict(gridcolor='#1a2744', title='Drawdown %'),
        hovermode='x unified',
        margin=dict(l=60, r=20, t=40, b=40),
        height=250,
    )
    return fig


def create_generation_chart(gen_stats):
    """Create generation progress chart. Handles both normal and walk-forward formats."""
    if not gen_stats:
        return go.Figure()
    df = pd.DataFrame(gen_stats)
    # Handle walk-forward format (uses global_gen) vs normal (uses generation)
    x_col = "generation" if "generation" in df.columns else "global_gen"
    if x_col not in df.columns:
        x_col = "gen"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Best Fitness Score", "Best Sharpe Ratio"),
                        vertical_spacing=0.12)
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df["best_score"],
        mode='lines+markers', name='Best Score',
        line=dict(color='#00bcd4', width=2),
        marker=dict(size=6),
    ), row=1, col=1)
    if "avg_score" in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df["avg_score"],
            mode='lines', name='Avg Score',
            line=dict(color='#8892a4', width=1, dash='dot'),
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df["best_sharpe"],
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


def create_trade_distribution_chart(trades_df):
    """Create a trade PnL distribution histogram."""
    if trades_df is None or trades_df.empty:
        return None
    pnl_col = None
    for col in ["PnL", "Return", "pnl", "return"]:
        if col in trades_df.columns:
            pnl_col = col
            break
    if pnl_col is None:
        return None

    fig = go.Figure()
    values = trades_df[pnl_col].dropna()
    colors = ['#00e676' if v > 0 else '#ff5252' for v in values]
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=30,
        marker_color='#00bcd4',
        opacity=0.7,
        name='Trade PnL',
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#ffc107", line_width=2)
    fig.update_layout(
        title=dict(text='Trade PnL Distribution', font=dict(color='#e0e0e0', size=14)),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#8892a4'),
        xaxis=dict(gridcolor='#1a2744', title='PnL'),
        yaxis=dict(gridcolor='#1a2744', title='Frequency'),
        margin=dict(l=60, r=20, t=40, b=40),
        height=300,
    )
    return fig


def _update_live_wf_chart(placeholder, all_stats, total_folds, gens_per_fold):
    """Render a live 2x2 chart grid during walk-forward: Score, Sharpe, Return, Profit Factor."""
    if not all_stats:
        return
    df = pd.DataFrame(all_stats)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Best Fitness Score", "Best Sharpe Ratio",
                         "Best Return %", "Best Profit Factor"),
        vertical_spacing=0.15, horizontal_spacing=0.08,
    )

    # Color each fold differently
    fold_colors = ['#00bcd4', '#00e676', '#ffc107', '#ff5252', '#ab47bc',
                   '#29b6f6', '#66bb6a', '#ffa726', '#ef5350', '#7e57c2',
                   '#26c6da', '#9ccc65']

    for fold_num in df["fold"].unique():
        fold_df = df[df["fold"] == fold_num]
        color = fold_colors[(fold_num - 1) % len(fold_colors)]
        name = f"Fold {fold_num}"

        fig.add_trace(go.Scatter(
            x=fold_df["global_gen"], y=fold_df["best_score"],
            mode='lines', name=name, line=dict(color=color, width=2),
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=fold_df["global_gen"], y=fold_df["best_sharpe"],
            mode='lines', name=name, line=dict(color=color, width=2),
            showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=fold_df["global_gen"], y=fold_df["best_return"],
            mode='lines', name=name, line=dict(color=color, width=2),
            showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=fold_df["global_gen"], y=fold_df["best_pf"],
            mode='lines', name=name, line=dict(color=color, width=2),
            showlegend=False,
        ), row=2, col=2)

    # Add fold separator lines
    for f in range(1, total_folds):
        x_sep = f * gens_per_fold
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(x=x_sep, line_dash="dot", line_color="#333d55",
                              line_width=1, row=row, col=col)

    fig.update_layout(
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#8892a4', size=11),
        height=500,
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor='#1a2744', title_text="", showticklabels=True)
    fig.update_yaxes(gridcolor='#1a2744')

    placeholder.plotly_chart(fig, use_container_width=True)


def create_walk_forward_fold_chart(fold_results, train_months, test_months):
    """Create horizontal bar chart showing train/test windows per fold (like the reference screenshots)."""
    fig = go.Figure()

    for fold in reversed(fold_results):
        fold_label = f"Fold {fold['fold_num']}"
        train_start = pd.Timestamp(fold["train_start"])
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])

        # Train bar
        fig.add_trace(go.Bar(
            y=[fold_label],
            x=[(train_end - train_start).days],
            base=[(train_start - pd.Timestamp(fold_results[0]["train_start"])).days],
            orientation='h',
            marker_color='#1e88e5',
            name=f'Train ({train_months}mo)' if fold["fold_num"] == 1 else None,
            showlegend=(fold["fold_num"] == 1),
            hovertemplate=f'Train: {train_start.strftime("%Y-%m-%d")} to {train_end.strftime("%Y-%m-%d")}<extra></extra>',
        ))
        # Test bar
        fig.add_trace(go.Bar(
            y=[fold_label],
            x=[(test_end - test_start).days],
            base=[(test_start - pd.Timestamp(fold_results[0]["train_start"])).days],
            orientation='h',
            marker_color='#ff9800',
            name=f'Blind Test ({test_months}mo)' if fold["fold_num"] == 1 else None,
            showlegend=(fold["fold_num"] == 1),
            hovertemplate=f'Test: {test_start.strftime("%Y-%m-%d")} to {test_end.strftime("%Y-%m-%d")}<extra></extra>',
        ))

    fig.update_layout(
        title=dict(text='Walk-Forward Fold Windows', font=dict(color='#e0e0e0', size=16)),
        paper_bgcolor='#0a0e17',
        plot_bgcolor='#0a0e17',
        font=dict(color='#8892a4'),
        xaxis=dict(gridcolor='#1a2744', title='Days from start'),
        yaxis=dict(gridcolor='#1a2744'),
        barmode='overlay',
        margin=dict(l=80, r=20, t=40, b=40),
        height=max(250, len(fold_results) * 40 + 100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
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


def get_validation_badge(metrics):
    """Get tiered validation badge based on metrics."""
    sharpe = metrics["sharpe"]
    pf = metrics["profit_factor"]
    wr = metrics["win_rate"]
    ret = metrics["total_return"]
    trades = metrics["total_trades"]

    if sharpe > 1.0 and pf > 1.5 and wr > 55 and ret > 0 and trades > 15:
        return "badge-strong", "STRONG"
    elif sharpe > 0.3 and pf > 1.0 and ret > 0 and trades > 10:
        return "badge-validated", "VALIDATED"
    elif sharpe > 0 and ret > 0 and trades > 5:
        return "badge-weak", "WEAK"
    else:
        return "badge-failed", "NOT VALIDATED"


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
if "running_wf" not in st.session_state:
    st.session_state.running_wf = False
if "selected_strategy_idx" not in st.session_state:
    st.session_state.selected_strategy_idx = 0
if "wf_results" not in st.session_state:
    st.session_state.wf_results = None
if "favorites" not in st.session_state:
    st.session_state.favorites = load_favorites()


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

        # Show valid periods based on interval
        interval_periods = {
            "1h": ["2y", "1y", "6mo", "3mo"],
            "1d": ["5y", "2y", "1y", "6mo", "3mo"],
            "15m": ["1mo", "7d"],
            "30m": ["1mo", "7d"],
            "5m": ["1mo", "7d"],
        }
        valid_periods = interval_periods.get(interval, ["2y", "1y", "6mo"])
        period = st.selectbox("Period", valid_periods, index=0)
        st.caption(f"Max for {interval}: {valid_periods[0]}")
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
    st.markdown("**Strategy Complexity**")
    st.caption("How many indicator rules per strategy. More = stricter signals but fewer trades.")
    sc_col1, sc_col2 = st.columns(2)
    min_entry = sc_col1.slider("Min Entry Indicators", 1, 4, 2)
    max_entry = sc_col2.slider("Max Entry Indicators", 2, 6, 4)
    sc_col3, sc_col4 = st.columns(2)
    min_exit = sc_col3.slider("Min Exit Indicators", 1, 2, 1)
    max_exit = sc_col4.slider("Max Exit Indicators", 1, 3, 2)
    # Enforce min <= max
    max_entry = max(max_entry, min_entry)
    max_exit = max(max_exit, min_exit)

    st.markdown("---")
    st.markdown("**Walk-Forward Analysis**")
    wf_enabled = st.checkbox("Enable Walk-Forward", value=False)
    if wf_enabled:
        wf_train_months = st.slider("Training Window (months)", 6, 24, 12)
        wf_test_months = st.slider("Test Window (months)", 1, 6, 3)
        wf_gens = st.slider("Generations per fold", 5, 50, 15)
        wf_pop = st.slider("Population per fold", 20, 200, 50, step=10)
        st.caption("Reduced pop/gens per fold for speed. Walk-forward runs the GA multiple times.")

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

                if st.session_state.data is not None and len(st.session_state.data) > 0:
                    train, val, test = split_data(st.session_state.data, train_pct, val_pct)
                    st.session_state.train = train
                    st.session_state.val = val
                    st.session_state.test = test
                    st.session_state.results = None
                    st.session_state.wf_results = None
                    st.success(f"Loaded {len(st.session_state.data):,} bars | Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
            except ValueError as e:
                st.error(str(e))
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
    sym_display = symbol if data_source == "Yahoo Finance" else "CSV"
    int_display = interval if data_source == "Yahoo Finance" else "auto"
    info_text = (
        f"SYMBOL: {sym_display} | "
        f"TIMEFRAME: {int_display} | "
        f"BARS: {len(data):,} | "
        f"SPLIT: {train_pct*100:.0f}% IS / {val_pct*100:.0f}% OOS | "
        f"ENTRY: {min_entry}-{max_entry} | EXIT: {min_exit}-{max_exit}"
    )
    st.markdown(f'<div class="dashboard-subtitle">{info_text}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="dashboard-subtitle">Load data to begin</div>', unsafe_allow_html=True)


# Control bar
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    if st.session_state.data is not None:
        if wf_enabled:
            if st.button("RUN WALK-FORWARD", use_container_width=True, type="primary"):
                st.session_state.running_wf = True
        else:
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
# Run Evolution (standard mode)
# ---------------------------------------------------------------------------
if st.session_state.running and st.session_state.train is not None:
    st.markdown('<div class="section-header">Evolution Progress</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    gen_chart_placeholder = st.empty()

    gen_stats_live = []
    all_gen_stats_evo = []

    def progress_callback(gen, total, stat):
        progress_bar.progress(gen / total)
        status_text.markdown(
            f"**Generation {gen}/{total}** | "
            f"Best Score: {stat['best_score']:.3f} | "
            f"Sharpe: {stat['best_sharpe']:.3f} | "
            f"Return: {stat['best_return']:.1f}% | "
            f"PF: {stat['best_pf']:.2f} | "
            f"WR: {stat['best_winrate']:.1f}% | "
            f"DD: {stat['best_max_dd']:.1f}% | "
            f"Time: {stat['total_time']:.1f}s"
        )
        gen_stats_live.append(stat)
        all_gen_stats_evo.append({
            "fold": 1, "gen": gen, "global_gen": gen,
            "best_score": stat["best_score"], "best_sharpe": stat["best_sharpe"],
            "best_return": stat["best_return"], "best_pf": stat["best_pf"],
            "best_winrate": stat["best_winrate"], "best_max_dd": stat["best_max_dd"],
            "avg_score": stat["avg_score"],
        })
        if gen % 2 == 0 or gen == total:
            _update_live_wf_chart(gen_chart_placeholder, all_gen_stats_evo, 1, total)

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
        min_entry=min_entry,
        max_entry=max_entry,
        min_exit=min_exit,
        max_exit=max_exit,
    )

    st.session_state.results = results
    st.session_state.running = False
    save_results(results)
    st.success("Evolution complete! Results saved.")
    st.rerun()


# ---------------------------------------------------------------------------
# Run Walk-Forward (with live per-generation progress)
# ---------------------------------------------------------------------------
if st.session_state.running_wf and st.session_state.data is not None:
    st.markdown('<div class="section-header">Walk-Forward Analysis Progress</div>', unsafe_allow_html=True)

    # Calculate folds first to show what's coming
    from genetic_indicator_engine import generate_walk_forward_folds, IndicatorCache as IC
    wf_folds = generate_walk_forward_folds(st.session_state.data, wf_train_months, wf_test_months)
    n_total_folds = len(wf_folds)

    if n_total_folds == 0:
        st.error("Not enough data for the selected train/test window sizes. Try shorter windows.")
        st.session_state.running_wf = False
    else:
        st.info(f"Running {n_total_folds} folds: {wf_train_months}mo train / {wf_test_months}mo test | "
                f"{wf_pop} pop x {wf_gens} gens per fold")

        progress_bar_wf = st.progress(0)
        status_text_wf = st.empty()

        # Live charts - 2x2 grid: Score+Sharpe (top), Return+PF (bottom)
        live_chart_placeholder = st.empty()
        fold_status_text = st.empty()

        # We'll run walk-forward manually so we can inject per-gen progress
        import time as _time
        wf_start_time = _time.time()
        fold_results_list = []
        oos_equity_segments = []
        all_gen_stats_wf = []  # Track ALL generation stats across all folds

        for fi, fold in enumerate(wf_folds):
            fold_start_time = _time.time()

            # Per-generation callback for THIS fold
            def make_gen_callback(fold_idx, total_folds, gen_total, all_stats):
                def gen_cb(gen, total, stat):
                    # Overall progress
                    overall = (fold_idx * gen_total + gen) / (total_folds * gen_total)
                    progress_bar_wf.progress(min(overall, 1.0))
                    status_text_wf.markdown(
                        f"**Fold {fold_idx+1}/{total_folds}** | Gen {gen}/{total} | "
                        f"Best Score: {stat['best_score']:.3f} | "
                        f"Sharpe: {stat['best_sharpe']:.3f} | "
                        f"Return: {stat['best_return']:.1f}% | "
                        f"PF: {stat['best_pf']:.2f} | "
                        f"WR: {stat['best_winrate']:.1f}%"
                    )
                    # Track for live chart
                    all_stats.append({
                        "fold": fold_idx + 1,
                        "gen": gen,
                        "global_gen": fold_idx * gen_total + gen,
                        "best_score": stat["best_score"],
                        "best_sharpe": stat["best_sharpe"],
                        "best_return": stat["best_return"],
                        "best_pf": stat["best_pf"],
                        "best_winrate": stat["best_winrate"],
                        "best_max_dd": stat["best_max_dd"],
                        "avg_score": stat["avg_score"],
                    })
                    # Update live chart every 2 gens or on last gen
                    if gen % 2 == 0 or gen == total:
                        _update_live_wf_chart(live_chart_placeholder, all_stats, total_folds, gen_total)
                return gen_cb

            gen_callback = make_gen_callback(fi, n_total_folds, wf_gens, all_gen_stats_wf)

            # Split fold's train data: 80% sub-train, 20% sub-val for IS/OOS fitness penalty
            fold_train_full = fold["train_df"]
            split_idx = int(len(fold_train_full) * 0.8)
            sub_train = fold_train_full.iloc[:split_idx]
            sub_val = fold_train_full.iloc[split_idx:]

            # Run GA with sub-validation to activate overfitting penalty
            fold_results = run_evolution(
                df_train=sub_train,
                df_val=sub_val,
                pop_size=wf_pop,
                n_generations=wf_gens,
                p_crossover=p_crossover,
                p_mutation=p_mutation,
                tournament_size=tournament_size,
                elite_size=elite_size,
                progress_callback=gen_callback,
                min_entry=min_entry,
                max_entry=max_entry,
                min_exit=min_exit,
                max_exit=max_exit,
            )

            best = fold_results["best_strategy"]

            # Evaluate on train (IS) and blind test (OOS)
            cache_train = IC(fold["train_df"])
            train_metrics = backtest_strategy(best, fold["train_df"], cache_train)
            cache_test = IC(fold["test_df"])
            test_metrics = backtest_strategy(best, fold["test_df"], cache_test)

            fold_time = _time.time() - fold_start_time

            if test_metrics["equity_curve"] is not None:
                oos_equity_segments.append(test_metrics["equity_curve"])

            entry_indicators = [c.indicator_name for c in best.entry_conditions]
            exit_indicators = [c.indicator_name for c in best.exit_conditions]

            fold_result_entry = {
                "fold_num": fold["fold_num"],
                "train_start": str(fold["train_start"]),
                "train_end": str(fold["train_end"]),
                "test_start": str(fold["test_start"]),
                "test_end": str(fold["test_end"]),
                "train_bars": len(fold["train_df"]),
                "test_bars": len(fold["test_df"]),
                "strategy": best,
                "entry_indicators": entry_indicators,
                "exit_indicators": exit_indicators,
                "direction": best.direction,
                "train_return": train_metrics["total_return"],
                "train_sharpe": train_metrics["sharpe"],
                "train_trades": train_metrics["total_trades"],
                "oos_return": test_metrics["total_return"],
                "oos_sharpe": test_metrics["sharpe"],
                "oos_trades": test_metrics["total_trades"],
                "oos_winrate": test_metrics["win_rate"],
                "oos_max_dd": test_metrics["max_drawdown"],
                "oos_profit_factor": test_metrics["profit_factor"],
                "fold_time": fold_time,
            }
            fold_results_list.append(fold_result_entry)

            fold_status_text.markdown(
                f"Fold {fi+1} done: OOS Return **{test_metrics['total_return']:+.2f}%** | "
                f"Sharpe **{test_metrics['sharpe']:.3f}** | "
                f"PF **{test_metrics['profit_factor']:.2f}** | "
                f"WR **{test_metrics['win_rate']:.1f}%** | "
                f"DD **{test_metrics['max_drawdown']:.1f}%** | "
                f"Time: {fold_time:.1f}s"
            )

        # Stitch equity curves
        from genetic_indicator_engine import _stitch_equity_curves
        stitched = _stitch_equity_curves(oos_equity_segments)

        oos_returns = [f["oos_return"] for f in fold_results_list]
        oos_sharpes = [f["oos_sharpe"] for f in fold_results_list]
        profitable_folds = sum(1 for r in oos_returns if r > 0)
        total_wf_time = _time.time() - wf_start_time

        wf_results = {
            "folds": fold_results_list,
            "n_folds": n_total_folds,
            "stitched_equity": stitched,
            "mean_oos_return": np.mean(oos_returns),
            "median_oos_return": np.median(oos_returns),
            "mean_oos_sharpe": np.mean(oos_sharpes),
            "profitable_folds": profitable_folds,
            "profitable_pct": profitable_folds / n_total_folds * 100,
            "total_oos_return": sum(oos_returns),
            "total_time": total_wf_time,
            "train_months": wf_train_months,
            "test_months": wf_test_months,
        }

        st.session_state.wf_results = wf_results
        st.session_state.running_wf = False

        # Also populate st.session_state.results so the main display works
        # Collect best strategies from all folds
        wf_best_strategies = [f["strategy"] for f in fold_results_list if f["strategy"] is not None]
        # Score them by OOS performance (not training!) for honest ranking
        for fi_idx, f in enumerate(fold_results_list):
            if f["strategy"] is not None:
                f["strategy"].fitness_score = compute_fitness_score({
                    "sharpe": f["oos_sharpe"], "sortino": 0.0,
                    "total_return": f["oos_return"], "win_rate": f["oos_winrate"],
                    "profit_factor": f["oos_profit_factor"],
                    "max_drawdown": f["oos_max_dd"], "total_trades": f["oos_trades"],
                })
                f["strategy"].fitness_sharpe = f["oos_sharpe"]
                f["strategy"].fitness_return = f["oos_return"]
                f["strategy"].fitness_trades = f["oos_trades"]
                f["strategy"].fitness_winrate = f["oos_winrate"]
                f["strategy"].fitness_max_dd = f["oos_max_dd"]
                f["strategy"].fitness_profit_factor = f["oos_profit_factor"]

        st.session_state.results = {
            "best_strategies": wf_best_strategies,
            "best_strategy": wf_best_strategies[0] if wf_best_strategies else None,
            "generation_stats": all_gen_stats_wf,
            "all_evaluated": [],
            "total_time": total_wf_time,
        }

        st.success(f"Walk-forward complete! {n_total_folds} folds in {total_wf_time:.1f}s | "
                   f"Profitable: {profitable_folds}/{n_total_folds}")
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
        st.markdown('<div class="section-header">Run Summary</div>', unsafe_allow_html=True)

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        n_combos = len(strategies)
        n_survivors = sum(1 for s in strategies if s.fitness_trades > 10)

        with pcol1:
            st.markdown(metric_card("Population", f"{pop_size}", "neutral"), unsafe_allow_html=True)
        with pcol2:
            st.markdown(metric_card("Generations", f"{len(gen_stats)}", "neutral"), unsafe_allow_html=True)
        with pcol3:
            total_evals = len(results.get("all_evaluated", []))
            st.markdown(metric_card("Evaluations", f"{total_evals:,}", "neutral"), unsafe_allow_html=True)
        with pcol4:
            best_score = strategies_sorted[0].fitness_score if strategies_sorted else 0
            st.markdown(metric_card("Best Score", f"{best_score:.2f}",
                                     "positive" if best_score > 5 else "neutral"), unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # Key metrics row - 8 columns for all important metrics
        # ---------------------------------------------------------------
        st.markdown(f'<div class="section-header">{eval_label}</div>', unsafe_allow_html=True)

        mcols = st.columns(8)

        ret_color = "positive" if metrics["total_return"] > 0 else "negative"
        with mcols[0]:
            st.markdown(metric_card("Net Profit", f"{metrics['total_return']:+.2f}%", ret_color),
                        unsafe_allow_html=True)
        with mcols[1]:
            sh_color = "positive" if metrics["sharpe"] > 1 else "neutral"
            st.markdown(metric_card("Sharpe Ratio", f"{metrics['sharpe']:.3f}", sh_color),
                        unsafe_allow_html=True)
        with mcols[2]:
            wr_color = "positive" if metrics["win_rate"] > 50 else "warning"
            st.markdown(metric_card("Win Rate", f"{metrics['win_rate']:.1f}%", wr_color),
                        unsafe_allow_html=True)
        with mcols[3]:
            avg_pnl = metrics["total_return"] / max(metrics["total_trades"], 1)
            avg_color = "positive" if avg_pnl > 0 else "negative"
            st.markdown(metric_card("Avg Trade", f"{avg_pnl:+.2f}%", avg_color),
                        unsafe_allow_html=True)
        with mcols[4]:
            dd_color = "warning" if abs(metrics["max_drawdown"]) > 20 else "neutral"
            st.markdown(metric_card("Max Drawdown", f"{metrics['max_drawdown']:.2f}%", dd_color),
                        unsafe_allow_html=True)
        with mcols[5]:
            pf_color = "positive" if metrics["profit_factor"] > 1.5 else "neutral"
            st.markdown(metric_card("Profit Factor", f"{metrics['profit_factor']:.3f}", pf_color),
                        unsafe_allow_html=True)
        with mcols[6]:
            st.markdown(metric_card("Sortino", f"{metrics['sortino']:.3f}",
                                     "positive" if metrics["sortino"] > 1 else "neutral"),
                        unsafe_allow_html=True)
        with mcols[7]:
            st.markdown(metric_card("Total Trades", f"{metrics['total_trades']}", "neutral"),
                        unsafe_allow_html=True)

        # Validation badge (tiered)
        badge_class, badge_text = get_validation_badge(metrics)
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
        # Tabs
        # ---------------------------------------------------------------
        tab_names = ["OUT-OF-SAMPLE", "IN-SAMPLE", "BOOTSTRAP", "WALK-FORWARD",
                      "Strategy Details", "All Strategies", "FAVORITES"]
        tab_oos, tab_is, tab_bootstrap, tab_wf, tab_details, tab_all, tab_fav = st.tabs(tab_names)

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

                    # Trade distribution
                    trade_chart = create_trade_distribution_chart(metrics_is.get("trades"))
                    if trade_chart:
                        st.plotly_chart(trade_chart, use_container_width=True)

                    # Detailed metrics table
                    st.markdown('<div class="section-header">In-Sample Metrics</div>', unsafe_allow_html=True)
                    stats_data = {
                        "Metric": ["Net Profit %", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                                   "Profit Factor", "Win Rate", "Avg Trade %",
                                   "Total Trades", "Max Drawdown %"],
                        "Value": [
                            f"{metrics_is['total_return']:.2f}%",
                            f"{metrics_is['sharpe']:.3f}",
                            f"{metrics_is['sortino']:.3f}",
                            f"{metrics_is['calmar']:.3f}",
                            f"{metrics_is['profit_factor']:.3f}",
                            f"{metrics_is['win_rate']:.2f}%",
                            f"{metrics_is['total_return']/max(metrics_is['total_trades'],1):.3f}%",
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

                    # OOS metrics row
                    oos_cols = st.columns(6)
                    with oos_cols[0]:
                        st.markdown(metric_card("OOS Return", f"{metrics_oos['total_return']:+.2f}%",
                                                 "positive" if metrics_oos['total_return'] > 0 else "negative"),
                                    unsafe_allow_html=True)
                    with oos_cols[1]:
                        st.markdown(metric_card("OOS Sharpe", f"{metrics_oos['sharpe']:.3f}",
                                                 "positive" if metrics_oos['sharpe'] > 1 else "neutral"),
                                    unsafe_allow_html=True)
                    with oos_cols[2]:
                        st.markdown(metric_card("OOS Win Rate", f"{metrics_oos['win_rate']:.1f}%",
                                                 "positive" if metrics_oos['win_rate'] > 50 else "warning"),
                                    unsafe_allow_html=True)
                    with oos_cols[3]:
                        st.markdown(metric_card("OOS Max DD", f"{metrics_oos['max_drawdown']:.2f}%",
                                                 "warning" if abs(metrics_oos['max_drawdown']) > 20 else "neutral"),
                                    unsafe_allow_html=True)
                    with oos_cols[4]:
                        st.markdown(metric_card("Profit Factor", f"{metrics_oos['profit_factor']:.3f}",
                                                 "positive" if metrics_oos['profit_factor'] > 1.5 else "neutral"),
                                    unsafe_allow_html=True)
                    with oos_cols[5]:
                        st.markdown(metric_card("OOS Trades", f"{metrics_oos['total_trades']}", "neutral"),
                                    unsafe_allow_html=True)

                    # Trade distribution
                    trade_chart = create_trade_distribution_chart(metrics_oos.get("trades"))
                    if trade_chart:
                        st.plotly_chart(trade_chart, use_container_width=True)

                    # OOS validation badge
                    oos_badge_class, oos_badge_text = get_validation_badge(metrics_oos)
                    st.markdown(f'<div style="text-align: center;"><span class="{oos_badge_class}">'
                                f'OOS: {oos_badge_text}</span></div>', unsafe_allow_html=True)
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

        # ---------------------------------------------------------------
        # Walk-Forward Tab
        # ---------------------------------------------------------------
        with tab_wf:
            if st.session_state.wf_results is not None:
                wf = st.session_state.wf_results

                # Summary metrics
                st.markdown('<div class="section-header">Walk-Forward Summary</div>', unsafe_allow_html=True)
                wf_cols = st.columns(6)
                with wf_cols[0]:
                    st.markdown(metric_card("Folds", f"{wf['n_folds']}", "neutral"), unsafe_allow_html=True)
                with wf_cols[1]:
                    st.markdown(metric_card("Profitable Folds",
                                             f"{wf['profitable_folds']}/{wf['n_folds']}",
                                             "positive" if wf['profitable_pct'] > 60 else "warning"),
                                unsafe_allow_html=True)
                with wf_cols[2]:
                    st.markdown(metric_card("Mean OOS Return",
                                             f"{wf['mean_oos_return']:+.2f}%",
                                             "positive" if wf['mean_oos_return'] > 0 else "negative"),
                                unsafe_allow_html=True)
                with wf_cols[3]:
                    st.markdown(metric_card("Median OOS Return",
                                             f"{wf['median_oos_return']:+.2f}%",
                                             "positive" if wf['median_oos_return'] > 0 else "negative"),
                                unsafe_allow_html=True)
                with wf_cols[4]:
                    st.markdown(metric_card("Mean OOS Sharpe",
                                             f"{wf['mean_oos_sharpe']:.3f}",
                                             "positive" if wf['mean_oos_sharpe'] > 0.5 else "neutral"),
                                unsafe_allow_html=True)
                with wf_cols[5]:
                    st.markdown(metric_card("Total Time",
                                             f"{wf['total_time']:.0f}s",
                                             "neutral"),
                                unsafe_allow_html=True)

                # Fold Visualizer
                st.markdown('<div class="section-header">Fold Visualizer</div>', unsafe_allow_html=True)
                fold_chart = create_walk_forward_fold_chart(
                    wf["folds"], wf["train_months"], wf["test_months"])
                st.plotly_chart(fold_chart, use_container_width=True)

                # Stitched OOS Equity Curve
                if wf["stitched_equity"] is not None:
                    st.markdown('<div class="section-header">Stitched OOS Equity Curve</div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(create_equity_chart(wf["stitched_equity"],
                                                        "Walk-Forward Stitched OOS Equity"),
                                   use_container_width=True)

                # Per-fold results table
                st.markdown('<div class="section-header">Per-Fold Results</div>', unsafe_allow_html=True)
                fold_table_data = []
                for f in wf["folds"]:
                    fold_table_data.append({
                        "Fold": f["fold_num"],
                        "Train Window": f"{f['train_start'][:10]} to {f['train_end'][:10]}",
                        "Test Window": f"{f['test_start'][:10]} to {f['test_end'][:10]}",
                        "Direction": f["direction"],
                        "Entry Indicators": ", ".join(f["entry_indicators"]),
                        "Train Sharpe": f"{f['train_sharpe']:.3f}",
                        "OOS Return": f"{f['oos_return']:+.2f}%",
                        "OOS Sharpe": f"{f['oos_sharpe']:.3f}",
                        "OOS Trades": f["oos_trades"],
                        "OOS Win Rate": f"{f['oos_winrate']:.1f}%",
                    })
                st.dataframe(pd.DataFrame(fold_table_data), use_container_width=True, hide_index=True)

                # OOS Return per fold bar chart
                fold_returns = [f["oos_return"] for f in wf["folds"]]
                fold_labels = [f"Fold {f['fold_num']}" for f in wf["folds"]]
                colors = ['#00e676' if r > 0 else '#ff5252' for r in fold_returns]
                fig_bar = go.Figure(go.Bar(
                    x=fold_labels, y=fold_returns,
                    marker_color=colors,
                    text=[f"{r:+.1f}%" for r in fold_returns],
                    textposition='outside',
                ))
                fig_bar.update_layout(
                    title=dict(text='OOS Return per Fold', font=dict(color='#e0e0e0', size=16)),
                    paper_bgcolor='#0a0e17',
                    plot_bgcolor='#0a0e17',
                    font=dict(color='#8892a4'),
                    xaxis=dict(gridcolor='#1a2744'),
                    yaxis=dict(gridcolor='#1a2744', title='Return (%)'),
                    margin=dict(l=60, r=20, t=40, b=40),
                    height=350,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            else:
                st.info("Enable Walk-Forward in the sidebar and click 'RUN WALK-FORWARD' to begin. "
                        "This trains the GA on rolling windows and tests blindly on unseen data - "
                        "the gold standard for strategy validation.")

        # ---------------------------------------------------------------
        # Strategy Details Tab
        # ---------------------------------------------------------------
        with tab_details:
            st.markdown('<div class="section-header">Strategy Parameters</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="param-box"><pre>{strategy_description(current_strat)}</pre></div>',
                        unsafe_allow_html=True)

            # Save to favorites button
            fav_col1, fav_col2 = st.columns([1, 3])
            with fav_col1:
                if st.button("Save to Favorites", use_container_width=True):
                    fav_entry = current_strat.to_dict()
                    fav_entry["saved_at"] = datetime.datetime.now().isoformat()
                    fav_entry["symbol"] = symbol if data_source == "Yahoo Finance" else "CSV"
                    fav_entry["interval"] = interval if data_source == "Yahoo Finance" else "auto"
                    fav_entry["description"] = strategy_description(current_strat)
                    st.session_state.favorites.append(fav_entry)
                    save_favorites(st.session_state.favorites)
                    st.success("Strategy saved to favorites!")
            with fav_col2:
                sym_for_export = symbol if data_source == "Yahoo Finance" else "BTC-USD"
                int_for_export = interval if data_source == "Yahoo Finance" else "1h"
                script = export_strategy_script(current_strat, sym_for_export, int_for_export)
                st.download_button(
                    label="Export as Python Script",
                    data=script,
                    file_name=f"strategy_{current_strat.direction}_{current_strat.fitness_sharpe:.2f}.py",
                    mime="text/x-python",
                    use_container_width=True,
                )

            # Evolution progress chart
            if gen_stats:
                st.markdown('<div class="section-header">Evolution Progress</div>', unsafe_allow_html=True)
                st.plotly_chart(create_generation_chart(gen_stats), use_container_width=True)

        # ---------------------------------------------------------------
        # All Strategies Tab
        # ---------------------------------------------------------------
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
                    "# Entry": len(s.entry_conditions),
                    "# Exit": len(s.exit_conditions),
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
        # Favorites Tab
        # ---------------------------------------------------------------
        with tab_fav:
            st.markdown('<div class="section-header">Saved Favorites</div>', unsafe_allow_html=True)

            if st.session_state.favorites:
                for i, fav in enumerate(st.session_state.favorites):
                    with st.expander(
                        f"#{i+1} | {fav.get('direction', 'N/A').upper()} | "
                        f"Sharpe: {fav.get('fitness_sharpe', 0):.3f} | "
                        f"Return: {fav.get('fitness_return', 0):.1f}% | "
                        f"{fav.get('symbol', '')} {fav.get('interval', '')} | "
                        f"Saved: {fav.get('saved_at', 'N/A')[:10]}"
                    ):
                        # Show strategy details
                        if "description" in fav:
                            st.markdown(f'<div class="param-box"><pre>{fav["description"]}</pre></div>',
                                        unsafe_allow_html=True)

                        # Metrics
                        fc1, fc2, fc3, fc4 = st.columns(4)
                        with fc1:
                            st.markdown(metric_card("Sharpe", f"{fav.get('fitness_sharpe', 0):.3f}", "neutral"),
                                        unsafe_allow_html=True)
                        with fc2:
                            st.markdown(metric_card("Return", f"{fav.get('fitness_return', 0):.1f}%",
                                                     "positive" if fav.get('fitness_return', 0) > 0 else "negative"),
                                        unsafe_allow_html=True)
                        with fc3:
                            st.markdown(metric_card("Win Rate", f"{fav.get('fitness_winrate', 0):.1f}%", "neutral"),
                                        unsafe_allow_html=True)
                        with fc4:
                            st.markdown(metric_card("Profit Factor", f"{fav.get('fitness_profit_factor', 0):.3f}", "neutral"),
                                        unsafe_allow_html=True)

                        # Action buttons
                        btn_col1, btn_col2, btn_col3 = st.columns(3)
                        with btn_col1:
                            # Export as script
                            try:
                                fav_strat = strategy_from_dict(fav)
                                fav_script = export_strategy_script(
                                    fav_strat,
                                    fav.get("symbol", "BTC-USD"),
                                    fav.get("interval", "1h"),
                                )
                                st.download_button(
                                    label="Export Script",
                                    data=fav_script,
                                    file_name=f"favorite_{i+1}_{fav.get('direction', 'long')}.py",
                                    mime="text/x-python",
                                    key=f"export_fav_{i}",
                                    use_container_width=True,
                                )
                            except Exception as e:
                                st.error(f"Export error: {e}")
                        with btn_col2:
                            if st.button("Load Strategy", key=f"load_fav_{i}", use_container_width=True):
                                try:
                                    loaded_strat = strategy_from_dict(fav)
                                    # Add to current results if we have results
                                    if st.session_state.results is not None:
                                        st.session_state.results["best_strategies"].insert(0, loaded_strat)
                                        st.session_state.selected_strategy_idx = 0
                                        st.success("Strategy loaded! Switch to other tabs to evaluate it.")
                                        st.rerun()
                                    else:
                                        st.warning("Load data and run evolution first, then load favorites.")
                                except Exception as e:
                                    st.error(f"Load error: {e}")
                        with btn_col3:
                            if st.button("Delete", key=f"del_fav_{i}", use_container_width=True):
                                st.session_state.favorites.pop(i)
                                save_favorites(st.session_state.favorites)
                                st.rerun()

                # Export/Import favorites JSON
                st.markdown("---")
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.download_button(
                        label="Export All Favorites (JSON)",
                        data=json.dumps(st.session_state.favorites, indent=2, default=str),
                        file_name="geneticai_favorites.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with exp_col2:
                    uploaded_favs = st.file_uploader("Import Favorites JSON", type=["json"],
                                                      key="import_favs")
                    if uploaded_favs is not None:
                        try:
                            imported = json.loads(uploaded_favs.read())
                            st.session_state.favorites.extend(imported)
                            save_favorites(st.session_state.favorites)
                            st.success(f"Imported {len(imported)} favorites!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Import error: {e}")
            else:
                st.info("No favorites saved yet. Go to Strategy Details tab and click 'Save to Favorites' "
                        "to save a strategy you like.")

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

    **New Features:**
    - **Strategy Complexity**: Control how many indicators (2-6) the GA uses for entry/exit conditions
    - **Walk-Forward Analysis**: Rolling train/test windows to expose curve-fitting
    - **Favorites**: Save, export, and reload your best strategies as standalone Python scripts

    The genetic algorithm will evolve combinations of 24 technical indicators including:

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
