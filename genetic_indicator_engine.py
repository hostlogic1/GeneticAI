"""
Genetic Indicator Combination Engine
=====================================
Evolves trading strategies by combining well-known technical indicators
(RSI, SMA, MACD, Stochastic, ADX, CCI, etc.) using a genetic algorithm.

Each chromosome encodes a complete trading strategy:
  - Which indicators to use for entry/exit
  - Indicator parameters (periods, thresholds)
  - Logical combination (AND conditions)
  - Stop-loss and take-profit levels

Backtesting is done with vectorbt for speed.
Uses the 'ta' library for indicator computation.
"""

import copy
import random
import time
import math
import warnings
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import ta as ta_lib
import vectorbt as vbt
import dill

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

INITIAL_CASH = 10_000
COMMISSION_PCT = 0.001  # 0.1% per trade (typical crypto)

# ---------------------------------------------------------------------------
# 1. Indicator Definitions (using 'ta' library)
# ---------------------------------------------------------------------------
INDICATOR_REGISTRY = {}


def register_indicator(name, param_options, compute_fn, description=""):
    INDICATOR_REGISTRY[name] = {
        "param_options": param_options,
        "compute": compute_fn,
        "description": description,
    }


# Helper: ensure Series output
def _s(val, index):
    if isinstance(val, pd.Series):
        return val
    return pd.Series(val, index=index)


# --- RSI ---
register_indicator(
    "RSI",
    {"period": [7, 9, 14, 21, 28]},
    lambda df, p: {"RSI": ta_lib.momentum.RSIIndicator(df["Close"], window=p["period"]).rsi()},
    "Relative Strength Index",
)

# --- SMA crossover ---
register_indicator(
    "SMA",
    {"fast": [5, 10, 20], "slow": [50, 100, 200]},
    lambda df, p: {
        "SMA_fast": ta_lib.trend.SMAIndicator(df["Close"], window=p["fast"]).sma_indicator(),
        "SMA_slow": ta_lib.trend.SMAIndicator(df["Close"], window=p["slow"]).sma_indicator(),
    },
    "Simple Moving Average crossover",
)

# --- EMA crossover ---
register_indicator(
    "EMA",
    {"fast": [5, 8, 12, 21], "slow": [26, 50, 100, 200]},
    lambda df, p: {
        "EMA_fast": ta_lib.trend.EMAIndicator(df["Close"], window=p["fast"]).ema_indicator(),
        "EMA_slow": ta_lib.trend.EMAIndicator(df["Close"], window=p["slow"]).ema_indicator(),
    },
    "Exponential Moving Average crossover",
)

# --- MACD ---
register_indicator(
    "MACD",
    {"fast": [8, 12], "slow": [21, 26], "signal": [7, 9]},
    lambda df, p: {
        "MACD_line": ta_lib.trend.MACD(df["Close"], window_fast=p["fast"], window_slow=p["slow"], window_sign=p["signal"]).macd(),
        "MACD_signal": ta_lib.trend.MACD(df["Close"], window_fast=p["fast"], window_slow=p["slow"], window_sign=p["signal"]).macd_signal(),
        "MACD_hist": ta_lib.trend.MACD(df["Close"], window_fast=p["fast"], window_slow=p["slow"], window_sign=p["signal"]).macd_diff(),
    },
    "MACD line, signal, histogram",
)

# --- Stochastic ---
register_indicator(
    "Stochastic",
    {"k": [5, 9, 14], "d": [3, 5], "smooth_k": [3, 5]},
    lambda df, p: {
        "STOCH_K": ta_lib.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=p["k"], smooth_window=p["d"]).stoch(),
        "STOCH_D": ta_lib.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=p["k"], smooth_window=p["d"]).stoch_signal(),
    },
    "Stochastic Oscillator K and D",
)

# --- ADX ---
register_indicator(
    "ADX",
    {"period": [7, 14, 21]},
    lambda df, p: {
        "ADX": ta_lib.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=p["period"]).adx(),
        "DI_plus": ta_lib.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=p["period"]).adx_pos(),
        "DI_minus": ta_lib.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=p["period"]).adx_neg(),
    },
    "Average Directional Index",
)

# --- CCI ---
register_indicator(
    "CCI",
    {"period": [14, 20, 50]},
    lambda df, p: {"CCI": ta_lib.trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=p["period"]).cci()},
    "Commodity Channel Index",
)

# --- ROC ---
register_indicator(
    "ROC",
    {"period": [9, 12, 14, 21]},
    lambda df, p: {"ROC": ta_lib.momentum.ROCIndicator(df["Close"], window=p["period"]).roc()},
    "Rate of Change",
)

# --- MFI ---
register_indicator(
    "MFI",
    {"period": [10, 14, 20]},
    lambda df, p: {"MFI": ta_lib.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=p["period"]).money_flow_index()},
    "Money Flow Index",
)

# --- Bollinger Bands ---
register_indicator(
    "BollingerBands",
    {"period": [10, 20, 30], "std": [1.5, 2.0, 2.5]},
    lambda df, p: {
        "BB_upper": ta_lib.volatility.BollingerBands(df["Close"], window=p["period"], window_dev=p["std"]).bollinger_hband(),
        "BB_mid": ta_lib.volatility.BollingerBands(df["Close"], window=p["period"], window_dev=p["std"]).bollinger_mavg(),
        "BB_lower": ta_lib.volatility.BollingerBands(df["Close"], window=p["period"], window_dev=p["std"]).bollinger_lband(),
        "BB_pctb": ta_lib.volatility.BollingerBands(df["Close"], window=p["period"], window_dev=p["std"]).bollinger_pband(),
    },
    "Bollinger Bands",
)

# --- Keltner Channel ---
register_indicator(
    "KeltnerChannel",
    {"period": [10, 20], "atr_mult": [1.0, 1.5, 2.0]},
    lambda df, p: {
        "KC_upper": ta_lib.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"], window=p["period"], multiplier=p["atr_mult"]).keltner_channel_hband(),
        "KC_basis": ta_lib.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"], window=p["period"], multiplier=p["atr_mult"]).keltner_channel_mband(),
        "KC_lower": ta_lib.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"], window=p["period"], multiplier=p["atr_mult"]).keltner_channel_lband(),
    },
    "Keltner Channel",
)

# --- KAMA ---
register_indicator(
    "KAMA",
    {"period": [10, 20, 30]},
    lambda df, p: {"KAMA": ta_lib.momentum.KAMAIndicator(df["Close"], window=p["period"]).kama()},
    "Kaufman Adaptive Moving Average",
)

# --- Williams %R ---
register_indicator(
    "WilliamsR",
    {"period": [7, 14, 21]},
    lambda df, p: {"WILLR": ta_lib.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=p["period"]).williams_r()},
    "Williams %R",
)

# --- ATR ---
register_indicator(
    "ATR",
    {"period": [7, 14, 21]},
    lambda df, p: {"ATR": ta_lib.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=p["period"]).average_true_range()},
    "Average True Range",
)

# --- OBV ---
register_indicator(
    "OBV",
    {},
    lambda df, p: {"OBV": ta_lib.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()},
    "On Balance Volume",
)

# --- Aroon ---
register_indicator(
    "Aroon",
    {"period": [14, 25]},
    lambda df, p: {
        "AROON_up": ta_lib.trend.AroonIndicator(df["Close"], window=p["period"]).aroon_up(),
        "AROON_down": ta_lib.trend.AroonIndicator(df["Close"], window=p["period"]).aroon_down(),
    },
    "Aroon Oscillator",
)

# --- CMF (Chaikin Money Flow) ---
register_indicator(
    "CMF",
    {"period": [10, 20, 21]},
    lambda df, p: {"CMF": ta_lib.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=p["period"]).chaikin_money_flow()},
    "Chaikin Money Flow",
)

# --- TSI (True Strength Index) ---
register_indicator(
    "TSI",
    {"fast": [13, 25], "slow": [7, 13]},
    lambda df, p: {"TSI": ta_lib.momentum.TSIIndicator(df["Close"], window_slow=p["fast"], window_fast=p["slow"]).tsi()},
    "True Strength Index",
)

# --- PPO ---
register_indicator(
    "PPO",
    {"fast": [12, 26], "slow": [26, 50]},
    lambda df, p: {"PPO": ta_lib.momentum.PercentagePriceOscillator(df["Close"], window_fast=p["fast"], window_slow=p["slow"]).ppo()},
    "Percentage Price Oscillator",
)

# --- Ultimate Oscillator ---
register_indicator(
    "UltimateOscillator",
    {"fast": [7], "medium": [14], "slow": [28]},
    lambda df, p: {"UO": ta_lib.momentum.UltimateOscillator(df["High"], df["Low"], df["Close"], window1=p["fast"], window2=p["medium"], window3=p["slow"]).ultimate_oscillator()},
    "Ultimate Oscillator",
)

# --- Ichimoku ---
register_indicator(
    "Ichimoku",
    {"tenkan": [9], "kijun": [26]},
    lambda df, p: {
        "ICH_conv": ta_lib.trend.IchimokuIndicator(df["High"], df["Low"], window1=p["tenkan"], window2=p["kijun"]).ichimoku_conversion_line(),
        "ICH_base": ta_lib.trend.IchimokuIndicator(df["High"], df["Low"], window1=p["tenkan"], window2=p["kijun"]).ichimoku_base_line(),
    },
    "Ichimoku Cloud",
)

# --- DPO ---
register_indicator(
    "DPO",
    {"period": [14, 20, 30]},
    lambda df, p: {"DPO": ta_lib.trend.DPOIndicator(df["Close"], window=p["period"]).dpo()},
    "Detrended Price Oscillator",
)

# --- VWAP ---
register_indicator(
    "VWAP",
    {},
    lambda df, p: {"VWAP": ta_lib.volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()},
    "Volume Weighted Average Price",
)

# --- Awesome Oscillator ---
register_indicator(
    "AwesomeOscillator",
    {"fast": [5], "slow": [34]},
    lambda df, p: {"AO": ta_lib.momentum.AwesomeOscillatorIndicator(df["High"], df["Low"], window1=p["fast"], window2=p["slow"]).awesome_oscillator()},
    "Awesome Oscillator",
)

INDICATOR_NAMES = list(INDICATOR_REGISTRY.keys())

# ---------------------------------------------------------------------------
# 2. Condition Types
# ---------------------------------------------------------------------------
CONDITION_TYPES = [
    "crosses_above",
    "crosses_below",
    "is_above",
    "is_below",
    "rising",
    "falling",
]


# ---------------------------------------------------------------------------
# 3. Strategy Gene Encoding
# ---------------------------------------------------------------------------
@dataclass
class IndicatorCondition:
    indicator_name: str
    params: Dict
    output_key: str
    condition_type: str
    threshold: float
    compare_indicator: Optional[str] = None
    compare_output_key: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class TradingStrategy:
    entry_conditions: List[IndicatorCondition]
    exit_conditions: List[IndicatorCondition]
    stop_loss_pct: float
    take_profit_pct: float
    direction: str = "long"

    fitness_sharpe: float = 0.0
    fitness_return: float = 0.0
    fitness_trades: int = 0
    fitness_winrate: float = 0.0
    fitness_max_dd: float = 0.0
    fitness_profit_factor: float = 0.0
    fitness_sortino: float = 0.0
    fitness_calmar: float = 0.0
    fitness_score: float = -999.0

    def to_dict(self):
        return {
            "entry_conditions": [c.to_dict() for c in self.entry_conditions],
            "exit_conditions": [c.to_dict() for c in self.exit_conditions],
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "direction": self.direction,
            "fitness_sharpe": self.fitness_sharpe,
            "fitness_return": self.fitness_return,
            "fitness_trades": self.fitness_trades,
            "fitness_winrate": self.fitness_winrate,
            "fitness_max_dd": self.fitness_max_dd,
            "fitness_profit_factor": self.fitness_profit_factor,
            "fitness_sortino": self.fitness_sortino,
            "fitness_calmar": self.fitness_calmar,
            "fitness_score": self.fitness_score,
        }


# ---------------------------------------------------------------------------
# 4. Random Strategy Generation
# ---------------------------------------------------------------------------
def random_params(indicator_name: str) -> Dict:
    info = INDICATOR_REGISTRY[indicator_name]
    params = {}
    for param_name, options in info["param_options"].items():
        params[param_name] = random.choice(options)
    return params


def random_output_key(indicator_name: str, params: Dict, df: pd.DataFrame = None) -> str:
    info = INDICATOR_REGISTRY[indicator_name]
    try:
        if df is not None and len(df) > 250:
            sample = df.iloc[:250].copy()
        else:
            sample = df
        outputs = info["compute"](sample, params)
        keys = [k for k, v in outputs.items() if v is not None and not v.isna().all()]
        if keys:
            return random.choice(keys)
    except Exception:
        pass
    return indicator_name


def get_threshold_for_indicator(output_key: str) -> float:
    key = output_key.upper()
    if "RSI" in key or "MFI" in key or "WILLR" in key or "STOCH" in key:
        return random.choice([20, 25, 30, 50, 70, 75, 80])
    elif "CCI" in key:
        return random.choice([-200, -100, -50, 0, 50, 100, 200])
    elif "ADX" in key or "DI_" in key:
        return random.choice([15, 20, 25, 30, 40])
    elif "ROC" in key or "PPO" in key or "DPO" in key:
        return random.choice([-5, -2, -1, 0, 1, 2, 5])
    elif "CMF" in key or "TSI" in key:
        return random.choice([-0.1, -0.05, 0, 0.05, 0.1])
    elif "AROON" in key:
        return random.choice([30, 50, 70, 80])
    elif "BB_pctb" in key:
        return random.choice([0.0, 0.2, 0.5, 0.8, 1.0])
    elif "MACD_hist" in key:
        return 0.0
    elif "UO" in key:
        return random.choice([30, 40, 50, 60, 70])
    elif "AO" in key:
        return 0.0
    else:
        return 0.0


def random_condition(df: pd.DataFrame) -> IndicatorCondition:
    ind_name = random.choice(INDICATOR_NAMES)
    params = random_params(ind_name)
    output_key = random_output_key(ind_name, params, df)
    cond_type = random.choice(CONDITION_TYPES)
    threshold = get_threshold_for_indicator(output_key)

    compare_ind = None
    compare_key = None
    if cond_type in ("crosses_above", "crosses_below") and random.random() < 0.3:
        compare_ind = random.choice(INDICATOR_NAMES)
        compare_params = random_params(compare_ind)
        compare_key = random_output_key(compare_ind, compare_params, df)

    return IndicatorCondition(
        indicator_name=ind_name,
        params=params,
        output_key=output_key,
        condition_type=cond_type,
        threshold=threshold,
        compare_indicator=compare_ind,
        compare_output_key=compare_key,
    )


def random_strategy(df: pd.DataFrame) -> TradingStrategy:
    n_entry = random.randint(1, 3)
    n_exit = random.randint(1, 2)
    return TradingStrategy(
        entry_conditions=[random_condition(df) for _ in range(n_entry)],
        exit_conditions=[random_condition(df) for _ in range(n_exit)],
        stop_loss_pct=round(random.uniform(0.01, 0.10), 3),
        take_profit_pct=round(random.uniform(0.02, 0.20), 3),
        direction=random.choice(["long", "short", "both"]),
    )


# ---------------------------------------------------------------------------
# 5. Indicator Computation Cache
# ---------------------------------------------------------------------------
class IndicatorCache:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._cache: Dict[str, pd.Series] = {}

    def get(self, indicator_name: str, params: Dict, output_key: str) -> pd.Series:
        cache_key = f"{indicator_name}_{json.dumps(params, sort_keys=True)}_{output_key}"
        if cache_key not in self._cache:
            try:
                info = INDICATOR_REGISTRY[indicator_name]
                outputs = info["compute"](self.df, params)
                for k, v in outputs.items():
                    ck = f"{indicator_name}_{json.dumps(params, sort_keys=True)}_{k}"
                    if v is not None:
                        self._cache[ck] = v
            except Exception:
                self._cache[cache_key] = pd.Series(np.nan, index=self.df.index)
        return self._cache.get(cache_key, pd.Series(np.nan, index=self.df.index))

    def clear(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# 6. Signal Generation from Strategy
# ---------------------------------------------------------------------------
def evaluate_condition(cond: IndicatorCondition, cache: IndicatorCache) -> pd.Series:
    series = cache.get(cond.indicator_name, cond.params, cond.output_key)

    if cond.condition_type == "is_above":
        return series > cond.threshold
    elif cond.condition_type == "is_below":
        return series < cond.threshold
    elif cond.condition_type == "rising":
        return series > series.shift(1)
    elif cond.condition_type == "falling":
        return series < series.shift(1)
    elif cond.condition_type == "crosses_above":
        if cond.compare_indicator and cond.compare_output_key:
            other = cache.get(cond.compare_indicator, cond.params, cond.compare_output_key)
        else:
            other = pd.Series(cond.threshold, index=series.index)
        return (series > other) & (series.shift(1) <= other.shift(1))
    elif cond.condition_type == "crosses_below":
        if cond.compare_indicator and cond.compare_output_key:
            other = cache.get(cond.compare_indicator, cond.params, cond.compare_output_key)
        else:
            other = pd.Series(cond.threshold, index=series.index)
        return (series < other) & (series.shift(1) >= other.shift(1))
    else:
        return pd.Series(False, index=series.index)


def generate_signals(strategy: TradingStrategy, cache: IndicatorCache) -> Tuple[pd.Series, pd.Series]:
    idx = cache.df.index

    # Entry: AND of all entry conditions
    entry_signals = pd.Series(True, index=idx)
    for cond in strategy.entry_conditions:
        sig = evaluate_condition(cond, cache)
        entry_signals = entry_signals & sig.reindex(idx, fill_value=False)

    # Exit: OR of any exit condition
    exit_signals = pd.Series(False, index=idx)
    for cond in strategy.exit_conditions:
        sig = evaluate_condition(cond, cache)
        exit_signals = exit_signals | sig.reindex(idx, fill_value=False)

    entry_signals = entry_signals.fillna(False).astype(bool)
    exit_signals = exit_signals.fillna(False).astype(bool)

    return entry_signals, exit_signals


# ---------------------------------------------------------------------------
# 7. Backtesting with vectorbt
# ---------------------------------------------------------------------------
def backtest_strategy(
    strategy: TradingStrategy,
    df: pd.DataFrame,
    cache: IndicatorCache = None,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION_PCT,
) -> Dict:
    if cache is None:
        cache = IndicatorCache(df)

    try:
        entries, exits = generate_signals(strategy, cache)

        if entries.sum() == 0:
            return _empty_metrics()

        price = df["Close"]

        if strategy.direction == "long":
            pf = vbt.Portfolio.from_signals(
                price, entries=entries, exits=exits,
                init_cash=initial_cash, fees=commission,
                sl_stop=strategy.stop_loss_pct, tp_stop=strategy.take_profit_pct,
                freq="1h",
            )
        elif strategy.direction == "short":
            pf = vbt.Portfolio.from_signals(
                price, short_entries=entries, short_exits=exits,
                init_cash=initial_cash, fees=commission,
                sl_stop=strategy.stop_loss_pct, tp_stop=strategy.take_profit_pct,
                freq="1h",
            )
        else:  # both
            pf = vbt.Portfolio.from_signals(
                price, entries=entries, exits=exits,
                short_entries=exits, short_exits=entries,
                init_cash=initial_cash, fees=commission,
                sl_stop=strategy.stop_loss_pct, tp_stop=strategy.take_profit_pct,
                freq="1h",
            )

        stats = pf.stats()
        total_trades = int(stats.get("Total Trades", 0))

        if total_trades < 5:
            return _empty_metrics()

        total_return = float(stats.get("Total Return [%]", 0))
        sharpe = float(stats.get("Sharpe Ratio", 0))
        sortino = float(stats.get("Sortino Ratio", 0))
        calmar = float(stats.get("Calmar Ratio", 0))
        max_dd = float(stats.get("Max Drawdown [%]", 0))
        win_rate = float(stats.get("Win Rate [%]", 0))
        profit_factor = float(stats.get("Profit Factor", 0))

        for name in ["sharpe", "sortino", "calmar", "profit_factor"]:
            val = locals()[name]
            if not np.isfinite(val):
                locals()[name]  # can't reassign via locals, handle below

        sharpe = 0.0 if not np.isfinite(sharpe) else sharpe
        sortino = 0.0 if not np.isfinite(sortino) else sortino
        calmar = 0.0 if not np.isfinite(calmar) else calmar
        profit_factor = 0.0 if not np.isfinite(profit_factor) else profit_factor

        equity_curve = pf.value()
        try:
            trades_records = pf.trades.records_readable
        except Exception:
            trades_records = pd.DataFrame()

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "equity_curve": equity_curve,
            "trades": trades_records,
            "portfolio": pf,
            "stats": stats,
        }

    except Exception:
        return _empty_metrics()


def _empty_metrics():
    return {
        "total_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "total_trades": 0,
        "equity_curve": None,
        "trades": pd.DataFrame(),
        "portfolio": None,
        "stats": None,
    }


def compute_fitness_score(metrics: Dict) -> float:
    if metrics["total_trades"] < 5:
        return -999.0

    score = (
        metrics["sharpe"] * 2.0
        + metrics["sortino"] * 1.5
        + (metrics["total_return"] / 100.0) * 1.0
        + (metrics["win_rate"] / 100.0) * 0.5
        + min(metrics["profit_factor"], 5.0) * 1.0
        - abs(metrics["max_drawdown"]) / 100.0 * 2.0
    )

    if metrics["total_trades"] < 20:
        score *= 0.5

    return score


# ---------------------------------------------------------------------------
# 8. Genetic Operators
# ---------------------------------------------------------------------------
def crossover(strat1: TradingStrategy, strat2: TradingStrategy) -> Tuple[TradingStrategy, TradingStrategy]:
    child1 = copy.deepcopy(strat1)
    child2 = copy.deepcopy(strat2)

    if random.random() < 0.5 and child1.entry_conditions and child2.entry_conditions:
        idx1 = random.randrange(len(child1.entry_conditions))
        idx2 = random.randrange(len(child2.entry_conditions))
        child1.entry_conditions[idx1], child2.entry_conditions[idx2] = \
            child2.entry_conditions[idx2], child1.entry_conditions[idx1]

    if random.random() < 0.5 and child1.exit_conditions and child2.exit_conditions:
        idx1 = random.randrange(len(child1.exit_conditions))
        idx2 = random.randrange(len(child2.exit_conditions))
        child1.exit_conditions[idx1], child2.exit_conditions[idx2] = \
            child2.exit_conditions[idx2], child1.exit_conditions[idx1]

    if random.random() < 0.3:
        child1.stop_loss_pct, child2.stop_loss_pct = child2.stop_loss_pct, child1.stop_loss_pct
    if random.random() < 0.3:
        child1.take_profit_pct, child2.take_profit_pct = child2.take_profit_pct, child1.take_profit_pct

    return child1, child2


def mutate(strat: TradingStrategy, df: pd.DataFrame) -> TradingStrategy:
    child = copy.deepcopy(strat)

    mutation_type = random.choice([
        "replace_entry_cond", "replace_exit_cond",
        "tweak_threshold", "tweak_sl_tp",
        "add_condition", "remove_condition",
        "change_params", "flip_direction",
    ])

    if mutation_type == "replace_entry_cond" and child.entry_conditions:
        idx = random.randrange(len(child.entry_conditions))
        child.entry_conditions[idx] = random_condition(df)

    elif mutation_type == "replace_exit_cond" and child.exit_conditions:
        idx = random.randrange(len(child.exit_conditions))
        child.exit_conditions[idx] = random_condition(df)

    elif mutation_type == "tweak_threshold":
        all_conds = child.entry_conditions + child.exit_conditions
        if all_conds:
            cond = random.choice(all_conds)
            cond.threshold *= random.uniform(0.8, 1.2)

    elif mutation_type == "tweak_sl_tp":
        child.stop_loss_pct = max(0.005, min(0.15, child.stop_loss_pct * random.uniform(0.7, 1.3)))
        child.take_profit_pct = max(0.01, min(0.30, child.take_profit_pct * random.uniform(0.7, 1.3)))

    elif mutation_type == "add_condition":
        if len(child.entry_conditions) < 4:
            child.entry_conditions.append(random_condition(df))

    elif mutation_type == "remove_condition":
        if len(child.entry_conditions) > 1:
            child.entry_conditions.pop(random.randrange(len(child.entry_conditions)))

    elif mutation_type == "change_params":
        all_conds = child.entry_conditions + child.exit_conditions
        if all_conds:
            cond = random.choice(all_conds)
            cond.params = random_params(cond.indicator_name)

    elif mutation_type == "flip_direction":
        child.direction = random.choice(["long", "short", "both"])

    return child


# ---------------------------------------------------------------------------
# 9. Genetic Algorithm Evolution
# ---------------------------------------------------------------------------
def run_evolution(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame = None,
    pop_size: int = 100,
    n_generations: int = 20,
    p_crossover: float = 0.7,
    p_mutation: float = 0.3,
    tournament_size: int = 3,
    elite_size: int = 5,
    progress_callback=None,
) -> Dict:
    start_time = time.time()
    cache_train = IndicatorCache(df_train)
    cache_val = IndicatorCache(df_val) if df_val is not None else None

    print(f"Creating initial population of {pop_size} strategies...")
    population = [random_strategy(df_train) for _ in range(pop_size)]

    generation_stats = []
    all_evaluated = []
    best_score_ever = -999.0
    best_strategy_ever = None

    for gen in range(n_generations):
        gen_start = time.time()

        for i, strat in enumerate(population):
            metrics = backtest_strategy(strat, df_train, cache_train)
            score = compute_fitness_score(metrics)

            strat.fitness_score = score
            strat.fitness_sharpe = metrics["sharpe"]
            strat.fitness_return = metrics["total_return"]
            strat.fitness_trades = metrics["total_trades"]
            strat.fitness_winrate = metrics["win_rate"]
            strat.fitness_max_dd = metrics["max_drawdown"]
            strat.fitness_profit_factor = metrics["profit_factor"]
            strat.fitness_sortino = metrics["sortino"]
            strat.fitness_calmar = metrics["calmar"]

            all_evaluated.append({
                "generation": gen + 1,
                "index": i,
                "score": score,
                "sharpe": metrics["sharpe"],
                "total_return": metrics["total_return"],
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "max_dd": metrics["max_drawdown"],
                "profit_factor": metrics["profit_factor"],
                "sortino": metrics["sortino"],
            })

        population.sort(key=lambda s: s.fitness_score, reverse=True)

        if population[0].fitness_score > best_score_ever:
            best_score_ever = population[0].fitness_score
            best_strategy_ever = copy.deepcopy(population[0])

        scores = [s.fitness_score for s in population]
        valid_scores = [s for s in scores if s > -999]
        gen_time = time.time() - gen_start

        stat = {
            "generation": gen + 1,
            "best_score": max(scores),
            "avg_score": np.mean(valid_scores) if valid_scores else 0,
            "median_score": np.median(valid_scores) if valid_scores else 0,
            "best_sharpe": population[0].fitness_sharpe,
            "best_return": population[0].fitness_return,
            "best_trades": population[0].fitness_trades,
            "best_winrate": population[0].fitness_winrate,
            "best_max_dd": population[0].fitness_max_dd,
            "best_pf": population[0].fitness_profit_factor,
            "gen_time": gen_time,
            "total_time": time.time() - start_time,
        }
        generation_stats.append(stat)

        print(f"Gen {gen+1:3d}/{n_generations} | Best: {stat['best_score']:.3f} | "
              f"Sharpe: {stat['best_sharpe']:.3f} | Return: {stat['best_return']:.1f}% | "
              f"Trades: {stat['best_trades']} | Time: {gen_time:.1f}s")

        if progress_callback:
            progress_callback(gen + 1, n_generations, stat)

        # Selection + new generation
        if gen < n_generations - 1:
            new_population = list(map(copy.deepcopy, population[:elite_size]))

            while len(new_population) < pop_size:
                t1 = random.sample(population, min(tournament_size, len(population)))
                parent1 = max(t1, key=lambda s: s.fitness_score)
                t2 = random.sample(population, min(tournament_size, len(population)))
                parent2 = max(t2, key=lambda s: s.fitness_score)

                if random.random() < p_crossover:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

                if random.random() < p_mutation:
                    child1 = mutate(child1, df_train)
                if random.random() < p_mutation:
                    child2 = mutate(child2, df_train)

                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)

            population = new_population

    top_strategies = sorted(population, key=lambda s: s.fitness_score, reverse=True)[:20]

    if df_val is not None:
        print("\nValidating top strategies on out-of-sample data...")
        for strat in top_strategies:
            val_metrics = backtest_strategy(strat, df_val, cache_val)
            strat.val_return = val_metrics["total_return"]
            strat.val_sharpe = val_metrics["sharpe"]
            strat.val_score = compute_fitness_score(val_metrics)

    total_time = time.time() - start_time
    print(f"\nEvolution complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best ever score: {best_score_ever:.3f}")

    return {
        "best_strategies": top_strategies,
        "best_strategy": best_strategy_ever,
        "generation_stats": generation_stats,
        "all_evaluated": pd.DataFrame(all_evaluated),
        "total_time": total_time,
    }


# ---------------------------------------------------------------------------
# 10. Data Loading Utilities
# ---------------------------------------------------------------------------
YFINANCE_LIMITS = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "1h": "2y", "1d": "10y", "1wk": "10y", "1mo": "10y",
}

def load_data_yfinance(symbol: str = "SOL-USD", interval: str = "1h",
                       period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    # Clamp period to yfinance max for this interval
    max_period = YFINANCE_LIMITS.get(interval, "2y")
    period_order = ["7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
    if period in period_order and max_period in period_order:
        if period_order.index(period) > period_order.index(max_period):
            print(f"Warning: {interval} data only supports up to {max_period}. Clamping period.")
            period = max_period

    print(f"Downloading {symbol} {interval} data ({period})...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for {symbol} ({interval}, {period}). "
            f"Check the symbol or try a different interval/period. "
            f"Max period for {interval} is {max_period}."
        )

    available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if len(available) < 4:
        raise ValueError(f"Missing OHLC columns. Got: {list(df.columns)}")

    df = df[available].dropna()
    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} bars returned for {symbol} ({interval}, {period}). "
            f"Need at least 50. Try a longer period or different interval."
        )

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def load_data_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col.lower().replace(" ", "") in ("date", "datetime", "time", "timestamp", "gmttime"):
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    col_map = {}
    for c in df.columns:
        for r in ["Open", "High", "Low", "Close", "Volume"]:
            if c.lower() == r.lower():
                col_map[c] = r
    df = df.rename(columns=col_map)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna()


def split_data(df: pd.DataFrame, train_pct: float = 0.6,
               val_pct: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"Train: {len(train)} bars ({train.index[0]} to {train.index[-1]})")
    print(f"Val:   {len(val)} bars ({val.index[0]} to {val.index[-1]})")
    print(f"Test:  {len(test)} bars ({test.index[0]} to {test.index[-1]})")

    return train, val, test


# ---------------------------------------------------------------------------
# 11. Bootstrap Validation
# ---------------------------------------------------------------------------
def bootstrap_validation(strategy: TradingStrategy, df: pd.DataFrame,
                         n_samples: int = 100, sample_pct: float = 0.5,
                         threshold: float = 0.0) -> Dict:
    returns = []
    sharpes = []
    n_bars = int(len(df) * sample_pct)

    for _ in range(n_samples):
        start_idx = random.randint(0, len(df) - n_bars - 1)
        sample = df.iloc[start_idx:start_idx + n_bars].copy()
        cache = IndicatorCache(sample)
        metrics = backtest_strategy(strategy, sample, cache)
        returns.append(metrics["total_return"])
        sharpes.append(metrics["sharpe"])

    pass_count = sum(1 for r in returns if r > threshold)
    pass_rate = pass_count / n_samples * 100

    return {
        "pass_rate": pass_rate,
        "returns": returns,
        "sharpes": sharpes,
        "mean_return": np.mean(returns),
        "median_return": np.median(returns),
        "std_return": np.std(returns),
        "mean_sharpe": np.mean(sharpes),
    }


# ---------------------------------------------------------------------------
# 12. Save/Load Results
# ---------------------------------------------------------------------------
def save_results(results: Dict, path: str = "evolution_results.dill"):
    # Strip non-serializable portfolio objects before saving
    save_copy = copy.deepcopy(results)
    for strat in save_copy.get("best_strategies", []):
        # Only keep serializable attributes
        pass
    with open(path, "wb") as f:
        dill.dump(save_copy, f)
    print(f"Results saved to {path}")


def load_results(path: str = "evolution_results.dill") -> Dict:
    with open(path, "rb") as f:
        return dill.load(f)


# ---------------------------------------------------------------------------
# 13. Main Entry Point
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  GENETIC INDICATOR COMBINATION ENGINE")
    print("=" * 70)

    df = load_data_yfinance("SOL-USD", interval="1h", period="2y")
    train, val, test = split_data(df)

    results = run_evolution(
        df_train=train,
        df_val=val,
        pop_size=100,
        n_generations=25,
        p_crossover=0.7,
        p_mutation=0.3,
    )

    best = results["best_strategy"]
    print("\n" + "=" * 70)
    print("  BEST STRATEGY - TEST SET RESULTS")
    print("=" * 70)

    cache_test = IndicatorCache(test)
    test_metrics = backtest_strategy(best, test, cache_test)
    print(f"  Return:        {test_metrics['total_return']:.2f}%")
    print(f"  Sharpe:        {test_metrics['sharpe']:.3f}")
    print(f"  Sortino:       {test_metrics['sortino']:.3f}")
    print(f"  Max Drawdown:  {test_metrics['max_drawdown']:.2f}%")
    print(f"  Win Rate:      {test_metrics['win_rate']:.1f}%")
    print(f"  Profit Factor: {test_metrics['profit_factor']:.3f}")
    print(f"  Total Trades:  {test_metrics['total_trades']}")

    print("\nRunning bootstrap validation...")
    bootstrap = bootstrap_validation(best, df, n_samples=50)
    print(f"  Bootstrap Pass Rate: {bootstrap['pass_rate']:.1f}%")

    save_results(results)
    print("\nDone!")


if __name__ == "__main__":
    main()
