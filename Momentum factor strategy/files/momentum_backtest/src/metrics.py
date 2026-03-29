"""
metrics.py
==========
Risk-adjusted performance metrics for strategy evaluation.

All metrics follow standard industry conventions used in quantitative finance.
Annual figures assume 252 trading days per year.

References
----------
Sharpe, W.F. (1994): The Sharpe Ratio. Journal of Portfolio Management.
Sortino & Price (1994): Performance Measurement in a Downside Risk Framework.
Calmar (1991): Calmar Ratio. Futures Magazine.
Lo, A.W. (2002): The Statistics of Sharpe Ratios. FAJ, 58(4).
"""

import numpy as np
import pandas as pd
from typing import Optional


TRADING_DAYS = 252
RISK_FREE_RATE = 0.02  # Annualised risk-free rate assumption


# ── Core metrics ──────────────────────────────────────────────────────────────

def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return."""
    n_years = len(returns) / TRADING_DAYS
    cumulative = (1 + returns).prod()
    return cumulative ** (1 / n_years) - 1


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
) -> float:
    """
    Annualised Sharpe ratio (excess return / annualised vol).

    Uses the daily risk-free rate approximation: rf_daily = (1+rf)^(1/252) - 1
    """
    rf_daily = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
    mar: float = 0.0,
) -> float:
    """
    Sortino ratio using downside deviation below MAR (minimum acceptable return).
    """
    rf_daily = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    downside = returns[returns < mar] - mar
    downside_vol = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)
    if downside_vol == 0:
        return np.nan
    return (excess.mean() * TRADING_DAYS) / downside_vol


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return drawdown.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    return (cum - running_max) / running_max


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio = annualised return / abs(max drawdown)."""
    ann_ret = annualised_return(returns)
    mdd = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd != 0 else np.nan


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (positive = loss)."""
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR) at given confidence level (positive = loss)."""
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= -var]
    return -tail.mean() if len(tail) > 0 else np.nan


def win_rate(returns: pd.Series) -> float:
    """Fraction of positive-return days."""
    return (returns > 0).mean()


def avg_win_loss(returns: pd.Series) -> float:
    """Ratio of average winning day to average losing day (absolute)."""
    wins = returns[returns > 0].mean()
    losses = abs(returns[returns < 0].mean())
    return wins / losses if losses != 0 else np.nan


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Information ratio: active return / tracking error.
    """
    active = returns - benchmark_returns.reindex(returns.index, fill_value=0.0)
    te = active.std() * np.sqrt(TRADING_DAYS)
    if te == 0:
        return np.nan
    return (active.mean() * TRADING_DAYS) / te


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Market beta via OLS regression of strategy on benchmark."""
    bench = benchmark_returns.reindex(returns.index, fill_value=0.0)
    cov = np.cov(returns, bench)
    return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = RISK_FREE_RATE,
) -> float:
    """Jensen's alpha (annualised)."""
    b = beta(returns, benchmark_returns)
    rf_daily = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    bench = benchmark_returns.reindex(returns.index, fill_value=0.0)
    ann_ret = annualised_return(returns)
    ann_bench = annualised_return(bench)
    return ann_ret - (risk_free + b * (ann_bench - risk_free))


def skewness(returns: pd.Series) -> float:
    """Return distribution skewness."""
    return float(returns.skew())


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (normal = 0)."""
    return float(returns.kurtosis())


# ── Summary table ─────────────────────────────────────────────────────────────

def compute_full_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_name: str = "Momentum",
    benchmark_name: str = "SPY",
) -> pd.DataFrame:
    """
    Compute the complete performance tearsheet metrics table.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns.
    benchmark_returns : pd.Series
        Daily benchmark returns.
    strategy_name : str
        Label for the strategy column.
    benchmark_name : str
        Label for the benchmark column.

    Returns
    -------
    pd.DataFrame : Metrics comparison table.
    """
    bench = benchmark_returns.reindex(strategy_returns.index, fill_value=0.0)

    def _metrics(ret: pd.Series, is_strategy: bool) -> dict:
        m = {
            "Annualised Return (%)": round(annualised_return(ret) * 100, 2),
            "Annualised Volatility (%)": round(annualised_volatility(ret) * 100, 2),
            "Sharpe Ratio": round(sharpe_ratio(ret), 3),
            "Sortino Ratio": round(sortino_ratio(ret), 3),
            "Calmar Ratio": round(calmar_ratio(ret), 3),
            "Max Drawdown (%)": round(max_drawdown(ret) * 100, 2),
            "VaR 95% (daily, %)": round(value_at_risk(ret) * 100, 3),
            "CVaR 95% (daily, %)": round(expected_shortfall(ret) * 100, 3),
            "Win Rate (%)": round(win_rate(ret) * 100, 2),
            "Avg Win / Avg Loss": round(avg_win_loss(ret), 3),
            "Skewness": round(skewness(ret), 3),
            "Excess Kurtosis": round(kurtosis(ret), 3),
        }
        if is_strategy:
            m["Beta"] = round(beta(ret, bench), 3)
            m["Alpha (ann., %)"] = round(alpha(ret, bench) * 100, 2)
            m["Information Ratio"] = round(information_ratio(ret, bench), 3)
        return m

    strat_m = _metrics(strategy_returns, is_strategy=True)
    bench_m = _metrics(bench, is_strategy=False)

    all_keys = list(strat_m.keys())
    strat_vals = [strat_m.get(k, "—") for k in all_keys]
    bench_vals = [bench_m.get(k, "—") for k in all_keys]

    return pd.DataFrame(
        {strategy_name: strat_vals, benchmark_name: bench_vals},
        index=all_keys,
    )


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free: float = RISK_FREE_RATE,
) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    rf_daily = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    return (
        excess.rolling(window).mean()
        / excess.rolling(window).std()
        * np.sqrt(TRADING_DAYS)
    )


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Pivot table of monthly returns (rows = year, cols = month).
    """
    monthly = (1 + returns).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")
    table = monthly.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="ret")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ][: pivot.columns.max()]
    # Annual total
    pivot["Annual"] = (1 + monthly.groupby(monthly.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )).values - 1
    return pivot * 100   # percentages
