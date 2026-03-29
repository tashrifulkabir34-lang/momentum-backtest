"""
portfolio.py
============
Portfolio construction and monthly rebalancing logic.

Strategy overview
-----------------
On the last trading day of each month:
  1. Rank all stocks by the 12-1 month momentum signal.
  2. Select the top `top_pct`% as LONG positions (equal-weighted).
  3. Optionally select the bottom `bottom_pct`% as SHORT positions.
  4. Compute transaction costs on turnover.
  5. Hold until next rebalance.

Design decisions
----------------
- Equal weighting within each leg avoids concentration risk and is standard
  in academic momentum research (Jegadeesh & Titman 1993).
- Monthly rebalancing balances signal decay vs. transaction costs.
- A minimum universe size guard (MIN_VALID_STOCKS) prevents degenerate
  portfolios during data warm-up.

References
----------
Moskowitz & Grinblatt (1999): Do industries explain momentum?
    Journal of Finance, 54(4), 1249-1290.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_PCT = 0.10        # Top decile = long leg
DEFAULT_BOTTOM_PCT = 0.10     # Bottom decile = short leg (long-short mode)
MIN_VALID_STOCKS = 20         # Minimum stocks needed to form a valid portfolio
DEFAULT_TC_BPS = 10           # One-way transaction cost in basis points


# ── Portfolio weights ─────────────────────────────────────────────────────────

def compute_weights(
    momentum_rank: pd.Series,
    top_pct: float = DEFAULT_TOP_PCT,
    bottom_pct: float = DEFAULT_BOTTOM_PCT,
    long_only: bool = True,
) -> pd.Series:
    """
    Compute target portfolio weights for a single rebalance date.

    Parameters
    ----------
    momentum_rank : pd.Series
        Cross-sectional percentile rank [0, 1] for all stocks.
    top_pct : float
        Fraction of stocks forming the long portfolio.
    bottom_pct : float
        Fraction of stocks forming the short portfolio (long-short only).
    long_only : bool
        If True, long-only strategy; else long-short.

    Returns
    -------
    weights : pd.Series
        Target weights summing to 1.0 (long-only) or with zero net exposure
        (long-short).
    """
    valid = momentum_rank.dropna()
    if len(valid) < MIN_VALID_STOCKS:
        return pd.Series(dtype=float)

    long_mask = valid >= (1 - top_pct)
    n_long = long_mask.sum()
    weights = pd.Series(0.0, index=valid.index)

    if n_long == 0:
        return pd.Series(dtype=float)

    weights[long_mask] = 1.0 / n_long

    if not long_only:
        short_mask = valid <= bottom_pct
        n_short = short_mask.sum()
        if n_short > 0:
            weights[short_mask] = -1.0 / n_short   # net-zero dollar neutral

    return weights.reindex(momentum_rank.index, fill_value=0.0)


def get_rebalance_dates(
    date_index: pd.DatetimeIndex,
    start_date: Optional[str] = None,
) -> pd.DatetimeIndex:
    """
    Return the last trading day of each month within the date_index.

    Parameters
    ----------
    date_index : pd.DatetimeIndex
        Full business-day index.
    start_date : str, optional
        Earliest date to include. Used to skip the warm-up period.
    """
    monthly = (
        pd.Series(date_index, index=date_index)
        .resample("BME")
        .last()
        .dropna()
    )
    if start_date:
        monthly = monthly[monthly >= pd.Timestamp(start_date)]
    return pd.DatetimeIndex(monthly.values)


# ── Transaction cost model ────────────────────────────────────────────────────

def compute_transaction_costs(
    old_weights: pd.Series,
    new_weights: pd.Series,
    tc_bps: float = DEFAULT_TC_BPS,
) -> float:
    """
    Estimate one-way transaction costs as a fraction of portfolio value.

    Model: cost = tc_bps/10000 * sum(|Δw|)
    This is the standard linear TC model used in AQR research.

    Parameters
    ----------
    old_weights : pd.Series
        Previous period target weights.
    new_weights : pd.Series
        New period target weights.
    tc_bps : float
        One-way cost in basis points (e.g. 10 bps = 0.10%).

    Returns
    -------
    float : Fraction of NAV consumed by transaction costs.
    """
    all_tickers = old_weights.index.union(new_weights.index)
    old = old_weights.reindex(all_tickers, fill_value=0.0)
    new = new_weights.reindex(all_tickers, fill_value=0.0)
    turnover = (new - old).abs().sum()
    return (tc_bps / 10_000) * turnover


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(
    prices: pd.DataFrame,
    momentum_rank: pd.DataFrame,
    top_pct: float = DEFAULT_TOP_PCT,
    bottom_pct: float = DEFAULT_BOTTOM_PCT,
    long_only: bool = True,
    tc_bps: float = DEFAULT_TC_BPS,
    warmup_months: int = 14,
) -> dict:
    """
    Run the monthly-rebalanced momentum backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices. Index = dates, columns = tickers.
    momentum_rank : pd.DataFrame
        Cross-sectional percentile rank. Same shape as prices.
    top_pct : float
        Fraction of stocks in long leg.
    bottom_pct : float
        Fraction of stocks in short leg (long-short only).
    long_only : bool
        Long-only (True) or long-short (False) strategy.
    tc_bps : float
        One-way transaction cost in basis points.
    warmup_months : int
        Months of price history required before first trade.

    Returns
    -------
    dict with keys:
        'returns'          : Daily portfolio returns (pd.Series)
        'weights_history'  : Dict mapping rebalance date → weights (pd.Series)
        'turnover_history' : pd.Series of portfolio turnover per rebalance
        'tc_history'       : pd.Series of transaction costs per rebalance
        'signal_dates'     : List of rebalance dates
    """
    # Determine start date (after warm-up)
    first_valid = prices.index[0] + pd.DateOffset(months=warmup_months)
    rebal_dates = get_rebalance_dates(prices.index, start_date=str(first_valid.date()))

    daily_returns = prices.pct_change()

    portfolio_returns = pd.Series(0.0, index=prices.index)
    weights_history = {}
    turnover_history = {}
    tc_history = {}

    current_weights = pd.Series(dtype=float)
    all_dates = prices.index

    for i, rebal_date in enumerate(rebal_dates):
        # Compute new weights from signal on rebalance date
        if rebal_date not in momentum_rank.index:
            continue
        new_weights = compute_weights(
            momentum_rank.loc[rebal_date],
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            long_only=long_only,
        )
        if new_weights.empty:
            continue

        # Transaction costs
        tc = compute_transaction_costs(current_weights, new_weights, tc_bps)
        turnover = (new_weights.reindex(current_weights.index.union(new_weights.index), fill_value=0.0)
                    - current_weights.reindex(current_weights.index.union(new_weights.index), fill_value=0.0)
                    ).abs().sum()

        tc_history[rebal_date] = tc
        turnover_history[rebal_date] = turnover
        weights_history[rebal_date] = new_weights.copy()
        current_weights = new_weights.copy()

        # Holding period: from day after rebalance to next rebalance
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else all_dates[-1]
        hold_dates = all_dates[(all_dates > rebal_date) & (all_dates <= next_rebal)]

        for date in hold_dates:
            if date not in daily_returns.index:
                continue
            day_rets = daily_returns.loc[date].reindex(current_weights.index, fill_value=0.0)
            port_ret = (current_weights * day_rets).sum()

            # Deduct transaction costs on rebalance day (day after rebal date)
            if date == hold_dates[0]:
                port_ret -= tc

            portfolio_returns.loc[date] = port_ret

    # Trim to valid range
    valid_start = rebal_dates[0] if len(rebal_dates) > 0 else all_dates[0]
    portfolio_returns = portfolio_returns.loc[valid_start:]

    return {
        "returns": portfolio_returns,
        "weights_history": weights_history,
        "turnover_history": pd.Series(turnover_history),
        "tc_history": pd.Series(tc_history),
        "signal_dates": list(rebal_dates),
    }
