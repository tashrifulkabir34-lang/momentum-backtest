"""
signals.py
==========
Momentum factor signal construction.

Implements the classic 12-1 month (Jegadeesh & Titman 1993) price momentum
signal:
    MOM_i,t = (P_{t-21} / P_{t-252}) - 1

The 1-month skip lag avoids short-term mean-reversion (microstructure reversal)
documented by Jegadeesh (1990).

References
----------
Jegadeesh & Titman (1993): Returns to buying winners and selling losers.
    Journal of Finance, 48(1), 65-91.
Asness, Moskowitz & Pedersen (2013): Value and Momentum Everywhere.
    Journal of Finance, 68(3), 929-985.
"""

import numpy as np
import pandas as pd


# ── Business-day approximations ───────────────────────────────────────────────
LOOKBACK_DAYS = 252   # ~12 months
SKIP_DAYS = 21        # ~1 month  (the "skip lag")
MIN_HISTORY = LOOKBACK_DAYS + SKIP_DAYS


def compute_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = LOOKBACK_DAYS,
    skip: int = SKIP_DAYS,
) -> pd.DataFrame:
    """
    Compute cross-sectional 12-1 month momentum for every stock on every date.

    Signal = cumulative return from t-lookback to t-skip
    (skipping the most recent `skip` trading days).

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted-close prices. Index = dates, columns = tickers.
    lookback : int
        Number of trading days in the formation period (default 252 ≈ 12 months).
    skip : int
        Number of most-recent days to skip (default 21 ≈ 1 month).

    Returns
    -------
    momentum : pd.DataFrame
        Raw momentum signal (same shape as prices, NaN before sufficient history).
    """
    # Price at the start of the formation window
    price_start = prices.shift(lookback)
    # Price at the end of the formation window (skip the last `skip` days)
    price_end = prices.shift(skip)

    momentum = (price_end / price_start) - 1.0
    return momentum


def cross_sectional_rank(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw signal to cross-sectional percentile rank [0, 1] each day.
    Stocks with fewer than MIN_HISTORY days of data are excluded (NaN).
    """
    return signal.rank(axis=1, pct=True)


def construct_signals(prices: pd.DataFrame) -> dict:
    """
    End-to-end signal construction pipeline.

    Returns
    -------
    dict with keys:
        'momentum_raw'   : raw 12-1m return signal
        'momentum_rank'  : cross-sectional percentile rank
        'signal_quality' : fraction of stocks with valid signal each date
    """
    mom_raw = compute_momentum_signal(prices)
    mom_rank = cross_sectional_rank(mom_raw)
    signal_quality = mom_raw.notna().sum(axis=1) / prices.shape[1]

    return {
        "momentum_raw": mom_raw,
        "momentum_rank": mom_rank,
        "signal_quality": signal_quality,
    }
