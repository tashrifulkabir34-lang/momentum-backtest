"""
data_generator.py
=================
Generates realistic synthetic S&P 500 constituent price data for backtesting.

Methodology:
    - Simulates 100 stocks across 11 GICS sectors over ~15 years (2008-2023)
    - Uses a multi-factor return model: market beta + sector + idiosyncratic
    - Incorporates fat tails (Student-t), volatility clustering (GARCH-like),
      and realistic cross-sectional dispersion to mimic real equity markets
    - Benchmark (SPY) is derived from market factor + noise

References:
    - Fama-French (1993): Common risk factors in stock returns
    - Campbell, Lo, MacKinlay (1997): The Econometrics of Financial Markets
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ── Constants ────────────────────────────────────────────────────────────────

SEED = 42
START_DATE = "2008-01-01"
END_DATE = "2023-12-31"
N_STOCKS = 100
INITIAL_PRICE = 100.0

SECTORS = [
    "Information Technology",
    "Health Care",
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

# Approximate S&P 500 sector weights (%)
SECTOR_WEIGHTS = [27, 13, 13, 11, 9, 9, 6, 4, 3, 2, 2]

# Sector betas to market factor
SECTOR_BETAS = {
    "Information Technology": 1.20,
    "Health Care": 0.80,
    "Financials": 1.15,
    "Consumer Discretionary": 1.10,
    "Communication Services": 1.05,
    "Industrials": 1.00,
    "Consumer Staples": 0.65,
    "Energy": 0.95,
    "Utilities": 0.50,
    "Real Estate": 0.75,
    "Materials": 1.05,
}

# Annualised sector volatility (idiosyncratic component)
SECTOR_IDIO_VOL = {
    "Information Technology": 0.28,
    "Health Care": 0.22,
    "Financials": 0.25,
    "Consumer Discretionary": 0.24,
    "Communication Services": 0.23,
    "Industrials": 0.20,
    "Consumer Staples": 0.16,
    "Energy": 0.27,
    "Utilities": 0.14,
    "Real Estate": 0.20,
    "Materials": 0.22,
}


# ── Helper functions ─────────────────────────────────────────────────────────

def _assign_sectors(n_stocks: int, rng: np.random.Generator) -> pd.Series:
    """Assign sectors to stocks proportional to S&P 500 weights."""
    weights = np.array(SECTOR_WEIGHTS, dtype=float)
    weights /= weights.sum()
    counts = np.round(weights * n_stocks).astype(int)
    # Adjust rounding error
    counts[-1] += n_stocks - counts.sum()
    sectors = []
    for sector, count in zip(SECTORS, counts):
        sectors.extend([sector] * count)
    tickers = [f"STK{i:03d}" for i in range(1, n_stocks + 1)]
    return pd.Series(sectors[:n_stocks], index=tickers, name="sector")


def _simulate_market_factor(
    n_days: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate daily market factor returns with:
    - Long-run drift ~7% annualised
    - GARCH(1,1)-like volatility clustering
    - Student-t innovations (fat tails, df=5)
    - Two crisis episodes (2008-09, 2020)
    """
    daily_drift = 0.07 / 252
    base_vol = 0.16 / np.sqrt(252)

    returns = np.zeros(n_days)
    sigma2 = base_vol**2

    omega = 0.000001
    alpha = 0.09   # ARCH
    beta = 0.88    # GARCH

    for t in range(n_days):
        sigma = np.sqrt(sigma2)
        z = rng.standard_t(df=5) / np.sqrt(5 / 3)  # standardise
        returns[t] = daily_drift + sigma * z
        sigma2 = omega + alpha * returns[t] ** 2 + beta * sigma2

    # Inject crisis drawdowns
    crisis_2008 = int(n_days * 0.05)   # ~mid-2008
    crisis_2020 = int(n_days * 0.75)   # ~early-2020
    for start, length, magnitude in [
        (crisis_2008, 120, -0.008),
        (crisis_2020, 40, -0.018),
    ]:
        returns[start : start + length] += magnitude

    return returns


def _simulate_stock_returns(
    market_returns: np.ndarray,
    sector_series: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate individual stock returns using a two-factor model:
        r_i,t = beta_i * r_mkt,t + beta_s * r_sector,t + epsilon_i,t

    Momentum is seeded by slight persistence in idiosyncratic returns.
    """
    n_days = len(market_returns)
    tickers = sector_series.index.tolist()
    n_stocks = len(tickers)

    all_returns = np.zeros((n_days, n_stocks))

    # Sector factor returns (correlated sub-index)
    sector_returns = {}
    for sector in SECTORS:
        sector_mkt_beta = 0.7
        idio_vol = SECTOR_IDIO_VOL[sector] / np.sqrt(252)
        sector_returns[sector] = (
            sector_mkt_beta * market_returns
            + rng.normal(0, idio_vol, n_days)
        )

    for i, ticker in enumerate(tickers):
        sector = sector_series[ticker]
        mkt_beta = SECTOR_BETAS[sector] * rng.uniform(0.7, 1.3)
        sector_beta = rng.uniform(0.1, 0.4)
        idio_vol = SECTOR_IDIO_VOL[sector] * rng.uniform(0.8, 1.2) / np.sqrt(252)

        # Slight persistence in idiosyncratic component (seeds momentum)
        idio = np.zeros(n_days)
        rho = rng.uniform(0.02, 0.06)   # mild autocorrelation
        for t in range(1, n_days):
            idio[t] = rho * idio[t - 1] + rng.normal(0, idio_vol)

        all_returns[:, i] = (
            mkt_beta * market_returns
            + sector_beta * sector_returns[sector]
            + idio
        )

    return pd.DataFrame(all_returns, columns=tickers)


def generate_price_data(
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Generate synthetic daily price data for backtesting.

    Returns
    -------
    prices : pd.DataFrame
        Daily adjusted close prices, shape (n_days, n_stocks).
    benchmark : pd.DataFrame
        Daily benchmark (SPY-like) prices, shape (n_days, 1).
    sector_map : pd.Series
        Sector label for each ticker.
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    n_days = len(dates)

    sector_map = _assign_sectors(N_STOCKS, rng)
    market_returns = _simulate_market_factor(n_days, rng)
    stock_returns_df = _simulate_stock_returns(market_returns, sector_map, rng)
    stock_returns_df.index = dates

    # Convert returns → prices
    prices = INITIAL_PRICE * (1 + stock_returns_df).cumprod()

    # Benchmark: market factor + small noise
    bench_noise = rng.normal(0, 0.001, n_days)
    bench_returns = market_returns + bench_noise
    benchmark = pd.DataFrame(
        INITIAL_PRICE * (1 + pd.Series(bench_returns, index=dates)).cumprod(),
        columns=["SPY"],
    )

    return prices, benchmark, sector_map


if __name__ == "__main__":
    prices, benchmark, sector_map = generate_price_data()
    print(f"Prices shape   : {prices.shape}")
    print(f"Date range     : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Sectors        :\n{sector_map.value_counts()}")
    print(f"Benchmark head :\n{benchmark.head()}")
