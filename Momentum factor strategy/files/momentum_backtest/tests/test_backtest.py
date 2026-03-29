"""
test_backtest.py
================
Unit tests for the momentum backtest pipeline.

Tests cover:
  - Signal construction correctness
  - Portfolio weight properties
  - Metric computations
  - Transaction cost model
  - Data generation consistency

Run with:
    python -m pytest tests/test_backtest.py -v
or:
    python tests/test_backtest.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_generator import generate_price_data
from signals import compute_momentum_signal, cross_sectional_rank
from portfolio import compute_weights, compute_transaction_costs
from metrics import (
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
    value_at_risk,
    expected_shortfall,
)


# ── Test helpers ──────────────────────────────────────────────────────────────

def _make_prices(n_days=300, n_stocks=20, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.015, size=(n_days, n_stocks))
    prices = 100 * (1 + returns).cumprod(axis=0)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"STK{i:03d}" for i in range(n_stocks)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_returns(n_days=500, seed=1):
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0004, 0.012, n_days)
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.Series(vals, index=idx)


# ── Signal tests ──────────────────────────────────────────────────────────────

class TestSignals:
    def test_momentum_shape(self):
        prices = _make_prices()
        mom = compute_momentum_signal(prices)
        assert mom.shape == prices.shape, "Signal shape must match prices"

    def test_momentum_nan_warmup(self):
        prices = _make_prices(n_days=300)
        mom = compute_momentum_signal(prices)
        # The skip lag (21 days) means we need prices[skip:] / prices[lookback:]
        # First valid at row=lookback (shift(lookback) is NaN before that)
        # Row 252 is first valid: prices.shift(252) first non-NaN is row 252
        assert mom.iloc[:252].isna().all().all(), "Warm-up rows must be NaN"

    def test_rank_bounds(self):
        prices = _make_prices(n_days=300, n_stocks=30)
        mom = compute_momentum_signal(prices)
        rank = cross_sectional_rank(mom)
        valid = rank.dropna(how="all")
        assert (valid.min(axis=1) >= 0).all(), "Rank must be >= 0"
        assert (valid.max(axis=1) <= 1).all(), "Rank must be <= 1"

    def test_rank_monotone(self):
        """Higher raw momentum → higher rank."""
        prices = _make_prices(n_days=300, n_stocks=10)
        mom = compute_momentum_signal(prices)
        rank = cross_sectional_rank(mom)
        valid_dates = mom.dropna(how="all").index
        date = valid_dates[-1]
        raw = mom.loc[date].dropna()
        rnk = rank.loc[date].dropna()
        common = raw.index.intersection(rnk.index)
        corr = raw[common].corr(rnk[common])
        assert corr > 0.95, f"Rank should be monotone with signal, got corr={corr:.4f}"


# ── Portfolio tests ───────────────────────────────────────────────────────────

class TestPortfolio:
    def test_long_weights_sum_to_one(self):
        rank = pd.Series(np.linspace(0, 1, 50))
        weights = compute_weights(rank, top_pct=0.2, long_only=True)
        assert abs(weights.sum() - 1.0) < 1e-9, "Long-only weights must sum to 1"

    def test_long_weights_non_negative(self):
        rank = pd.Series(np.linspace(0, 1, 50))
        weights = compute_weights(rank, top_pct=0.2, long_only=True)
        assert (weights >= 0).all(), "Long-only weights must all be non-negative"

    def test_long_short_approx_zero_net(self):
        rank = pd.Series(np.linspace(0, 1, 50))
        weights = compute_weights(rank, top_pct=0.2, bottom_pct=0.2, long_only=False)
        # Dollar-neutral: sum should be approximately 0
        assert abs(weights.sum()) < 1e-9, "Long-short weights must be dollar-neutral"

    def test_top_stocks_selected(self):
        n = 50
        rank = pd.Series(np.linspace(0, 1, n))
        weights = compute_weights(rank, top_pct=0.1, long_only=True)
        # Top 10% → indices 45-49
        assert (weights.iloc[:45] == 0).all(), "Low-rank stocks must have zero weight"
        assert (weights.iloc[45:] > 0).all(), "Top-rank stocks must have positive weight"


# ── Transaction cost tests ────────────────────────────────────────────────────

class TestTransactionCosts:
    def test_zero_cost_no_change(self):
        weights = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        tc = compute_transaction_costs(weights, weights, tc_bps=10)
        assert tc == 0.0, "No turnover → zero TC"

    def test_full_turnover(self):
        old = pd.Series([1.0, 0.0], index=["A", "B"])
        new = pd.Series([0.0, 1.0], index=["A", "B"])
        tc = compute_transaction_costs(old, new, tc_bps=10)
        # Turnover = 2 (|−1| + |+1|), cost = 2 * 10/10000 = 0.002
        assert abs(tc - 0.002) < 1e-9, f"Expected TC=0.002, got {tc}"

    def test_tc_proportional_to_bps(self):
        old = pd.Series([0.6, 0.4], index=["A", "B"])
        new = pd.Series([0.4, 0.6], index=["A", "B"])
        tc10 = compute_transaction_costs(old, new, tc_bps=10)
        tc20 = compute_transaction_costs(old, new, tc_bps=20)
        assert abs(tc20 / tc10 - 2.0) < 1e-9, "TC should scale linearly with bps"


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_annualised_return_positive(self):
        # Use clearly positive daily returns to guarantee positive annualised return
        returns = pd.Series(
            [0.001] * 252, index=pd.bdate_range("2020-01-01", periods=252)
        )
        assert annualised_return(returns) > 0, "Positive returns → positive ann. return"

    def test_sharpe_sign(self):
        pos_returns = _make_returns() + 0.001
        neg_returns = _make_returns() - 0.001
        assert sharpe_ratio(pos_returns) > 0, "Positive returns → positive Sharpe"
        assert sharpe_ratio(neg_returns) < 0, "Negative returns → negative Sharpe"

    def test_max_drawdown_negative(self):
        returns = _make_returns()
        mdd = max_drawdown(returns)
        assert mdd <= 0, "Max drawdown must be non-positive"

    def test_max_drawdown_zero_for_monotone(self):
        """Strictly increasing NAV → zero drawdown."""
        returns = pd.Series([0.01] * 100, index=pd.bdate_range("2020-01-01", periods=100))
        assert abs(max_drawdown(returns)) < 1e-10, "Monotone up → zero drawdown"

    def test_var_less_than_es(self):
        returns = _make_returns()
        var = value_at_risk(returns, 0.95)
        es = expected_shortfall(returns, 0.95)
        assert es >= var, "ES must be >= VaR at same confidence level"

    def test_vol_positive(self):
        returns = _make_returns()
        assert annualised_volatility(returns) > 0


# ── Data generator test ───────────────────────────────────────────────────────

class TestDataGenerator:
    def test_reproducibility(self):
        p1, _, _ = generate_price_data(seed=42)
        p2, _, _ = generate_price_data(seed=42)
        assert (p1 == p2).all().all(), "Same seed → identical data"

    def test_prices_positive(self):
        prices, benchmark, _ = generate_price_data(seed=0)
        assert (prices > 0).all().all(), "All prices must be positive"
        assert (benchmark > 0).all().all(), "Benchmark must be positive"

    def test_shape(self):
        prices, benchmark, sector_map = generate_price_data()
        assert prices.shape[1] == 100, "Should have 100 stocks"
        assert benchmark.shape[1] == 1, "Benchmark should be 1 column"
        assert len(sector_map) == 100, "Sector map length mismatch"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tests():
    test_classes = [
        TestSignals,
        TestPortfolio,
        TestTransactionCosts,
        TestMetrics,
        TestDataGenerator,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method in methods:
            try:
                getattr(instance, method)()
                print(f"  ✅ {cls.__name__}.{method}")
                passed += 1
            except AssertionError as e:
                print(f"  ❌ {cls.__name__}.{method}: {e}")
                errors.append((cls.__name__, method, str(e)))
                failed += 1
            except Exception as e:
                print(f"  💥 {cls.__name__}.{method}: {type(e).__name__}: {e}")
                errors.append((cls.__name__, method, str(e)))
                failed += 1

    print(f"\n{'─'*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    if errors:
        print("\n  Failures:")
        for cls_name, method, msg in errors:
            print(f"    - {cls_name}.{method}: {msg}")
    return failed == 0


if __name__ == "__main__":
    print("Running momentum backtest tests...\n")
    success = run_all_tests()
    sys.exit(0 if success else 1)
