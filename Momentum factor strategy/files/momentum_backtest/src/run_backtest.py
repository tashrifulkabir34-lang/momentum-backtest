"""
run_backtest.py
===============
Main entry point for the Momentum Factor Strategy Backtest.

Execution flow
--------------
1. Generate synthetic S&P 500 constituent price data
2. Construct 12-1 month momentum signals
3. Split data into in-sample (2008-2018) and out-of-sample (2019-2023)
4. Run monthly-rebalanced backtest with transaction cost modelling
5. Compute full performance metrics
6. Generate and save tearsheet PNG + CSV results

Usage
-----
    python run_backtest.py [--long-short] [--top-pct 0.10] [--tc-bps 10]

All outputs are written to the `results/` directory.

Author : Momentum Backtest Project
License: MIT
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure src/ is on path when called from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_price_data
from signals import construct_signals
from portfolio import run_backtest
from metrics import compute_full_metrics, monthly_returns_table
from tearsheet import generate_tearsheet

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Momentum Factor Backtest — 12-1 Month Price Momentum"
    )
    parser.add_argument("--long-short", action="store_true",
                        help="Run long-short instead of long-only strategy")
    parser.add_argument("--top-pct", type=float, default=0.10,
                        help="Fraction of stocks in long portfolio (default 0.10 = top decile)")
    parser.add_argument("--bottom-pct", type=float, default=0.10,
                        help="Fraction of stocks in short portfolio (long-short only)")
    parser.add_argument("--tc-bps", type=float, default=10,
                        help="One-way transaction cost in basis points (default 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def print_section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def main() -> None:
    args = parse_args()
    strategy_type = "Long-Short" if args.long_short else "Long-Only"

    print_section(f"Momentum Factor Backtest  |  {strategy_type}")
    print(f"  Top quantile  : {args.top_pct:.0%}")
    print(f"  TC (one-way)  : {args.tc_bps:.0f} bps")
    print(f"  Random seed   : {args.seed}")

    # ── 1. Data generation ────────────────────────────────────────────────────
    print_section("Step 1 — Generating synthetic market data")
    prices, benchmark, sector_map = generate_price_data(seed=args.seed)
    bench_returns = benchmark["SPY"].pct_change().dropna()
    print(f"  Universe      : {prices.shape[1]} stocks")
    print(f"  Date range    : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Trading days  : {len(prices):,}")

    # ── 2. Signal construction ─────────────────────────────────────────────────
    print_section("Step 2 — Constructing 12-1 month momentum signals")
    signals = construct_signals(prices)
    mom_rank = signals["momentum_rank"]
    quality = signals["signal_quality"]
    print(f"  Avg signal coverage: {quality.mean():.1%} of universe")
    print(f"  First valid signal : {mom_rank.dropna(how='all').index[0].date()}")

    # ── 3. Train / test split ─────────────────────────────────────────────────
    IS_END = "2018-12-31"
    OOS_START = "2019-01-01"

    print_section("Step 3 — Train / Test split")
    print(f"  In-sample     : 2008-01-01 → {IS_END}")
    print(f"  Out-of-sample : {OOS_START} → 2023-12-31")

    # ── 4. Run backtest ───────────────────────────────────────────────────────
    print_section("Step 4 — Running full-period backtest")
    result = run_backtest(
        prices=prices,
        momentum_rank=mom_rank,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
        long_only=not args.long_short,
        tc_bps=args.tc_bps,
    )

    strat_returns = result["returns"]
    turnover_history = result["turnover_history"]
    tc_history = result["tc_history"]

    # Align benchmark
    bench_aligned = bench_returns.reindex(strat_returns.index, fill_value=0.0)

    # ── 5. Performance metrics ────────────────────────────────────────────────
    print_section("Step 5 — Performance metrics")

    for label, start, end in [
        ("Full period", strat_returns.index[0], strat_returns.index[-1]),
        ("In-sample", strat_returns.index[0], pd.Timestamp(IS_END)),
        ("Out-of-sample", pd.Timestamp(OOS_START), strat_returns.index[-1]),
    ]:
        s = strat_returns.loc[start:end]
        b = bench_aligned.loc[start:end]
        metrics_df = compute_full_metrics(s, b, "Momentum", "SPY")
        print(f"\n  [{label}]")
        print(metrics_df.to_string())

    # ── 6. Transaction cost summary ───────────────────────────────────────────
    print_section("Step 6 — Transaction cost analysis")
    avg_turnover = turnover_history.mean() * 100
    avg_tc = tc_history.mean() * 100 * 252 / 12  # annualised
    print(f"  Avg monthly turnover     : {avg_turnover:.1f}%")
    print(f"  Avg annualised TC drag   : {avg_tc:.2f}%")
    print(f"  Total rebalances         : {len(turnover_history)}")

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    print_section("Step 7 — Saving results")

    # Returns CSVs
    strat_returns.to_csv(os.path.join(RESULTS_DIR, "strategy_returns.csv"),
                         header=["momentum_return"])
    bench_aligned.to_csv(os.path.join(RESULTS_DIR, "benchmark_returns.csv"),
                         header=["spy_return"])

    # Full metrics table
    full_metrics = compute_full_metrics(strat_returns, bench_aligned, "Momentum", "SPY")
    full_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_full.csv"))

    # IS metrics
    is_metrics = compute_full_metrics(
        strat_returns.loc[:IS_END], bench_aligned.loc[:IS_END], "Momentum", "SPY"
    )
    is_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_in_sample.csv"))

    # OOS metrics
    oos_metrics = compute_full_metrics(
        strat_returns.loc[OOS_START:], bench_aligned.loc[OOS_START:], "Momentum", "SPY"
    )
    oos_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_out_of_sample.csv"))

    # Monthly returns
    try:
        monthly = monthly_returns_table(strat_returns)
        monthly.to_csv(os.path.join(RESULTS_DIR, "monthly_returns.csv"))
    except Exception:
        pass

    # Turnover / TC
    pd.DataFrame({"turnover": turnover_history, "tc_cost": tc_history}).to_csv(
        os.path.join(RESULTS_DIR, "turnover_tc.csv")
    )

    print(f"  CSVs saved to : {RESULTS_DIR}/")

    # ── 8. Tearsheet ──────────────────────────────────────────────────────────
    print_section("Step 8 — Generating tearsheet")
    tearsheet_path = os.path.join(RESULTS_DIR, "tearsheet.png")
    generate_tearsheet(
        strat_returns=strat_returns,
        bench_returns=bench_aligned,
        turnover_history=turnover_history,
        strat_name=f"Momentum {strategy_type} (12-1m)",
        bench_name="SPY Benchmark",
        split_date=OOS_START,
        output_path=tearsheet_path,
    )

    print_section("✅  Backtest complete")
    print(f"  Tearsheet : {tearsheet_path}")
    print(f"  Results   : {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
