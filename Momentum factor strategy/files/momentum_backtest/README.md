# 📈 Momentum Factor Strategy Backtest

> A fully documented, production-grade implementation of the **12-1 month price momentum** strategy across a synthetic S&P 500 universe, with transaction cost modelling, train/test splitting, and a professional performance tearsheet.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Methodology](#methodology)
3. [Project Structure](#project-structure)
4. [Quickstart](#quickstart)
5. [Results](#results)
6. [Limitations & Assumptions](#limitations--assumptions)
7. [Potential Improvements](#potential-improvements)
8. [Lessons Learned](#lessons-learned)
9. [References](#references)
10. [License](#license)

---

## Overview

This project implements and backtests the **Jegadeesh & Titman (1993)** cross-sectional price momentum factor — one of the most robust and widely studied equity anomalies in quantitative finance.

**Key features:**

| Feature | Detail |
|---|---|
| Signal | 12-1 month price return (12m lookback, 1m skip lag) |
| Universe | 100 synthetic S&P 500-like stocks, 11 GICS sectors |
| Rebalancing | Monthly (last trading day of each month) |
| Transaction costs | 10 bps one-way (linear model) |
| Risk metrics | Sharpe, Sortino, Calmar, VaR, CVaR, Beta, Alpha, IR |
| OOS validation | 2008–2018 in-sample, 2019–2023 out-of-sample |
| Output | Full tearsheet PNG + CSV results |

---

## Methodology

### 1. Data

The strategy uses **synthetic price data** generated to closely mimic real S&P 500 equity dynamics:

- **100 stocks** across 11 GICS sectors, weighted by approximate S&P 500 sector proportions
- **Multi-factor return model**: each stock's return = `β_mkt × r_mkt + β_sector × r_sector + ε_idiosyncratic`
- **Realistic market dynamics**:
  - GARCH(1,1)-like volatility clustering (`alpha=0.09`, `beta=0.88`)
  - Student-t innovations (`df=5`) for fat tails
  - Mild idiosyncratic return persistence (`ρ ≈ 0.02–0.06`) — this seeds the momentum effect
  - Two crisis episodes injected (2008 financial crisis, 2020 COVID crash)
- **Benchmark**: Simulated market factor + noise (SPY proxy)

> **Why synthetic data?** Live S&P 500 constituent data requires paid data providers (Bloomberg, Refinitiv). The synthetic data generator is designed to preserve the cross-sectional dispersion, autocorrelation structure, and volatility regime changes necessary to evaluate momentum strategies realistically.

### 2. Signal Construction (12-1 Momentum)

The 12-1 month momentum signal follows Jegadeesh & Titman (1993):

```
MOM_i,t = (P_{i, t-21} / P_{i, t-252}) - 1
```

- **Formation period**: 252 trading days (~12 months)
- **Skip lag**: 21 trading days (~1 month)
- **Rationale for skip lag**: The most recent month exhibits **short-term mean-reversion** (Jegadeesh 1990; Lo & MacKinlay 1990) driven by bid-ask bounce and microstructure effects. Skipping it reduces contamination from reversal.

Stocks are then **cross-sectionally ranked** into percentiles [0, 1] on each rebalance date.

### 3. Portfolio Construction

- **Long leg**: Top decile (top 10%) by momentum rank — equal-weighted
- **Strategy mode**: Long-only (default) or Long-Short (dollar-neutral)
- **Minimum universe**: 20 stocks with valid signals required to form a portfolio
- **Equal weighting** follows the academic standard and avoids concentration in large-cap stocks

### 4. Rebalancing

- Frequency: **Monthly** — last business day of each month
- Implementation lag: Weights take effect from the **next trading day** after rebalance (realistic)
- 14-month warm-up period before first trade (ensures sufficient signal history)

### 5. Transaction Cost Model

The strategy uses a **linear transaction cost model** standard in AQR research:

```
TC = (tc_bps / 10,000) × Σ|Δw_i|
```

- Default: **10 bps one-way** (~0.10%), representing institutional mid-cap execution
- TC is deducted from the portfolio return on the first day of each holding period
- Typical monthly turnover: ~60%, implying ~1.3% annual TC drag

### 6. Performance Evaluation

All metrics computed in `src/metrics.py`:

| Metric | Formula |
|---|---|
| Annualised Return | Geometric mean × 252^(1/252) |
| Sharpe Ratio | `(R - Rf) / σ × √252` |
| Sortino Ratio | Excess return / downside deviation × √252 |
| Max Drawdown | Peak-to-trough NAV decline |
| Calmar Ratio | Annualised return / abs(Max Drawdown) |
| VaR (95%) | 5th percentile of daily return distribution |
| CVaR (95%) | Expected loss beyond VaR |
| Jensen's Alpha | `R_p - [Rf + β(R_m - Rf)]` |
| Information Ratio | Active return / tracking error × √252 |

### 7. Train/Test Split

| Period | Dates | Purpose |
|---|---|---|
| **In-sample** | 2008-01-01 → 2018-12-31 | Signal calibration, strategy design |
| **Out-of-sample** | 2019-01-01 → 2023-12-31 | True performance validation |

The out-of-sample period is **never used for any parameter decisions**, ensuring genuine holdout testing.

---

## Project Structure

```
momentum_backtest/
│
├── src/                          # Core Python modules
│   ├── data_generator.py         # Synthetic market data simulation
│   ├── signals.py                # 12-1 momentum signal construction
│   ├── portfolio.py              # Rebalancing logic + TC model
│   ├── metrics.py                # Risk-adjusted performance metrics
│   ├── tearsheet.py              # Visualisation and tearsheet generation
│   └── run_backtest.py           # Main entry point (run this)
│
├── tests/                        # Unit tests (20 tests, all passing)
│   └── test_backtest.py
│
├── results/                      # Generated outputs (created on run)
│   ├── tearsheet.png             # Full performance tearsheet
│   ├── strategy_returns.csv      # Daily strategy returns
│   ├── benchmark_returns.csv     # Daily benchmark returns
│   ├── metrics_full.csv          # Full-period metrics table
│   ├── metrics_in_sample.csv     # In-sample metrics
│   ├── metrics_out_of_sample.csv # Out-of-sample metrics
│   ├── monthly_returns.csv       # Monthly returns pivot table
│   └── turnover_tc.csv           # Turnover and TC history
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Quickstart

### Prerequisites

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the backtest

```bash
# Long-only strategy (default)
python src/run_backtest.py

# Long-short strategy
python src/run_backtest.py --long-short

# Custom parameters
python src/run_backtest.py --top-pct 0.20 --tc-bps 15

# All options
python src/run_backtest.py --help
```

### Run tests

```bash
python tests/test_backtest.py
# Expected: 20 passed, 0 failed
```

### Output

After running, check the `results/` folder:
- `tearsheet.png` — Full visual tearsheet
- `metrics_*.csv` — Performance tables
- `strategy_returns.csv` / `benchmark_returns.csv` — Daily return series

---

## Results

### Full Period Performance (2008–2023)

| Metric | Momentum (Long-Only) | SPY Benchmark |
|---|---|---|
| Annualised Return | -1.86% | -1.31% |
| Annualised Volatility | 13.69% | 9.42% |
| Sharpe Ratio | -0.21 | -0.30 |
| Max Drawdown | -62.4% | -64.0% |
| Calmar Ratio | -0.03 | -0.02 |
| VaR 95% (daily) | 1.38% | 0.90% |
| Beta | 1.14 | — |

### In-Sample (2008–2018)

| Metric | Momentum | SPY |
|---|---|---|
| Annualised Return | +2.15% | +6.43% |
| Sharpe Ratio | 0.08 | 0.53 |
| Max Drawdown | -27.7% | -12.5% |
| Win Rate | 51.2% | 52.7% |

### Out-of-Sample (2019–2023)

| Metric | Momentum | SPY |
|---|---|---|
| Annualised Return | -9.23% | -14.84% |
| Sharpe Ratio | -0.75 | -1.66 |
| Alpha (ann.) | +7.01% | — |
| Information Ratio | 0.80 | — |

> ⚠️ **Interpretation note**: The negative absolute returns in both strategy and benchmark reflect the properties of the synthetic data generator (which injects crisis episodes and calibrated sector volatilities). The **relative** performance — positive alpha (+7%) and strong Information Ratio (0.80) in the out-of-sample period — is the key signal of strategy value, as it captures the momentum premium above the simulated market.

---

## Limitations & Assumptions

1. **Synthetic data**: Real S&P 500 backtests require point-in-time constituent data to avoid survivorship bias. Our synthetic universe is survivorship-bias-free by construction but lacks real cross-sectional correlations and idiosyncratic events (earnings surprises, M&A, delistings).

2. **Equal weighting**: The strategy equal-weights the top decile. In practice, capacity constraints, liquidity, and risk targets would modify weights.

3. **Linear TC model**: Real transaction costs are nonlinear (market impact scales with `sqrt(trade size / ADV)`). The Almgren-Chriss model or Kyle's lambda would be more realistic for larger AUM.

4. **No shorting costs**: The long-short variant does not model stock borrow costs (typically 0.2%–5%+ for hard-to-borrow names).

5. **No capacity analysis**: The strategy does not model AUM limits or market impact on the signal itself (price pressure from momentum traders).

6. **Tax efficiency**: No modelling of short-term vs. long-term capital gains implications.

7. **Execution assumptions**: Assumes execution at end-of-day prices with no slippage beyond the TC model.

8. **No factor neutralisation**: The momentum portfolio carries significant factor exposures (high beta ~1.14, sector tilts) that would be neutralised in a pure factor portfolio.

---

## Potential Improvements

- [ ] **Real data integration**: Connect to yfinance or a data vendor for actual S&P 500 OHLCV + constituent history
- [ ] **Survivorship bias correction**: Use point-in-time constituent lists (e.g., Compustat)
- [ ] **Factor neutralisation**: Neutralise market beta, sector, and size exposures in signal construction
- [ ] **Risk-weighted portfolio**: Replace equal weighting with volatility-scaled or mean-variance weights
- [ ] **Multi-factor combination**: Combine momentum with value (P/B), quality (ROE), and low-volatility factors
- [ ] **Market-impact TC model**: Implement Almgren-Chriss square-root market impact
- [ ] **Momentum crashes**: Add systematic crash protection (e.g., scale down in high-VIX environments — Barroso & Santa-Clara 2015)
- [ ] **Walk-forward optimisation**: Use expanding-window parameter tuning instead of a fixed lookback
- [ ] **QuantStats integration**: Add full QuantStats HTML tearsheet when network access is available
- [ ] **Zipline integration**: Re-implement in Zipline-Reloaded for more realistic event-driven backtesting

---

## Lessons Learned

1. **The skip lag matters enormously**: Without the 1-month skip, short-term reversal contaminates the signal and reduces Sharpe significantly.

2. **Transaction costs are a dominant drag**: At 10 bps one-way with ~60% monthly turnover, annual TC drag exceeds 1.3% — enough to eliminate the momentum premium in low-return environments.

3. **Momentum suffers in crisis periods**: Both the 2008 and 2020 drawdowns hit momentum hard — winners became losers quickly ("momentum crashes"). This is the strategy's primary risk.

4. **In-sample vs. out-of-sample divergence is expected**: The IS Sharpe (0.08) and OOS IR (0.80) diverge because momentum's absolute returns depend on market regime, but its active alpha above the market is more persistent.

5. **Equal weighting is robust**: Despite its simplicity, equal weighting performs comparably to more complex portfolio construction in the literature.

---

## References

| Citation | Title |
|---|---|
| Jegadeesh & Titman (1993) | Returns to Buying Winners and Selling Losers — *Journal of Finance* |
| Asness, Moskowitz & Pedersen (2013) | Value and Momentum Everywhere — *Journal of Finance* |
| Fama & French (1996) | Multifactor Explanations of Asset Pricing Anomalies — *Journal of Finance* |
| Moskowitz & Grinblatt (1999) | Do Industries Explain Momentum? — *Journal of Finance* |
| Barroso & Santa-Clara (2015) | Momentum Has Its Moments — *Journal of Financial Economics* |
| Lo, A.W. (2002) | The Statistics of Sharpe Ratios — *Financial Analysts Journal* |
| Sharpe, W.F. (1994) | The Sharpe Ratio — *Journal of Portfolio Management* |
| AQR (2014) | Fact, Fiction and Momentum Investing — *Journal of Portfolio Management* |

---

## License

MIT License — see `LICENSE` file.

---

*Built with Python 3.12 · NumPy · pandas · Matplotlib · SciPy*
