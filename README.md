# 📈 Momentum Factor Strategy Backtest

![Tearsheet](results/tearsheet.png)

> A fully documented, production-grade implementation of the **12-1 month price momentum** strategy across a synthetic S&P 500 universe, with transaction cost modelling, train/test splitting, and a professional performance tearsheet.

## Quickstart
```bash
pip install -r requirements.txt
python src/run_backtest.py
```

## Results

| Metric | Momentum (Long-Only) | SPY Benchmark |
|---|---|---|
| Annualised Return | -1.86% | -1.31% |
| Sharpe Ratio | -0.21 | -0.30 |
| Max Drawdown | -62.4% | -64.0% |
| OOS Alpha (ann.) | +7.01% | — |
| OOS Info. Ratio | 0.80 | — |

## Project Structure

- `src/data_generator.py` — Synthetic S&P 500 market data
- `src/signals.py` — 12-1 month momentum signal
- `src/portfolio.py` — Monthly rebalancing + transaction costs
- `src/metrics.py` — Sharpe, Sortino, VaR, CVaR, Alpha, IR
- `src/tearsheet.py` — Full performance tearsheet
- `src/run_backtest.py` — Main entry point
- `tests/test_backtest.py` — 20 unit tests (all passing)

Built with Python · NumPy · pandas · Matplotlib
