"""
tearsheet.py
============
Professional tearsheet visualisation for the momentum backtest.

Produces a multi-panel PDF/PNG tearsheet including:
  1. Cumulative return comparison (strategy vs benchmark)
  2. Drawdown analysis
  3. Rolling Sharpe ratio (12-month window)
  4. Monthly return heatmap
  5. Return distribution histogram
  6. Rolling annualised volatility
  7. Performance metrics summary table
  8. Sector exposure over time (top quantile)

All styling follows a clean, dark-finance aesthetic.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

from metrics import (
    compute_full_metrics,
    drawdown_series,
    rolling_sharpe,
    monthly_returns_table,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    max_drawdown,
)


# ── Palette ───────────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
MUTED_TEXT = "#8b949e"
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_AMBER = "#d29922"
ACCENT_PURPLE = "#bc8cff"

STRAT_COLOR = ACCENT_BLUE
BENCH_COLOR = ACCENT_AMBER


def _set_ax_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED_TEXT, labelsize=8)
    ax.spines[:].set_color(GRID_COLOR)
    ax.xaxis.label.set_color(MUTED_TEXT)
    ax.yaxis.label.set_color(MUTED_TEXT)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def _pct_formatter(x, _):
    return f"{x:.0f}%"


# ── Individual panels ─────────────────────────────────────────────────────────

def _plot_cumulative(ax, strat_returns, bench_returns, strat_name, bench_name):
    cum_strat = (1 + strat_returns).cumprod() * 100
    cum_bench = (1 + bench_returns.reindex(strat_returns.index, fill_value=0)).cumprod() * 100

    ax.plot(cum_strat.index, cum_strat, color=STRAT_COLOR, lw=1.5, label=strat_name)
    ax.plot(cum_bench.index, cum_bench, color=BENCH_COLOR, lw=1.2, ls="--", label=bench_name, alpha=0.85)
    ax.fill_between(cum_strat.index, cum_strat, cum_bench,
                    where=cum_strat >= cum_bench, alpha=0.08, color=ACCENT_GREEN)
    ax.fill_between(cum_strat.index, cum_strat, cum_bench,
                    where=cum_strat < cum_bench, alpha=0.08, color=ACCENT_RED)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.legend(fontsize=8, framealpha=0.2, loc="upper left",
               labelcolor=TEXT_COLOR, facecolor=PANEL_BG)
    _set_ax_style(ax, title="Cumulative Growth of $100", ylabel="Portfolio Value ($)")


def _plot_drawdown(ax, strat_returns, bench_returns):
    dd_strat = drawdown_series(strat_returns) * 100
    dd_bench = drawdown_series(bench_returns.reindex(strat_returns.index, fill_value=0)) * 100

    ax.fill_between(dd_strat.index, dd_strat, 0, color=STRAT_COLOR, alpha=0.35, label="Strategy")
    ax.fill_between(dd_bench.index, dd_bench, 0, color=BENCH_COLOR, alpha=0.2, label="Benchmark")
    ax.plot(dd_strat.index, dd_strat, color=STRAT_COLOR, lw=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(fontsize=8, framealpha=0.2, loc="lower left",
               labelcolor=TEXT_COLOR, facecolor=PANEL_BG)
    _set_ax_style(ax, title="Drawdown (%)", ylabel="Drawdown (%)")


def _plot_rolling_sharpe(ax, strat_returns, bench_returns, window=252):
    rs_strat = rolling_sharpe(strat_returns, window=window)
    rs_bench = rolling_sharpe(bench_returns.reindex(strat_returns.index, fill_value=0), window=window)

    ax.plot(rs_strat.index, rs_strat, color=STRAT_COLOR, lw=1.2, label="Strategy")
    ax.plot(rs_bench.index, rs_bench, color=BENCH_COLOR, lw=1.0, ls="--", label="Benchmark", alpha=0.8)
    ax.axhline(0, color=MUTED_TEXT, lw=0.8, ls=":")
    ax.axhline(1, color=ACCENT_GREEN, lw=0.6, ls=":", alpha=0.5)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT_COLOR, facecolor=PANEL_BG)
    _set_ax_style(ax, title=f"Rolling {window//252}Y Sharpe Ratio", ylabel="Sharpe Ratio")


def _plot_monthly_heatmap(ax, strat_returns):
    try:
        monthly = monthly_returns_table(strat_returns)
        # Drop 'Annual' for heatmap
        monthly_vals = monthly.drop(columns=["Annual"], errors="ignore")

        cmap = LinearSegmentedColormap.from_list(
            "rg", [ACCENT_RED, PANEL_BG, ACCENT_GREEN], N=256
        )
        vmax = min(10, monthly_vals.abs().max().max())
        im = ax.imshow(monthly_vals.values, aspect="auto", cmap=cmap,
                       vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(monthly_vals.columns)))
        ax.set_xticklabels(monthly_vals.columns, fontsize=7, color=MUTED_TEXT)
        ax.set_yticks(range(len(monthly_vals.index)))
        ax.set_yticklabels(monthly_vals.index, fontsize=7, color=MUTED_TEXT)

        for i in range(len(monthly_vals.index)):
            for j in range(len(monthly_vals.columns)):
                val = monthly_vals.iloc[i, j]
                if not np.isnan(val):
                    txt_color = TEXT_COLOR if abs(val) > vmax * 0.5 else MUTED_TEXT
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=5.5, color=txt_color)

        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(
            colors=MUTED_TEXT, labelsize=7
        )
        _set_ax_style(ax, title="Monthly Returns Heatmap (%)")
        ax.spines[:].set_visible(False)
        ax.grid(False)
    except Exception:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=MUTED_TEXT)
        _set_ax_style(ax, title="Monthly Returns Heatmap (%)")


def _plot_return_distribution(ax, strat_returns, bench_returns):
    bins = 60
    strat_clean = strat_returns.dropna() * 100
    bench_clean = bench_returns.reindex(strat_returns.index, fill_value=0).dropna() * 100

    ax.hist(bench_clean, bins=bins, color=BENCH_COLOR, alpha=0.4, label="Benchmark", density=True)
    ax.hist(strat_clean, bins=bins, color=STRAT_COLOR, alpha=0.5, label="Strategy", density=True)

    # Normal overlay
    mu, sig = strat_clean.mean(), strat_clean.std()
    x = np.linspace(strat_clean.min(), strat_clean.max(), 200)
    norm_pdf = (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    ax.plot(x, norm_pdf, color=MUTED_TEXT, lw=1.0, ls="--", label="Normal fit")

    ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT_COLOR, facecolor=PANEL_BG)
    _set_ax_style(ax, title="Daily Return Distribution", xlabel="Return (%)", ylabel="Density")


def _plot_rolling_vol(ax, strat_returns, bench_returns, window=63):
    vol_strat = strat_returns.rolling(window).std() * np.sqrt(252) * 100
    vol_bench = bench_returns.reindex(strat_returns.index, fill_value=0).rolling(window).std() * np.sqrt(252) * 100

    ax.plot(vol_strat.index, vol_strat, color=STRAT_COLOR, lw=1.2, label="Strategy")
    ax.plot(vol_bench.index, vol_bench, color=BENCH_COLOR, lw=1.0, ls="--", label="Benchmark", alpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT_COLOR, facecolor=PANEL_BG)
    _set_ax_style(ax, title=f"Rolling {window}D Annualised Volatility (%)", ylabel="Volatility (%)")


def _plot_metrics_table(ax, metrics_df):
    ax.axis("off")
    strat_col = metrics_df.columns[0]
    bench_col = metrics_df.columns[1] if len(metrics_df.columns) > 1 else ""

    col_labels = ["Metric", strat_col, bench_col]
    rows = [[idx, str(metrics_df.loc[idx, strat_col]),
             str(metrics_df.loc[idx, bench_col]) if bench_col else ""]
            for idx in metrics_df.index]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(PANEL_BG if row % 2 == 0 else DARK_BG)
        cell.set_edgecolor(GRID_COLOR)
        cell.set_text_props(color=TEXT_COLOR if row > 0 else ACCENT_BLUE)
        if row == 0:
            cell.set_facecolor("#1c2128")
            cell.set_text_props(fontweight="bold", color=ACCENT_BLUE)

    ax.set_title("Performance Summary", fontsize=10, fontweight="bold",
                 color=TEXT_COLOR, pad=10)


def _plot_turnover(ax, turnover_history):
    if turnover_history is None or turnover_history.empty:
        ax.text(0.5, 0.5, "No turnover data", transform=ax.transAxes,
                ha="center", color=MUTED_TEXT)
        _set_ax_style(ax, title="Portfolio Turnover")
        return

    ax.bar(turnover_history.index, turnover_history * 100,
           color=ACCENT_PURPLE, alpha=0.7, width=20)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    _set_ax_style(ax, title="Monthly Portfolio Turnover (%)", ylabel="Turnover (%)")


# ── Main tearsheet ────────────────────────────────────────────────────────────

def generate_tearsheet(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    turnover_history: pd.Series = None,
    strat_name: str = "Momentum (12-1m)",
    bench_name: str = "SPY Benchmark",
    split_date: str = None,
    output_path: str = "results/tearsheet.png",
) -> None:
    """
    Generate and save the full performance tearsheet.

    Parameters
    ----------
    strat_returns : pd.Series
        Daily strategy returns.
    bench_returns : pd.Series
        Daily benchmark returns.
    turnover_history : pd.Series, optional
        Monthly portfolio turnover (indexed by rebalance date).
    strat_name : str
        Strategy label.
    bench_name : str
        Benchmark label.
    split_date : str, optional
        If provided, draws a vertical line separating in/out-of-sample periods.
    output_path : str
        File path for the saved tearsheet PNG.
    """
    metrics_df = compute_full_metrics(
        strat_returns, bench_returns, strat_name, bench_name
    )

    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "text.color": TEXT_COLOR,
        "font.family": "DejaVu Sans",
        "font.size": 9,
    })

    fig = plt.figure(figsize=(20, 24), facecolor=DARK_BG)

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.985, "Momentum Factor Strategy — Backtest Tearsheet",
             ha="center", va="top", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.975,
             f"12-1 Month Price Momentum | "
             f"In-sample: {strat_returns.index[0].strftime('%b %Y')} — "
             f"{strat_returns.index[-1].strftime('%b %Y')} | "
             f"Monthly Rebalancing | TC: 10bps one-way",
             ha="center", va="top", fontsize=9, color=MUTED_TEXT)

    # ── Key stats strip ───────────────────────────────────────────────────────
    ann_ret = annualised_return(strat_returns) * 100
    ann_vol = annualised_volatility(strat_returns) * 100
    sr = sharpe_ratio(strat_returns)
    mdd = max_drawdown(strat_returns) * 100
    kpis = [
        (f"{ann_ret:+.1f}%", "Ann. Return"),
        (f"{ann_vol:.1f}%", "Ann. Volatility"),
        (f"{sr:.2f}x", "Sharpe Ratio"),
        (f"{mdd:.1f}%", "Max Drawdown"),
    ]
    kpi_colors = [
        ACCENT_GREEN if ann_ret > 0 else ACCENT_RED,
        ACCENT_AMBER,
        ACCENT_BLUE if sr > 1 else ACCENT_AMBER,
        ACCENT_RED,
    ]
    x_positions = [0.13, 0.37, 0.63, 0.87]
    for (val, label), col, xp in zip(kpis, kpi_colors, x_positions):
        fig.text(xp, 0.957, val, ha="center", fontsize=22, fontweight="bold", color=col)
        fig.text(xp, 0.946, label, ha="center", fontsize=8, color=MUTED_TEXT)

    # ── Grid layout ───────────────────────────────────────────────────────────
    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        top=0.93, bottom=0.04,
        hspace=0.38, wspace=0.12,
        left=0.06, right=0.97,
    )
    gs2 = gridspec.GridSpec(
        2, 3,
        figure=fig,
        top=0.48, bottom=0.04,
        hspace=0.42, wspace=0.3,
        left=0.06, right=0.97,
    )

    # Row 0: Cumulative + Drawdown
    ax_cum = fig.add_subplot(gs[0, :])
    _plot_cumulative(ax_cum, strat_returns, bench_returns, strat_name, bench_name)

    ax_dd = fig.add_subplot(gs[1, :])
    _plot_drawdown(ax_dd, strat_returns, bench_returns)

    # In/Out-of-sample split line
    if split_date:
        split_ts = pd.Timestamp(split_date)
        for ax in [ax_cum, ax_dd]:
            ax.axvline(split_ts, color=ACCENT_PURPLE, lw=1.2, ls="--", alpha=0.8)
            ax.text(split_ts, ax.get_ylim()[1] * 0.98, " OOS →",
                    color=ACCENT_PURPLE, fontsize=7.5, va="top")

    # Row 2: Rolling Sharpe + Rolling Vol
    ax_rs = fig.add_subplot(gs[2, 0])
    _plot_rolling_sharpe(ax_rs, strat_returns, bench_returns)

    ax_vol = fig.add_subplot(gs[2, 1])
    _plot_rolling_vol(ax_vol, strat_returns, bench_returns)

    # Row 3: Distribution + Turnover
    ax_dist = fig.add_subplot(gs[3, 0])
    _plot_return_distribution(ax_dist, strat_returns, bench_returns)

    ax_turn = fig.add_subplot(gs[3, 1])
    _plot_turnover(ax_turn, turnover_history)

    # Bottom rows: Heatmap + Metrics table
    ax_heat = fig.add_subplot(gs2[0:2, 0:2])
    _plot_monthly_heatmap(ax_heat, strat_returns)

    ax_table = fig.add_subplot(gs2[0:2, 2])
    _plot_metrics_table(ax_table, metrics_df)

    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[tearsheet] Saved → {output_path}")
