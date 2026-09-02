"""Survivorship Bias Comparison Script with Explicit Diagnostics.

Runs the 12-1 momentum backtest under both modes (Survivorship Biased vs Point-in-Time Corrected)
and outputs a side-by-side comparative breakdown with explicit ticker diagnostics and Shumway (-30%) delisting alerts.
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np

# Add root directory to path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import fetch_market_data, DEFAULT_UNIVERSE
from src.universe_construction import load_historical_changes, get_delisting_events
from src.backtest_engine import run_momentum_backtest
from src.risk_metrics import compute_full_performance_summary
from src.factor_regression import run_factor_regression
from src.significance_test import compute_deflated_sharpe_ratio
from src.utils.logger import logger


def run_survivorship_bias_comparison() -> pd.DataFrame:
    """Executes comparative survivorship bias backtests and prints explicit diagnostic reports.

    Returns:
        pd.DataFrame: Comparative performance metrics table across Mode A and Mode B.
    """
    print("=" * 85)
    print(" SURVIVORSHIP BIAS DIAGNOSTICS & COMPARISON REPORT")
    print("=" * 85)

    config_path: str = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "start_date": "2014-01-01",
            "end_date": "2024-12-31",
            "cost_bps": 10.0,
            "delisting_return_pct": -0.30,
        }

    # 1. Ingest market data with force_reload to apply fresh delisting injections
    logger.info("Fetching market data including historical delistings...")
    market_data = fetch_market_data(
        start_date=config.get("start_date", "2014-01-01"),
        end_date=config.get("end_date", "2024-12-31"),
        cache_dir="data",
        force_reload=True,  # Force reload to incorporate updated delisting events
        include_historical_delistings=True,
        delisting_return_pct=config.get("delisting_return_pct", -0.30),
    )

    all_prices: pd.DataFrame = market_data["adj_close"]
    delist_events: dict = get_delisting_events("data/sp500_historical_changes.csv")
    mode_a_tickers: list = [c for c in DEFAULT_UNIVERSE if c in all_prices.columns]
    mode_b_tickers: list = all_prices.columns.tolist()

    delisted_added_tickers: list = [t for t in delist_events.keys() if t in mode_b_tickers]

    # Print Explicit Diagnostics
    print("\n" + "-" * 85)
    print(" DIAGNOSTIC UNIVERSE BREAKDOWN")
    print("-" * 85)
    print(f"- Total Tickers in Mode A (Current S&P 500 Only): {len(mode_a_tickers)}")
    print(f"- Total Tickers in Mode B (Point-in-Time Corrected): {len(mode_b_tickers)}")
    print(f"- Historical Delisted / Removed Tickers Added in Mode B: {len(delisted_added_tickers)}")

    # Printed Table of Delisted Tickers
    delist_rows = []
    for t in delisted_added_tickers:
        delist_rows.append(
            {
                "Ticker": t,
                "Removal Date": delist_events[t].strftime("%Y-%m-%d"),
                "Assumed Delisting Return": f"{config.get('delisting_return_pct', -0.30) * 100:.1f}% (Shumway 1997)",
            }
        )

    delist_df = pd.DataFrame(delist_rows)
    print("\n[Mode B Delisted / Removed Tickers Manifest]")
    print(delist_df.to_string(index=False))

    # Warning Check
    if len(delisted_added_tickers) < 10:
        print("\n" + "!" * 85)
        print(" WARNING: Fewer than 10 delisted tickers were successfully added to Mode B!")
        print(" Survivorship bias correction may be underestimated due to missing price feeds.")
        print("!" * 85)

    # 2. Mode A Backtest (Biased)
    logger.info("Executing Mode A Backtest (Survivorship Biased)...")
    res_biased = run_momentum_backtest(
        data=market_data,
        start_year=2015,
        end_year=2024,
        cost_bps=config.get("cost_bps", 10.0),
        use_survivorship_bias_correction=False,
    )

    # 3. Mode B Backtest (Point-in-Time Corrected + Shumway -30%)
    logger.info("Executing Mode B Backtest (Point-in-Time Corrected + Shumway -30%)...")
    res_corrected = run_momentum_backtest(
        data=market_data,
        start_year=2015,
        end_year=2024,
        cost_bps=config.get("cost_bps", 10.0),
        use_survivorship_bias_correction=True,
    )

    spy_rets = res_corrected["spy_returns"]

    # 4. Compute Metrics
    summary_biased = compute_full_performance_summary(
        res_biased["net_returns"], name="Mode A: Biased"
    )
    summary_corrected = compute_full_performance_summary(
        res_corrected["net_returns"], name="Mode B: Point-in-Time Corrected"
    )
    summary_spy = compute_full_performance_summary(spy_rets, name="SPY Benchmark")

    capm_biased = run_factor_regression(res_biased["net_returns"], spy_rets)
    capm_corrected = run_factor_regression(res_corrected["net_returns"], spy_rets)

    dsr_biased, _, _ = compute_deflated_sharpe_ratio(res_biased["net_returns"], num_trials=10)
    dsr_corrected, _, _ = compute_deflated_sharpe_ratio(res_corrected["net_returns"], num_trials=10)

    metrics_list = [
        (
            "Annualized Return",
            summary_biased["Annualized Return"],
            summary_corrected["Annualized Return"],
            summary_spy["Annualized Return"],
        ),
        (
            "Annualized Volatility",
            summary_biased["Annualized Volatility"],
            summary_corrected["Annualized Volatility"],
            summary_spy["Annualized Volatility"],
        ),
        (
            "Sharpe Ratio",
            summary_biased["Sharpe Ratio"],
            summary_corrected["Sharpe Ratio"],
            summary_spy["Sharpe Ratio"],
        ),
        (
            "Sortino Ratio",
            summary_biased["Sortino Ratio"],
            summary_corrected["Sortino Ratio"],
            summary_spy["Sortino Ratio"],
        ),
        (
            "Calmar Ratio",
            summary_biased["Calmar Ratio"],
            summary_corrected["Calmar Ratio"],
            summary_spy["Calmar Ratio"],
        ),
        (
            "Max Drawdown",
            summary_biased["Max Drawdown"],
            summary_corrected["Max Drawdown"],
            summary_spy["Max Drawdown"],
        ),
        (
            "Daily Hit Rate",
            summary_biased["Daily Hit Rate"],
            summary_corrected["Daily Hit Rate"],
            summary_spy["Daily Hit Rate"],
        ),
        (
            "Monthly Hit Rate",
            summary_biased["Monthly Hit Rate"],
            summary_corrected["Monthly Hit Rate"],
            summary_spy["Monthly Hit Rate"],
        ),
        (
            "CAPM Alpha (Annual)",
            capm_biased["summary_table"]["Annualized Alpha"],
            capm_corrected["summary_table"]["Annualized Alpha"],
            "0.00%",
        ),
        (
            "Alpha t-stat (HAC)",
            capm_biased["summary_table"]["Alpha t-stat (HAC)"],
            capm_corrected["summary_table"]["Alpha t-stat (HAC)"],
            "N/A",
        ),
        (
            "Market Beta",
            capm_biased["summary_table"]["Market Beta"],
            capm_corrected["summary_table"]["Market Beta"],
            "1.000",
        ),
        (
            "Deflated Sharpe Ratio (DSR)",
            f"{dsr_biased * 100:.2f}%",
            f"{dsr_corrected * 100:.2f}%",
            "N/A",
        ),
    ]

    comp_df = pd.DataFrame(
        metrics_list,
        columns=[
            "Metric",
            "Mode A (Survivorship Biased)",
            "Mode B (Point-in-Time Corrected)",
            "SPY Benchmark",
        ],
    ).set_index("Metric")

    print("\n" + "=" * 85)
    print(" SIDE-BY-SIDE SURVIVORSHIP BIAS COMPARISON TABLE")
    print("=" * 85)
    print(comp_df.to_string())

    print("\n" + "=" * 85)
    print(" QUANTIFIED SURVIVORSHIP BIAS INFLATION SUMMARY")
    print("=" * 85)
    print(
        f"- Sharpe Ratio Inflation Delta: {summary_biased['Sharpe Ratio'] - summary_corrected['Sharpe Ratio']:+.3f}"
    )
    print(
        f"- Max Drawdown Divergence: {abs(float(summary_corrected['Max Drawdown'].replace('%',''))) - abs(float(summary_biased['Max Drawdown'].replace('%',''))):+.2f}%"
    )
    print(
        f"- Annualized Alpha Inflation Delta: {float(capm_biased['summary_table']['Annualized Alpha'].replace('%','')) - float(capm_corrected['summary_table']['Annualized Alpha'].replace('%','')):+.2f}%"
    )
    print("=" * 85)

    print("\n" + "-" * 85)
    print(" DOCUMENTED DATA LIMITATIONS")
    print("-" * 85)
    print(
        "1. yfinance API Delisting Limitations: Free yfinance endpoints drop tickers after delisting."
    )
    print(
        "   For tickers without historical price data, synthetic pre-delisting price trajectories"
    )
    print("   were generated with an explicit -30% Shumway (1997) delisting shock on removal date.")
    print("2. Delisting Return Fixed Assumption: Realized delisting returns in production vary")
    print("   between corporate actions (M&A vs OTC bankruptcy liquidation). The -30% rule")
    print("   serves as a standardized academic proxy.")
    print("-" * 85)

    return comp_df


if __name__ == "__main__":
    run_survivorship_bias_comparison()
