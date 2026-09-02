"""Driver script to run the full quantitative pipeline end-to-end and display key output metrics."""

import os
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath("."))

from src.backtest_engine import (
    get_monthly_rebalance_dates,
    run_momentum_backtest,
)
from src.data_loader import fetch_market_data
from src.factor_regression import run_factor_regression
from src.ic_analysis import analyze_ic_decay
from src.risk_metrics import compute_full_performance_summary
from src.signal_construction import compute_momentum_12_1
from src.significance_test import (
    bootstrap_sharpe_confidence_interval,
    compute_deflated_sharpe_ratio,
)


def main():
    print("=" * 70)
    print(
        " QUANTITATIVE RESEARCH PIPELINE: 12-1 MOMENTUM WITH SHORT-TERM REVERSAL"
    )
    print("=" * 70)

    # 1. Load Market Data
    print("\n[Step 1] Ingesting & Cleaning Market Data (2014-2024)...")
    market_data = fetch_market_data(
        start_date="2014-01-01", end_date="2024-12-31", cache_dir="data"
    )
    adj_close = market_data["adj_close"]
    spy_df = market_data["spy"]

    # 2. Signal Construction
    print("\n[Step 2] Constructing Jegadeesh-Titman 12-1 Momentum Signal...")
    signal_df = compute_momentum_12_1(
        adj_close, lookback_days=252, skip_days=21
    )

    # 3. Information Coefficient Decay Analysis
    print(
        "\n[Step 3] Computing Spearman Information Coefficient (IC) & Decay Across Horizons..."
    )
    rebal_dates = get_monthly_rebalance_dates(adj_close.index)
    rebal_dates = rebal_dates[
        (rebal_dates.year >= 2015) & (rebal_dates.year <= 2024)
    ]

    ic_summary, ic_dict = analyze_ic_decay(
        signals=signal_df,
        prices=adj_close,
        horizons_days={"1M": 21, "3M": 63, "6M": 126, "12M": 252},
        rebalance_dates=rebal_dates,
    )
    print(ic_summary)

    # 4. Walk-Forward Backtest Execution (Mode A: Fixed Universe)
    print(
        "\n[Step 4] Running Walk-Forward Monthly Rebalancing Backtest (10 bps Cost, t+1 Execution)..."
    )
    bt_results = run_momentum_backtest(
        data=market_data,
        start_year=2015,
        end_year=2024,
        lookback_days=252,
        skip_days=21,
        num_quantiles=5,
        cost_bps=10.0,
        use_survivorship_bias_correction=False,
    )

    net_rets = bt_results["net_returns"]
    gross_rets = bt_results["gross_returns"]
    spy_rets = bt_results["spy_returns"]

    # 5. Out-of-Sample Performance Summary
    print("\n[Step 5] Out-of-Sample Risk & Performance Metrics:")
    net_summary = compute_full_performance_summary(
        net_rets, name="Strategy Net (10bps)"
    )
    gross_summary = compute_full_performance_summary(
        gross_rets, name="Strategy Gross"
    )
    spy_summary = compute_full_performance_summary(
        spy_rets, name="SPY Benchmark"
    )

    perf_df = pd.DataFrame([net_summary, gross_summary, spy_summary])
    print(perf_df.to_string())

    # 6. CAPM Factor Regression
    print(
        "\n[Step 6] CAPM Factor Regression (Newey-West HAC Robust Standard Errors):"
    )
    factor_res = run_factor_regression(net_rets, spy_rets, max_lags=5)
    print(factor_res["summary_table"].to_string())

    # 7. Deflated Sharpe Ratio & Bootstrap CI
    print("\n[Step 7] Overfitting Control & Deflated Sharpe Ratio (DSR):")
    dsr_prob, exp_max_sr, obs_sr = compute_deflated_sharpe_ratio(
        net_rets, num_trials=10
    )
    bs_res = bootstrap_sharpe_confidence_interval(
        net_rets, num_bootstraps=2500, block_size=21
    )

    print(f"Observed Out-of-Sample Sharpe: {obs_sr:.3f}")
    print(f"Expected Max Sharpe Under Null (10 Trials): {exp_max_sr:.3f}")
    print(f"Deflated Sharpe Ratio (DSR) Probability: {dsr_prob * 100:.2f}%")
    print(
        f"Bootstrap 95% Confidence Interval: [{bs_res['ci_lower']:.3f}, {bs_res['ci_upper']:.3f}]"
    )

    print("\n" + "=" * 70)
    print(" PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
