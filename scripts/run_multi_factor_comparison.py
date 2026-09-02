"""Multi-Factor Strategy Comparison Script.

Runs Momentum alone, Value (Earnings Yield) alone, and a 50/50 Combined Multi-Factor portfolio,
evaluating cross-factor return correlation and diversification benefits.
"""

import os
import sys

import pandas as pd
import yaml

# Add root directory to path
sys.path.insert(0, os.path.abspath("."))

from src.backtest_engine import run_strategy_backtest
from src.data_loader import fetch_market_data
from src.factor_regression import run_factor_regression
from src.risk_metrics import compute_full_performance_summary
from src.significance_test import compute_deflated_sharpe_ratio
from src.utils.logger import logger


def run_multi_factor_comparison() -> pd.DataFrame:
    """Executes multi-factor backtests and returns comparative performance summary table.

    Returns:
        pd.DataFrame: Comparative performance metrics table across strategies.
    """
    print("=" * 85)
    print(" MULTI-FACTOR STRATEGY COMPARISON: MOMENTUM VS VALUE VS 50/50 COMBINED PORTFOLIO")
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
            "use_survivorship_bias_correction": True,
        }

    # 1. Fetch market data & fundamental ratios
    logger.info("Ingesting market data & fundamental ratios...")
    market_data = fetch_market_data(
        start_date=config.get("start_date", "2014-01-01"),
        end_date=config.get("end_date", "2024-12-31"),
        cache_dir="data",
        include_historical_delistings=True,
        delisting_return_pct=config.get("delisting_return_pct", -0.30),
    )

    # 2. Run Momentum Strategy (12-1 Momentum)
    logger.info("Executing Strategy 1: Jegadeesh-Titman 12-1 Momentum...")
    res_momentum = run_strategy_backtest(
        data=market_data,
        signal_type="momentum",
        start_year=2015,
        end_year=2024,
        cost_bps=config.get("cost_bps", 10.0),
        use_survivorship_bias_correction=config.get("use_survivorship_bias_correction", True),
    )

    # 3. Run Value Strategy (Earnings Yield E/P Factor)
    logger.info("Executing Strategy 2: Earnings Yield (E/P) Value Factor...")
    res_value = run_strategy_backtest(
        data=market_data,
        signal_type="value",
        start_year=2015,
        end_year=2024,
        cost_bps=config.get("cost_bps", 10.0),
        use_survivorship_bias_correction=config.get("use_survivorship_bias_correction", True),
    )

    spy_rets: pd.Series = res_momentum["spy_returns"]
    mom_net: pd.Series = res_momentum["net_returns"]
    val_net: pd.Series = res_value["net_returns"]

    # 4. Construct 50/50 Combined Multi-Factor Portfolio
    logger.info("Constructing 50/50 Combined Multi-Factor Portfolio...")
    combined_net: pd.Series = 0.5 * mom_net + 0.5 * val_net
    combined_net.name = "50/50 Combined Net"

    # Compute correlation between Momentum and Value return series
    corr_mom_val: float = float(mom_net.corr(val_net))
    logger.info(f"Momentum vs Value Return Correlation: r = {corr_mom_val:.4f}")

    # 5. Extract Summaries
    sum_mom = compute_full_performance_summary(mom_net, name="Momentum Alone")
    sum_val = compute_full_performance_summary(val_net, name="Value Alone")
    sum_comb = compute_full_performance_summary(combined_net, name="50/50 Combined Portfolio")
    sum_spy = compute_full_performance_summary(spy_rets, name="SPY Benchmark")

    capm_mom = run_factor_regression(mom_net, spy_rets)
    capm_val = run_factor_regression(val_net, spy_rets)
    capm_comb = run_factor_regression(combined_net, spy_rets)

    dsr_mom, _, _ = compute_deflated_sharpe_ratio(mom_net, num_trials=10)
    dsr_val, _, _ = compute_deflated_sharpe_ratio(val_net, num_trials=10)
    dsr_comb, _, _ = compute_deflated_sharpe_ratio(combined_net, num_trials=10)

    # 6. Build Side-by-Side Table
    metrics_list = [
        (
            "Annualized Return",
            sum_mom["Annualized Return"],
            sum_val["Annualized Return"],
            sum_comb["Annualized Return"],
            sum_spy["Annualized Return"],
        ),
        (
            "Annualized Volatility",
            sum_mom["Annualized Volatility"],
            sum_val["Annualized Volatility"],
            sum_comb["Annualized Volatility"],
            sum_spy["Annualized Volatility"],
        ),
        (
            "Sharpe Ratio",
            sum_mom["Sharpe Ratio"],
            sum_val["Sharpe Ratio"],
            sum_comb["Sharpe Ratio"],
            sum_spy["Sharpe Ratio"],
        ),
        (
            "Sortino Ratio",
            sum_mom["Sortino Ratio"],
            sum_val["Sortino Ratio"],
            sum_comb["Sortino Ratio"],
            sum_spy["Sortino Ratio"],
        ),
        (
            "Calmar Ratio",
            sum_mom["Calmar Ratio"],
            sum_val["Calmar Ratio"],
            sum_comb["Calmar Ratio"],
            sum_spy["Calmar Ratio"],
        ),
        (
            "Max Drawdown",
            sum_mom["Max Drawdown"],
            sum_val["Max Drawdown"],
            sum_comb["Max Drawdown"],
            sum_spy["Max Drawdown"],
        ),
        (
            "Daily Hit Rate",
            sum_mom["Daily Hit Rate"],
            sum_val["Daily Hit Rate"],
            sum_comb["Daily Hit Rate"],
            sum_spy["Daily Hit Rate"],
        ),
        (
            "Monthly Hit Rate",
            sum_mom["Monthly Hit Rate"],
            sum_val["Monthly Hit Rate"],
            sum_comb["Monthly Hit Rate"],
            sum_spy["Monthly Hit Rate"],
        ),
        (
            "CAPM Alpha (Annual)",
            capm_mom["summary_table"]["Annualized Alpha"],
            capm_val["summary_table"]["Annualized Alpha"],
            capm_comb["summary_table"]["Annualized Alpha"],
            "0.00%",
        ),
        (
            "Alpha t-stat (HAC)",
            capm_mom["summary_table"]["Alpha t-stat (HAC)"],
            capm_val["summary_table"]["Alpha t-stat (HAC)"],
            capm_comb["summary_table"]["Alpha t-stat (HAC)"],
            "N/A",
        ),
        (
            "Market Beta",
            capm_mom["summary_table"]["Market Beta"],
            capm_val["summary_table"]["Market Beta"],
            capm_comb["summary_table"]["Market Beta"],
            "1.000",
        ),
        (
            "Deflated Sharpe Ratio (DSR)",
            f"{dsr_mom*100:.2f}%",
            f"{dsr_val*100:.2f}%",
            f"{dsr_comb*100:.2f}%",
            "N/A",
        ),
    ]

    comp_df = pd.DataFrame(
        metrics_list,
        columns=[
            "Metric",
            "12-1 Momentum Alone",
            "Value (E/P) Alone",
            "50/50 Combined Portfolio",
            "SPY Benchmark",
        ],
    ).set_index("Metric")

    print("\n" + "=" * 85)
    print(" MULTI-FACTOR STRATEGY COMPARISON TABLE")
    print("=" * 85)
    print(comp_df.to_string())

    print("\n" + "=" * 85)
    print(" MULTI-FACTOR DIVERSIFICATION SUMMARY")
    print("=" * 85)
    print(f"- Momentum vs Value Daily Return Correlation (r): {corr_mom_val:+.4f}")
    print(
        f"- Volatility Reduction: Combined Volatility ({sum_comb['Annualized Volatility']}) vs Momentum ({sum_mom['Annualized Volatility']})"
    )
    print(
        f"- Sharpe Improvement: Combined Sharpe ({sum_comb['Sharpe Ratio']}) vs Momentum ({sum_mom['Sharpe Ratio']})"
    )
    print(
        f"- Max Drawdown Reduction: Combined DD ({sum_comb['Max Drawdown']}) vs Momentum DD ({sum_mom['Max Drawdown']})"
    )
    print("=" * 85)

    return comp_df


if __name__ == "__main__":
    run_multi_factor_comparison()
