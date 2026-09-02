"""Factor Regression Module: CAPM Alpha & Beta Attribution.

Regresses strategy returns against market benchmark (SPY) excess returns
using OLS with Newey-West (HAC) robust standard errors to isolate true alpha.
"""

from typing import Any, Dict

import pandas as pd
import statsmodels.api as sm

from src.utils.logger import logger


def run_factor_regression(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_daily: float = 0.0,
    periods_per_year: int = 252,
    max_lags: int = 5,
) -> Dict[str, Any]:
    """Performs OLS Factor Regression of strategy returns against benchmark returns.

    Model:
        R_strategy(t) - Rf = Alpha + Beta * (R_benchmark(t) - Rf) + epsilon(t)

    Uses Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors.

    Args:
        strategy_returns: Daily net strategy return series.
        benchmark_returns: Daily benchmark (SPY) return series.
        rf_daily: Daily risk-free interest rate (default 0.0).
        periods_per_year: Periods per year (252 for daily data).
        max_lags: Number of lags for Newey-West HAC covariance estimator.

    Returns:
        Dict[str, Any]: Regression output dict containing alpha_annual, alpha_tstat,
            alpha_pvalue, beta, beta_tstat, beta_pvalue, r2, r2_adj, summary_table, and model_fit.
    """
    df: pd.DataFrame = pd.DataFrame(
        {"strategy": strategy_returns, "benchmark": benchmark_returns}
    ).dropna()

    y: pd.Series = df["strategy"] - rf_daily
    x: pd.Series = df["benchmark"] - rf_daily
    X: pd.DataFrame = sm.add_constant(x)

    model = sm.OLS(y, X)
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

    alpha_daily: float = float(results.params["const"])
    alpha_annual: float = alpha_daily * periods_per_year
    alpha_tstat: float = float(results.tvalues["const"])
    alpha_pval: float = float(results.pvalues["const"])

    beta: float = float(results.params["benchmark"])
    beta_tstat: float = float(results.tvalues["benchmark"])
    beta_pval: float = float(results.pvalues["benchmark"])

    r2: float = float(results.rsquared)
    r2_adj: float = float(results.rsquared_adj)

    summary_series: pd.Series = pd.Series(
        {
            "Annualized Alpha": f"{alpha_annual * 100:.2f}%",
            "Alpha t-stat (HAC)": round(alpha_tstat, 3),
            "Alpha p-value": f"{alpha_pval:.4f}",
            "Market Beta": round(beta, 3),
            "Beta t-stat (HAC)": round(beta_tstat, 3),
            "Beta p-value": f"{beta_pval:.4f}",
            "R-Squared": round(r2, 4),
            "Adjusted R-Squared": round(r2_adj, 4),
            "Num Observations": int(results.nobs),
        },
        name="CAPM Factor Attribution",
    )

    logger.info(
        f"CAPM Regression complete: Alpha={alpha_annual*100:.2f}% (t={alpha_tstat:.2f}), Beta={beta:.3f}"
    )

    return {
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_annual,
        "alpha_tstat": alpha_tstat,
        "alpha_pvalue": alpha_pval,
        "beta": beta,
        "beta_tstat": beta_tstat,
        "beta_pvalue": beta_pval,
        "r2": r2,
        "r2_adj": r2_adj,
        "summary_table": summary_series,
        "model_fit": results,
    }
