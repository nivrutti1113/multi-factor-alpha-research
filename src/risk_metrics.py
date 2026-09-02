"""Risk & Out-of-Sample Performance Metrics Module.

Computes institutional quantitative risk metrics: Sharpe, Sortino, Calmar, Max Drawdown,
Hit Rate, Skewness, Kurtosis, and Cumulative Return metrics.
"""


import numpy as np
import pandas as pd

from src.utils.logger import logger


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Computes cumulative return series from daily returns.

    Args:
        returns: Daily return series.

    Returns:
        pd.Series: Cumulative return growth series starting at 1.0.
    """
    return (1.0 + returns).cumprod()


def compute_sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Computes Annualized Sharpe Ratio.

    Args:
        returns: Daily return series.
        rf: Daily risk-free interest rate (default 0.0).
        periods_per_year: Number of periods per year (252 for daily data).

    Returns:
        float: Annualized Sharpe Ratio.
    """
    excess_ret: pd.Series = returns - rf
    std: float = float(excess_ret.std(ddof=1))
    if std <= 0 or np.isnan(std):
        return 0.0
    return float((excess_ret.mean() / std) * np.sqrt(periods_per_year))


def compute_sortino_ratio(
    returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252
) -> float:
    """Computes Annualized Sortino Ratio considering downside volatility.

    Args:
        returns: Daily return series.
        rf: Daily risk-free interest rate (default 0.0).
        periods_per_year: Number of periods per year (252).

    Returns:
        float: Annualized Sortino Ratio.
    """
    excess_ret: pd.Series = returns - rf
    downside_ret: np.ndarray = np.minimum(excess_ret, 0.0)
    downside_std: float = float(np.sqrt(np.mean(downside_ret**2)))

    if downside_std <= 0 or np.isnan(downside_std):
        return 0.0
    return float((excess_ret.mean() / downside_std) * np.sqrt(periods_per_year))


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Computes underwater drawdown time series.

    Args:
        returns: Daily return series.

    Returns:
        pd.Series: Drawdown series in negative percentages (e.g. -0.15 for -15%).
    """
    cum_ret: pd.Series = compute_cumulative_returns(returns)
    running_max: pd.Series = cum_ret.cummax()
    drawdown: pd.Series = (cum_ret - running_max) / running_max
    return drawdown


def compute_max_drawdown(returns: pd.Series) -> float:
    """Computes Maximum Drawdown magnitude as a negative float.

    Args:
        returns: Daily return series.

    Returns:
        float: Maximum drawdown magnitude (e.g. -0.224 for -22.4%).
    """
    dd: pd.Series = compute_drawdown_series(returns)
    return float(dd.min())


def compute_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Computes Calmar Ratio (Annualized Return / |Max Drawdown|).

    Args:
        returns: Daily return series.
        periods_per_year: Number of periods per year (252).

    Returns:
        float: Calmar Ratio.
    """
    ann_ret: float = float(returns.mean() * periods_per_year)
    max_dd: float = abs(compute_max_drawdown(returns))
    if max_dd <= 0:
        return 0.0
    return float(ann_ret / max_dd)


def compute_hit_rate(returns: pd.Series) -> float:
    """Computes percentage of positive return periods.

    Args:
        returns: Return series.

    Returns:
        float: Hit rate ratio between 0.0 and 1.0.
    """
    valid: pd.Series = returns.dropna()
    if len(valid) == 0:
        return 0.0
    return float((valid > 0).mean())


def compute_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """Converts daily return series into monthly compounded return series.

    Args:
        daily_returns: Daily return series.

    Returns:
        pd.Series: Monthly compounded return series.
    """
    monthly: pd.Series = (1.0 + daily_returns).groupby(pd.Grouper(freq="ME")).prod() - 1.0
    return monthly


def compute_full_performance_summary(
    returns: pd.Series, name: str = "Strategy", periods_per_year: int = 252
) -> pd.Series:
    """Computes comprehensive quantitative risk and performance metric summary.

    Args:
        returns: Daily return series.
        name: Strategy name label for Series output.
        periods_per_year: Periods per year (252 for daily data).

    Returns:
        pd.Series: Metric summary containing Sharpe, Sortino, Calmar, Max DD, Win Rate, etc.
    """
    cum_ret: pd.Series = compute_cumulative_returns(returns)
    total_ret: float = float(cum_ret.iloc[-1] - 1.0) if len(cum_ret) > 0 else 0.0

    ann_ret: float = float(returns.mean() * periods_per_year)
    ann_vol: float = float(returns.std(ddof=1) * np.sqrt(periods_per_year))

    sharpe: float = compute_sharpe_ratio(returns, periods_per_year=periods_per_year)
    sortino: float = compute_sortino_ratio(returns, periods_per_year=periods_per_year)
    max_dd: float = compute_max_drawdown(returns)
    calmar: float = compute_calmar_ratio(returns, periods_per_year=periods_per_year)

    daily_hit: float = compute_hit_rate(returns)
    monthly_rets: pd.Series = compute_monthly_returns(returns)
    monthly_hit: float = compute_hit_rate(monthly_rets)

    skew: float = float(returns.skew())
    kurt: float = float(returns.kurtosis())

    logger.debug(
        f"Computed performance metrics for '{name}': Ann Ret={ann_ret*100:.2f}%, Sharpe={sharpe:.3f}"
    )

    return pd.Series(
        {
            "Total Return": f"{total_ret * 100:.2f}%",
            "Annualized Return": f"{ann_ret * 100:.2f}%",
            "Annualized Volatility": f"{ann_vol * 100:.2f}%",
            "Sharpe Ratio": round(sharpe, 3),
            "Sortino Ratio": round(sortino, 3),
            "Calmar Ratio": round(calmar, 3),
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Daily Hit Rate": f"{daily_hit * 100:.2f}%",
            "Monthly Hit Rate": f"{monthly_hit * 100:.2f}%",
            "Skewness": round(skew, 3),
            "Kurtosis": round(kurt, 3),
            "Num Days": len(returns),
        },
        name=name,
    )
