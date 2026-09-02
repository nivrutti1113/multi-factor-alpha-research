"""
Information Coefficient (IC) Analysis Module.

Computes Spearman Rank Information Coefficient between signal and forward returns
across multiple horizons (1, 3, 6, 12 months) and analyzes alpha decay.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_forward_returns(
    prices: pd.DataFrame, horizons_days: List[int] = [21, 63, 126, 252]
) -> Dict[int, pd.DataFrame]:
    """
    Compute forward returns for specified trading day horizons.

    Forward return at day t for horizon h:
        R_fwd(t, h) = P(t + h) / P(t) - 1.0

    Parameters
    ----------
    prices : pd.DataFrame
        Wide DataFrame of prices (dates x tickers).
    horizons_days : list of int
        List of forward horizons in trading days (e.g. 21=1M, 63=3M, 126=6M, 252=12M).

    Returns
    -------
    dict of int -> pd.DataFrame
        Dictionary mapping horizon days to forward return DataFrames (dates x tickers).
    """
    fwd_returns = {}
    for h in horizons_days:
        # Shift back by h days so index t gets price at t+h
        fwd_price = prices.shift(-h)
        fwd_returns[h] = (fwd_price / prices) - 1.0
    return fwd_returns


def compute_period_ic(signal_series: pd.Series, fwd_return_series: pd.Series) -> float:
    """
    Compute Spearman rank correlation for a single period across stocks.

    Parameters
    ----------
    signal_series : pd.Series
        Signal values for all tickers on a given date.
    fwd_return_series : pd.Series
        Forward returns for all tickers on a given date.

    Returns
    -------
    float
        Spearman rank correlation coefficient (or NaN if insufficient data).
    """
    combined = pd.DataFrame({"signal": signal_series, "fwd_ret": fwd_return_series}).dropna()
    if len(combined) < 5:
        return np.nan

    corr, _ = stats.spearmanr(combined["signal"], combined["fwd_ret"])
    return float(corr)


def compute_ic_series(
    signals: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """
    Compute time series of Information Coefficient (IC) on specified dates.

    Parameters
    ----------
    signals : pd.DataFrame
        Signal DataFrame (dates x tickers).
    fwd_returns : pd.DataFrame
        Forward return DataFrame for a specific horizon (dates x tickers).
    rebalance_dates : pd.DatetimeIndex, optional
        Specific dates (e.g., month-end) to evaluate IC. Defaults to all dates.

    Returns
    -------
    pd.Series
        Time series of Spearman rank IC values.
    """
    if rebalance_dates is not None:
        valid_dates = signals.index.intersection(rebalance_dates)
        sig_sub = signals.loc[valid_dates]
        fwd_sub = fwd_returns.loc[valid_dates]
    else:
        sig_sub = signals
        fwd_sub = fwd_returns

    ic_list = []
    dates_list = []

    for date in sig_sub.index:
        ic_val = compute_period_ic(sig_sub.loc[date], fwd_sub.loc[date])
        ic_list.append(ic_val)
        dates_list.append(date)

    return pd.Series(ic_list, index=dates_list, name="IC").dropna()


def analyze_ic_decay(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    horizons_days: Dict[str, int] = {"1M": 21, "3M": 63, "6M": 126, "12M": 252},
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Analyze Information Coefficient across multiple forward horizons to quantify alpha decay.

    Parameters
    ----------
    signals : pd.DataFrame
        Signal DataFrame (dates x tickers).
    prices : pd.DataFrame
        Price DataFrame (dates x tickers).
    horizons_days : dict of str -> int
        Mapping of horizon label to trading days (e.g. {'1M': 21, '3M': 63, ...}).
    rebalance_dates : pd.DatetimeIndex, optional
        Dates on which signal rebalancing occurs.

    Returns
    -------
    tuple of (pd.DataFrame, dict of str -> pd.Series)
        - summary_df: Summary metrics table (Mean IC, Std IC, IC IR, t-stat, % Positive).
        - ic_series_dict: Dictionary mapping horizon label to IC time-series.
    """
    fwd_dict = compute_forward_returns(prices, horizons_days=list(horizons_days.values()))

    summary_rows = []
    ic_series_dict = {}

    for label, h_days in horizons_days.items():
        fwd_ret = fwd_dict[h_days]
        ic_series = compute_ic_series(signals, fwd_ret, rebalance_dates=rebalance_dates)
        ic_series_dict[label] = ic_series

        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        n_periods = len(ic_series)
        ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
        t_stat = mean_ic / (std_ic / np.sqrt(n_periods)) if std_ic > 0 and n_periods > 0 else np.nan
        pos_ratio = (ic_series > 0).mean()

        summary_rows.append(
            {
                "Horizon": label,
                "Trading Days": h_days,
                "Mean IC": mean_ic,
                "Std IC": std_ic,
                "IC IR": ic_ir,
                "t-statistic": t_stat,
                "Positive IC %": pos_ratio,
                "Num Periods": n_periods,
            }
        )

    summary_df = pd.DataFrame(summary_rows).set_index("Horizon")
    return summary_df, ic_series_dict


if __name__ == "__main__":
    # Smoke test
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    tickers = [f"STK_{i}" for i in range(20)]
    prices = pd.DataFrame(
        np.random.randn(300, 20).cumsum(axis=0) + 100, index=dates, columns=tickers
    )

    from signal_construction import compute_momentum_12_1

    sig = compute_momentum_12_1(prices, lookback_days=252, skip_days=21)

    summary, ic_series = analyze_ic_decay(sig, prices)
    print("IC Summary Table:\n", summary)
