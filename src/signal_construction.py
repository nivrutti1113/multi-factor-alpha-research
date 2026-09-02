"""Signal Construction Module: Jegadeesh-Titman 12-1 Momentum & Earnings Yield Value Factor.

Computes 12-1 momentum signals and Earnings Yield (E/P = 1 / P/E) Value factor signals,
ranking stocks cross-sectionally into dollar-neutral quintiles.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.logger import logger


def compute_momentum_12_1(
    prices: pd.DataFrame, lookback_days: int = 252, skip_days: int = 21
) -> pd.DataFrame:
    """Computes 12-1 Momentum Signal per Jegadeesh-Titman (1993).

    Signal at time t:
        Signal(t) = P(t - skip_days) / P(t - lookback_days) - 1.0

    Skipping the most recent month (21 trading days) eliminates short-term reversal contamination.

    Args:
        prices: Wide DataFrame of Adjusted Close prices (dates x tickers).
        lookback_days: Total lookback window in trading days (252 ~ 12 months).
        skip_days: Recent window to skip in trading days (21 ~ 1 month).

    Returns:
        pd.DataFrame: DataFrame of raw momentum signals (dates x tickers).

    Raises:
        ValueError: If lookback_days is less than or equal to skip_days.
    """
    if lookback_days <= skip_days:
        logger.error("lookback_days must be strictly greater than skip_days.")
        raise ValueError("lookback_days must be strictly greater than skip_days.")

    p_recent: pd.DataFrame = prices.shift(skip_days)
    p_past: pd.DataFrame = prices.shift(lookback_days)

    signal: pd.DataFrame = (p_recent / p_past) - 1.0
    logger.debug(f"Computed 12-1 Momentum signal across {len(prices)} dates.")
    return signal


def compute_value_signal_ep(prices: pd.DataFrame, earnings_yield: pd.DataFrame) -> pd.DataFrame:
    """Computes Earnings Yield (E/P = 1 / P/E) Value factor signal.

    Higher Earnings Yield represents undervalued/cheap stocks (Value factor).

    Args:
        prices: Wide DataFrame of Adjusted Close prices (dates x tickers).
        earnings_yield: Wide DataFrame of Earnings Yield metrics (dates x tickers).

    Returns:
        pd.DataFrame: Value factor signal DataFrame (dates x tickers).
    """
    # Align indices and columns
    signal: pd.DataFrame = earnings_yield.reindex(index=prices.index, columns=prices.columns)
    logger.debug(f"Computed Earnings Yield Value signal across {len(prices)} dates.")
    return signal


def compute_quantile_weights(
    signal_df: pd.DataFrame, num_quantiles: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ranks stocks cross-sectionally into quintiles and generates dollar-neutral weights.

    Args:
        signal_df: DataFrame of signal values (dates x tickers).
        num_quantiles: Number of cross-sectional groups (default 5 for quintiles).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - quantile_ranks: DataFrame of quintile ranks (1=bottom, 5=top).
            - portfolio_weights: DataFrame of dollar-neutral weights (+1/N_long Q5, -1/N_short Q1).
    """

    def _rank_row(row: pd.Series) -> pd.Series:
        valid_mask: pd.Series = row.dropna()
        if len(valid_mask) < num_quantiles:
            return pd.Series(np.nan, index=row.index)

        ranks: pd.Series = pd.qcut(valid_mask, q=num_quantiles, labels=False, duplicates="drop") + 1
        res: pd.Series = pd.Series(np.nan, index=row.index)
        res[valid_mask.index] = ranks
        return res

    quantile_ranks: pd.DataFrame = signal_df.apply(_rank_row, axis=1)
    weights: pd.DataFrame = pd.DataFrame(0.0, index=signal_df.index, columns=signal_df.columns)

    for date in signal_df.index:
        row_ranks: pd.Series = quantile_ranks.loc[date]
        long_mask: pd.Series = row_ranks == num_quantiles
        short_mask: pd.Series = row_ranks == 1

        n_long: int = int(long_mask.sum())
        n_short: int = int(short_mask.sum())

        if n_long > 0 and n_short > 0:
            weights.loc[date, long_mask] = 1.0 / n_long
            weights.loc[date, short_mask] = -1.0 / n_short

    logger.debug(f"Generated cross-sectional quintile weights ({num_quantiles} quantiles).")
    return quantile_ranks, weights


def standardize_signal_zscore(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Computes cross-sectional Z-score for signals (mean=0, std=1 across tickers per date).

    Args:
        signal_df: DataFrame of raw signals (dates x tickers).

    Returns:
        pd.DataFrame: Cross-sectionally standardized z-scores.
    """
    mean: pd.Series = signal_df.mean(axis=1)
    std: pd.Series = signal_df.std(axis=1).replace(0, np.nan)
    return signal_df.sub(mean, axis=0).div(std, axis=0)
