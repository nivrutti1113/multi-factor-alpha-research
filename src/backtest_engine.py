"""Walk-Forward Backtest Engine Module.

Executes expanding-window, monthly rebalanced walk-forward backtests with
t+1 Open price execution, 10 bps transaction cost modeling, point-in-time index
membership tracking, and Shumway (1997) -30% delisting return adjustments.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import DEFAULT_UNIVERSE
from src.signal_construction import compute_momentum_12_1, compute_value_signal_ep
from src.universe_construction import build_point_in_time_mask
from src.utils.logger import logger


def get_monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Finds month-end trading dates (last available trading day of each month).

    Args:
        index: DatetimeIndex of trading days.

    Returns:
        pd.DatetimeIndex: Dates representing the last trading day of each month.
    """
    df_idx: pd.DataFrame = pd.DataFrame(index=index)
    df_idx["year"] = index.year
    df_idx["month"] = index.month

    rebal_dates: pd.Series = df_idx.groupby(["year", "month"]).apply(
        lambda x: x.index[-1]
    )
    return pd.DatetimeIndex(rebal_dates.values)


def run_strategy_backtest(
    data: Dict[str, pd.DataFrame],
    signal_type: str = "momentum",
    custom_signal_df: Optional[pd.DataFrame] = None,
    start_year: int = 2015,
    end_year: int = 2024,
    lookback_days: int = 252,
    skip_days: int = 21,
    num_quantiles: int = 5,
    cost_bps: float = 10.0,
    use_survivorship_bias_correction: bool = True,
) -> Dict[str, pd.Series]:
    """Executes monthly walk-forward backtest for Momentum, Value, or Combined strategy signals.

    Args:
        data: Market data dict containing 'adj_close', 'open', 'close', 'spy', 'asset_returns', 'delisting_returns'.
        signal_type: Type of signal ('momentum', 'value', 'custom'). Defaults to 'momentum'.
        custom_signal_df: Optional pre-constructed signal DataFrame when signal_type='custom'.
        start_year: Start year for out-of-sample evaluation (2015).
        end_year: End year for out-of-sample evaluation (2024).
        lookback_days: Momentum lookback in trading days (252).
        skip_days: Recent window skipped in trading days (21).
        num_quantiles: Number of quintiles (5).
        cost_bps: Transaction cost per trade in basis points (10.0 bps = 0.0010).
        use_survivorship_bias_correction: If True, applies dynamic point-in-time S&P 500 universe filtering and
            Shumway -30% delisting return adjustments.

    Returns:
        Dict[str, pd.Series]: Portfolio return series:
            - 'gross_returns': Daily gross returns series.
            - 'net_returns': Daily net returns series.
            - 'long_returns': Daily returns of Long side (Q5).
            - 'short_returns': Daily returns of Short side (Q1).
            - 'turnover': Monthly turnover series.
            - 'spy_returns': Benchmark daily return series.
    """
    prices: pd.DataFrame = data["adj_close"].copy()
    spy_df: pd.DataFrame = data["spy"]
    cost_rate: float = cost_bps / 10000.0

    # Filter universe and returns PRIOR to signal construction
    if use_survivorship_bias_correction:
        asset_daily_returns: pd.DataFrame = data.get(
            "delisting_returns", prices.pct_change().fillna(0.0)
        )
        pit_mask: pd.DataFrame = build_point_in_time_mask(
            prices.index, prices.columns.tolist(), DEFAULT_UNIVERSE
        )
    else:
        available_cols: List[str] = [
            c for c in DEFAULT_UNIVERSE if c in prices.columns
        ]
        prices = prices[available_cols]
        asset_daily_returns = prices.pct_change().fillna(0.0)
        pit_mask = pd.DataFrame(
            True, index=prices.index, columns=prices.columns
        )

    # Signal selection on properly scoped prices matrix
    if custom_signal_df is not None:
        signal: pd.DataFrame = custom_signal_df.reindex(
            index=prices.index, columns=prices.columns
        )
    elif signal_type == "value":
        signal = compute_value_signal_ep(prices, data["earnings_yield"])
    else:
        signal = compute_momentum_12_1(
            prices, lookback_days=lookback_days, skip_days=skip_days
        )

    rebal_signal_dates: pd.DatetimeIndex = get_monthly_rebalance_dates(
        prices.index
    )
    rebal_signal_dates = rebal_signal_dates[
        (rebal_signal_dates.year >= start_year)
        & (rebal_signal_dates.year <= end_year)
    ]

    date_to_pos: Dict[pd.Timestamp, int] = {
        d: i for i, d in enumerate(prices.index)
    }
    exec_schedule: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for sig_date in rebal_signal_dates:
        pos: int = date_to_pos[sig_date]
        if pos + 1 < len(prices.index):
            exec_date: pd.Timestamp = prices.index[pos + 1]
            exec_schedule.append((sig_date, exec_date))

    daily_dates: pd.DatetimeIndex = prices.loc[exec_schedule[0][1] :].index
    gross_ret: pd.Series = pd.Series(0.0, index=daily_dates)
    net_ret: pd.Series = pd.Series(0.0, index=daily_dates)
    long_ret: pd.Series = pd.Series(0.0, index=daily_dates)
    short_ret: pd.Series = pd.Series(0.0, index=daily_dates)
    turnover_series: pd.Series = pd.Series(
        0.0, index=[e[1] for e in exec_schedule]
    )

    current_weights: pd.Series = pd.Series(0.0, index=prices.columns)

    for k in range(len(exec_schedule)):
        sig_date, exec_date = exec_schedule[k]

        if k + 1 < len(exec_schedule):
            next_exec_date: pd.Timestamp = exec_schedule[k + 1][1]
            period_dates: pd.DatetimeIndex = prices.loc[
                exec_date:next_exec_date
            ].index[:-1]
        else:
            period_dates = prices.loc[exec_date:].index

        row_signal: pd.Series = signal.loc[sig_date]
        active_mask: pd.Series = pit_mask.loc[sig_date]
        valid_sig: pd.Series = row_signal[active_mask].dropna()

        if len(valid_sig) >= num_quantiles:
            ranks: pd.Series = (
                pd.qcut(
                    valid_sig,
                    q=num_quantiles,
                    labels=False,
                    duplicates="drop",
                )
                + 1
            )
            long_stocks: pd.Index = valid_sig.index[ranks == num_quantiles]
            short_stocks: pd.Index = valid_sig.index[ranks == 1]

            target_weights: pd.Series = pd.Series(0.0, index=prices.columns)
            if len(long_stocks) > 0:
                target_weights[long_stocks] = 1.0 / len(long_stocks)
            if len(short_stocks) > 0:
                target_weights[short_stocks] = -1.0 / len(short_stocks)
        else:
            target_weights = pd.Series(0.0, index=prices.columns)

        turnover: float = float(np.abs(target_weights - current_weights).sum())
        turnover_series.loc[exec_date] = turnover
        rebal_cost: float = turnover * cost_rate

        current_weights = target_weights.copy()

        for d in period_dates:
            r_d: pd.Series = (
                asset_daily_returns.loc[d]
                .reindex(prices.columns)
                .fillna(0.0)
            )

            w_long: pd.Series = np.maximum(current_weights, 0.0)
            w_short: pd.Series = np.minimum(current_weights, 0.0)

            r_long: float = float((w_long * r_d).sum())
            r_short: float = float((w_short * r_d).sum())
            p_gross: float = float((current_weights * r_d).sum())

            p_net: float = p_gross - rebal_cost if d == exec_date else p_gross

            gross_ret.loc[d] = p_gross
            net_ret.loc[d] = p_net
            long_ret.loc[d] = r_long
            short_ret.loc[d] = r_short

            denom: float = 1.0 + p_gross
            if abs(denom) > 1e-6:
                current_weights = current_weights * (1.0 + r_d) / denom

    spy_eval_ret: pd.Series = spy_df["Return"].reindex(daily_dates).fillna(0.0)

    logger.info(
        f"Backtest completed ({signal_type}): {len(net_ret)} daily returns, Avg Turnover: {turnover_series.mean() * 100:.2f}%."
    )

    return {
        "gross_returns": gross_ret,
        "net_returns": net_ret,
        "long_returns": long_ret,
        "short_returns": short_ret,
        "turnover": turnover_series,
        "spy_returns": spy_eval_ret,
    }


def run_momentum_backtest(
    data: Dict[str, pd.DataFrame], **kwargs: Any
) -> Dict[str, pd.Series]:
    """Wrapper function preserving backwards compatibility for momentum backtests."""
    return run_strategy_backtest(data, signal_type="momentum", **kwargs)
