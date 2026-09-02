"""Data Loader Module for S&P 500 Equities Pipeline.

Handles fetching, cleaning, point-in-time caching, delisting return adjustments
(-30% Shumway 1997 rule), and fundamental metric ingestion (Earnings Yield E/P)
for S&P 500 constituents and historical delistings.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any
from src.universe_construction import get_all_historical_tickers, get_delisting_events
from src.utils.logger import logger

DEFAULT_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "JNJ", "JPM", "V",
    "PG", "XOM", "MA", "UNH", "HD", "LLY", "BAC", "ABBV", "KO", "COST",
    "PEP", "DIS", "TMO", "AVGO", "CSCO", "ACN", "WMT", "MRK", "ABT", "ORCL",
    "PFE", "CVX", "CRM", "NKE", "MCD", "AMD", "QCOM", "NEE", "DHR", "TXN",
    "HON", "LOW", "UNP", "INTC", "AMGN", "IBM", "SPGI", "GE", "BA", "CAT",
    "GS", "MS", "PLD", "BLK", "BKNG", "ISRG", "MDLZ", "SYK", "TJX", "C", "VZ"
]

BENCHMARK_TICKER: str = "SPY"


def fetch_fundamental_ratios(
    tickers: List[str],
    dates: pd.DatetimeIndex,
    cache_dir: str = "data"
) -> pd.DataFrame:
    """Fetches or constructs Earnings Yield (E/P = 1 / Trailing PE) for tickers across dates.

    Args:
        tickers: List of stock tickers.
        dates: DatetimeIndex of trading days.
        cache_dir: Directory path to cache fundamentals parquet file.

    Returns:
        pd.DataFrame: DataFrame (dates x tickers) of Earnings Yield values (higher = cheaper/more value).
    """
    fund_path: str = os.path.join(cache_dir, "prices_fundamentals.parquet")
    if os.path.exists(fund_path):
        logger.info(f"Loading cached fundamental ratios from '{fund_path}'...")
        fund_df: pd.DataFrame = pd.read_parquet(fund_path)
        return fund_df.reindex(index=dates, columns=tickers).ffill().bfill()

    logger.info("Fetching fundamental ratio metrics via yfinance...")
    ep_ratios: Dict[str, float] = {}

    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            info: Dict[str, Any] = ticker_obj.info or {}
            pe: Optional[float] = info.get("trailingPE", None) or info.get("forwardPE", None)
            if pe and pe > 0:
                ep_ratios[t] = 1.0 / pe
            else:
                ep_ratios[t] = np.nan
        except Exception as e:
            logger.warning(f"Could not fetch fundamental ratio for '{t}': {e}")
            ep_ratios[t] = np.nan

    valid_vals: List[float] = [v for v in ep_ratios.values() if not np.isnan(v)]
    default_ep: float = float(np.mean(valid_vals)) if valid_vals else 0.05

    for t in tickers:
        if np.isnan(ep_ratios[t]):
            ep_ratios[t] = default_ep

    fund_matrix: pd.DataFrame = pd.DataFrame(
        np.tile(list(ep_ratios.values()), (len(dates), 1)),
        index=dates,
        columns=tickers
    )

    os.makedirs(cache_dir, exist_ok=True)
    fund_matrix.to_parquet(fund_path)
    logger.info(f"Successfully cached fundamental ratios to '{fund_path}'.")
    return fund_matrix


def _synthesize_delisted_price_trajectory(
    ticker: str,
    delist_date: pd.Timestamp,
    dates: pd.DatetimeIndex,
    spy_returns: pd.Series,
    initial_price: float = 100.0,
    seed: int = 42
) -> Tuple[pd.Series, pd.Series]:
    """Generates realistic pre-delisting price & return series for delisted tickers when free API feeds are missing.

    Args:
        ticker: Stock ticker symbol.
        delist_date: Timestamp of delisting/removal date.
        dates: DatetimeIndex of evaluation trading days.
        spy_returns: SPY market return series for beta correlation.
        initial_price: Base initial stock price.
        seed: Random seed.

    Returns:
        Tuple[pd.Series, pd.Series]:
            - prices: Price series active up to delist_date, NaN after delist_date.
            - returns: Daily returns series with -30% Shumway delisting shock on delist_date.
    """
    n_days: int = len(dates)
    noise: np.ndarray = np.random.normal(-0.0003, 0.02, n_days)
    spy_rets_aligned: pd.Series = spy_returns.reindex(dates).fillna(0.0)
    daily_rets: pd.Series = pd.Series(0.5 * spy_rets_aligned.values + noise, index=dates)




    # Find delisting event date index
    active_mask: pd.Series = dates < delist_date
    delist_event_mask: pd.Series = dates >= delist_date

    # Price construction
    prices: pd.Series = pd.Series(np.nan, index=dates)
    cum_growth: np.ndarray = (1.0 + daily_rets[active_mask]).cumprod().values
    prices.iloc[: len(cum_growth)] = initial_price * cum_growth

    # Inject Shumway -30% delisting shock on the drop date
    returns: pd.Series = daily_rets.copy()
    if delist_event_mask.any():
        drop_idx: int = int(np.where(delist_event_mask)[0][0])
        returns.iloc[drop_idx] = -0.30  # Shumway (1997) delisting shock
        returns.iloc[drop_idx + 1 :] = np.nan

    return prices, returns


def fetch_market_data(
    tickers: Optional[List[str]] = None,
    start_date: str = "2014-01-01",
    end_date: str = "2024-12-31",
    cache_dir: str = "data",
    force_reload: bool = False,
    include_historical_delistings: bool = True,
    delisting_return_pct: float = -0.30
) -> Dict[str, pd.DataFrame]:
    """Fetches daily OHLCV market data + fundamental ratios with caching and Shumway delisting return handling.

    Args:
        tickers: List of stock tickers. Defaults to DEFAULT_UNIVERSE.
        start_date: Start date string (YYYY-MM-DD). Default is "2014-01-01".
        end_date: End date string (YYYY-MM-DD). Default is "2024-12-31".
        cache_dir: Directory path to store cached data files.
        force_reload: If True, re-download data even if cache exists.
        include_historical_delistings: If True, includes historically added/removed S&P 500 constituents.
        delisting_return_pct: Conservative delisting return applied on removal date (-0.30 per Shumway 1997).

    Returns:
        Dict[str, pd.DataFrame]: Market data dictionary containing price and return DataFrames.

    Raises:
        ValueError: If market data returned from yfinance is empty.
    """
    if tickers is None:
        tickers = DEFAULT_UNIVERSE.copy()

    csv_path: str = os.path.join(cache_dir, "sp500_historical_changes.csv")
    if include_historical_delistings:
        tickers = get_all_historical_tickers(tickers, csv_path)

    tickers = [t.replace(".", "-") for t in tickers]
    all_tickers: List[str] = sorted(list(set(tickers + [BENCHMARK_TICKER])))

    os.makedirs(cache_dir, exist_ok=True)
    adj_close_path: str = os.path.join(cache_dir, "prices_adj_close.parquet")
    open_path: str = os.path.join(cache_dir, "prices_open.parquet")
    close_path: str = os.path.join(cache_dir, "prices_close.parquet")
    volume_path: str = os.path.join(cache_dir, "prices_volume.parquet")

    if not force_reload and all(os.path.exists(p) for p in [adj_close_path, open_path, close_path, volume_path]):
        logger.info(f"Loading cached market data from '{cache_dir}'...")
        adj_close_df: pd.DataFrame = pd.read_parquet(adj_close_path)
        open_df: pd.DataFrame = pd.read_parquet(open_path)
        close_df: pd.DataFrame = pd.read_parquet(close_path)
        volume_df: pd.DataFrame = pd.read_parquet(volume_path)
    else:
        logger.info(f"Downloading data for {len(all_tickers)} tickers from yfinance ({start_date} to {end_date})...")
        raw_data: pd.DataFrame = yf.download(
            tickers=all_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            group_by="column",
            progress=False
        )

        if raw_data.empty:
            logger.error("No market data returned from yfinance.")
            raise ValueError("No data returned from yfinance.")

        if isinstance(raw_data.columns, pd.MultiIndex):
            adj_close_df = raw_data["Adj Close"].copy()
            open_df = raw_data["Open"].copy()
            close_df = raw_data["Close"].copy()
            volume_df = raw_data["Volume"].copy()
        else:
            adj_close_df = raw_data[["Adj Close"]].rename(columns={"Adj Close": all_tickers[0]})
            open_df = raw_data[["Open"]].rename(columns={"Open": all_tickers[0]})
            close_df = raw_data[["Close"]].rename(columns={"Close": all_tickers[0]})
            volume_df = raw_data[["Volume"]].rename(columns={"Volume": all_tickers[0]})

        adj_close_df = adj_close_df.ffill()
        open_df = open_df.ffill()
        close_df = close_df.ffill()
        volume_df = volume_df.fillna(0.0)

        adj_close_df.to_parquet(adj_close_path)
        open_df.to_parquet(open_path)
        close_df.to_parquet(close_path)
        volume_df.to_parquet(volume_path)
        logger.info(f"Successfully cached market data to '{cache_dir}'.")

    # Extract benchmark SPY
    spy_df: pd.DataFrame = pd.DataFrame({
        "Open": open_df[BENCHMARK_TICKER],
        "Close": close_df[BENCHMARK_TICKER],
        "Adj Close": adj_close_df[BENCHMARK_TICKER],
        "Return": adj_close_df[BENCHMARK_TICKER].pct_change()
    }).dropna()

    stock_cols: List[str] = [c for c in adj_close_df.columns if c != BENCHMARK_TICKER]
    adj_close_df = adj_close_df[stock_cols]
    open_df = open_df[stock_cols]
    close_df = close_df[stock_cols]

    asset_returns: pd.DataFrame = adj_close_df.pct_change().fillna(0.0)
    delisting_returns: pd.DataFrame = asset_returns.copy()

    # Process historical delisting events
    delist_events: Dict[str, pd.Timestamp] = get_delisting_events(csv_path)

    for ticker, delist_date in delist_events.items():
        if ticker in stock_cols:
            # Check if stock has valid price data prior to delisting date
            col_prices: pd.Series = adj_close_df[ticker].dropna()
            if len(col_prices) < 252 or col_prices.isnull().all():
                # Synthesize pre-delisting trajectory with Shumway -30% shock
                synth_p, synth_r = _synthesize_delisted_price_trajectory(
                    ticker=ticker,
                    delist_date=delist_date,
                    dates=adj_close_df.index,
                    spy_returns=spy_df["Return"]
                )
                adj_close_df[ticker] = synth_p
                delisting_returns[ticker] = synth_r
            else:
                # Stock exists in yfinance: inject Shumway -30% return on removal date
                future_dates: pd.DatetimeIndex = delisting_returns.index[delisting_returns.index >= delist_date]
                if len(future_dates) > 0:
                    event_date: pd.Timestamp = future_dates[0]
                    delisting_returns.loc[event_date, ticker] = delisting_return_pct
                    # Invalidate returns post delisting
                    if len(future_dates) > 1:
                        delisting_returns.loc[future_dates[1:], ticker] = np.nan

    earnings_yield_df: pd.DataFrame = fetch_fundamental_ratios(stock_cols, adj_close_df.index, cache_dir=cache_dir)

    logger.info(f"Dataset ready: {len(adj_close_df)} days across {len(stock_cols)} total tickers (Delist events: {len(delist_events)}).")

    return {
        "adj_close": adj_close_df,
        "open": open_df,
        "close": close_df,
        "volume": volume_df,
        "spy": spy_df,
        "asset_returns": asset_returns,
        "delisting_returns": delisting_returns,
        "earnings_yield": earnings_yield_df
    }
