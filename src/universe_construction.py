"""Point-in-Time Universe Construction Module.

Tracks exact S&P 500 index constituent membership on every rebalance date
using documented historical additions and removals to eliminate survivorship bias.
"""

import os
from typing import Dict, List, Optional

import pandas as pd

from src.utils.logger import logger


def load_historical_changes(csv_path: str = "data/sp500_historical_changes.csv") -> pd.DataFrame:
    """Loads documented historical S&P 500 additions and removals dataset.

    Args:
        csv_path: Path to CSV file containing historical changes.

    Returns:
        pd.DataFrame: DataFrame sorted by Date with columns ['Date', 'Action', 'Ticker', 'Security'].

    Raises:
        FileNotFoundError: If csv_path does not exist on disk.
    """
    if not os.path.exists(csv_path):
        logger.error(f"Historical changes file not found at '{csv_path}'.")
        raise FileNotFoundError(f"Historical changes file not found at '{csv_path}'.")

    df: pd.DataFrame = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    logger.debug(f"Loaded {len(df)} historical index membership change records.")
    return df


def get_all_historical_tickers(
    current_universe: List[str], csv_path: str = "data/sp500_historical_changes.csv"
) -> List[str]:
    """Retrieves union of current tickers plus all historically added/removed tickers.

    Args:
        current_universe: List of current S&P 500 stock tickers.
        csv_path: Path to historical changes CSV.

    Returns:
        List[str]: Combined sorted list of all current and historical tickers.
    """
    changes_df: pd.DataFrame = load_historical_changes(csv_path)
    hist_tickers: List[str] = changes_df["Ticker"].unique().tolist()
    combined: List[str] = sorted(list(set(current_universe + hist_tickers)))
    logger.debug(f"Combined ticker count (Current + Historical): {len(combined)}")
    return combined


def build_point_in_time_mask(
    dates: pd.DatetimeIndex,
    tickers: List[str],
    current_universe: List[str],
    csv_path: str = "data/sp500_historical_changes.csv",
) -> pd.DataFrame:
    """Constructs point-in-time boolean matrix (dates x tickers) indicating index membership.

    Rules:
    1. A stock in current_universe is active, UNLESS added on date T_add (inactive prior to T_add).
    2. A historical ticker removed on T_remove is active prior to T_remove, inactive on/after T_remove.
    3. A historical ticker added on T_add and removed on T_remove is active strictly in [T_add, T_remove).

    Args:
        dates: DatetimeIndex of trading days.
        tickers: List of all stock tickers to evaluate.
        current_universe: List of tickers in current S&P 500 universe.
        csv_path: Path to historical changes CSV.

    Returns:
        pd.DataFrame: Boolean DataFrame (dates x tickers) where True = active constituent on date t.
    """
    changes_df: pd.DataFrame = load_historical_changes(csv_path)

    # Group additions and removals by ticker
    adds_by_ticker: Dict[str, pd.Timestamp] = (
        changes_df[changes_df["Action"] == "ADD"].groupby("Ticker")["Date"].min().to_dict()
    )
    removes_by_ticker: Dict[str, pd.Timestamp] = (
        changes_df[changes_df["Action"] == "REMOVE"].groupby("Ticker")["Date"].min().to_dict()
    )

    mask_df: pd.DataFrame = pd.DataFrame(True, index=dates, columns=tickers)

    for ticker in tickers:
        add_date: Optional[pd.Timestamp] = adds_by_ticker.get(ticker, None)
        remove_date: Optional[pd.Timestamp] = removes_by_ticker.get(ticker, None)

        if ticker in current_universe:
            if add_date is not None:
                mask_df.loc[mask_df.index < add_date, ticker] = False
        else:
            if remove_date is not None:
                mask_df.loc[mask_df.index >= remove_date, ticker] = False
            if add_date is not None:
                mask_df.loc[mask_df.index < add_date, ticker] = False

    logger.info(
        f"Constructed Point-in-Time universe mask across {len(dates)} dates and {len(tickers)} tickers."
    )
    return mask_df


def get_delisting_events(
    csv_path: str = "data/sp500_historical_changes.csv",
) -> Dict[str, pd.Timestamp]:
    """Extracts removal/delisting dates for historical tickers.

    Args:
        csv_path: Path to historical changes CSV.

    Returns:
        Dict[str, pd.Timestamp]: Mapping of ticker symbol to removal date.
    """
    changes_df: pd.DataFrame = load_historical_changes(csv_path)
    removes: pd.DataFrame = changes_df[changes_df["Action"] == "REMOVE"]
    return dict(zip(removes["Ticker"], removes["Date"]))
