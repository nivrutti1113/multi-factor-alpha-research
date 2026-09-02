"""Statistical Significance & Overfitting Test Module.

Computes Deflated Sharpe Ratio (DSR) per Bailey & López de Prado (2014) and
Stationary Block Bootstrap confidence intervals for out-of-sample Sharpe Ratio.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import logger


def compute_probabilistic_sharpe_ratio(
    returns: pd.Series, sr_benchmark: float = 0.0, periods_per_year: int = 252
) -> float:
    """Computes Probabilistic Sharpe Ratio (PSR) taking non-normality into account.

    Args:
        returns: Daily strategy return series.
        sr_benchmark: Annualized benchmark Sharpe Ratio to test against (e.g. 0.0).
        periods_per_year: Periods per year (252).

    Returns:
        float: PSR probability value between 0.0 and 1.0.
    """
    clean_rets: pd.Series = returns.dropna()
    N: int = len(clean_rets)
    if N < 10:
        return 0.0

    sr_daily: float = float(clean_rets.mean() / clean_rets.std(ddof=1))
    sr_bench_daily: float = sr_benchmark / np.sqrt(periods_per_year)

    skew: float = float(clean_rets.skew())
    kurt: float = float(clean_rets.kurtosis() + 3.0)

    sr_var: float = (1.0 - skew * sr_daily + ((kurt - 1.0) / 4.0) * (sr_daily**2)) / (N - 1)
    sr_std: float = float(np.sqrt(max(sr_var, 1e-8)))

    z_stat: float = (sr_daily - sr_bench_daily) / sr_std
    psr: float = float(stats.norm.cdf(z_stat))
    return psr


def compute_deflated_sharpe_ratio(
    returns: pd.Series,
    num_trials: int = 10,
    std_sr_trials: float = 0.5,
    periods_per_year: int = 252,
) -> Tuple[float, float, float]:
    """Computes Deflated Sharpe Ratio (DSR) to correct for multiple testing overfitting.

    Args:
        returns: Daily strategy return series.
        num_trials: Estimated number of strategy variations backtested (default 10).
        std_sr_trials: Standard deviation of Sharpe ratios across backtested trials.
        periods_per_year: Periods per year (252).

    Returns:
        Tuple[float, float, float]:
            - dsr_pvalue: DSR probability value (Probability that SR > expected max SR under null).
            - expected_max_sr: Annualized expected maximum Sharpe ratio under the null hypothesis.
            - observed_sr: Annualized observed Sharpe ratio of the strategy.
    """
    clean_rets: pd.Series = returns.dropna()
    N: int = len(clean_rets)
    if N < 10:
        return 0.0, 0.0, 0.0

    sr_daily: float = float(clean_rets.mean() / clean_rets.std(ddof=1))
    observed_sr: float = sr_daily * np.sqrt(periods_per_year)

    euler: float = 0.5772156649015328

    if num_trials > 1:
        exp_max_sr_daily: float = (
            std_sr_trials
            / np.sqrt(periods_per_year)
            * (
                (1.0 - euler) * stats.norm.ppf(1.0 - 1.0 / num_trials)
                + euler * stats.norm.ppf(1.0 - 1.0 / (num_trials * np.e))
            )
        )
    else:
        exp_max_sr_daily = 0.0

    expected_max_sr_ann: float = exp_max_sr_daily * np.sqrt(periods_per_year)

    dsr_prob: float = compute_probabilistic_sharpe_ratio(
        returns, sr_benchmark=expected_max_sr_ann, periods_per_year=periods_per_year
    )

    logger.info(
        f"Deflated Sharpe Ratio (DSR): Observed SR={observed_sr:.3f}, DSR Prob={dsr_prob*100:.2f}%"
    )
    return dsr_prob, expected_max_sr_ann, observed_sr


def bootstrap_sharpe_confidence_interval(
    returns: pd.Series,
    num_bootstraps: int = 2500,
    block_size: int = 21,
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Computes Stationary Block Bootstrap confidence interval for Sharpe Ratio.

    Args:
        returns: Daily strategy return series.
        num_bootstraps: Number of bootstrap resamples (default 2,500).
        block_size: Contiguous block length in trading days (21 ~ 1 month).
        confidence_level: Confidence level (e.g. 0.95 for 95% CI).
        periods_per_year: Periods per year (252).
        random_state: Random seed for reproducibility.

    Returns:
        Dict[str, Any]: Dict containing mean_bootstrap_sr, ci_lower, ci_upper, and bootstrap_distribution.

    Raises:
        ValueError: If series length is smaller than block_size.
    """
    np.random.seed(random_state)
    clean_rets: np.ndarray = returns.dropna().values
    n_obs: int = len(clean_rets)

    if n_obs < block_size:
        logger.error("Series length is smaller than block_size.")
        raise ValueError("Series length is smaller than block_size.")

    bootstrap_srs: List[float] = []
    num_blocks: int = int(np.ceil(n_obs / block_size))

    for _ in range(num_bootstraps):
        start_indices: np.ndarray = np.random.randint(0, n_obs - block_size + 1, size=num_blocks)
        resampled_blocks: List[np.ndarray] = [
            clean_rets[idx : idx + block_size] for idx in start_indices
        ]
        resampled_rets: np.ndarray = np.concatenate(resampled_blocks)[:n_obs]

        std: float = float(np.std(resampled_rets, ddof=1))
        if std > 1e-8:
            sr: float = float((np.mean(resampled_rets) / std) * np.sqrt(periods_per_year))
            bootstrap_srs.append(sr)

    bs_arr: np.ndarray = np.array(bootstrap_srs)
    alpha: float = (1.0 - confidence_level) / 2.0
    ci_lower: float = float(np.percentile(bs_arr, alpha * 100))
    ci_upper: float = float(np.percentile(bs_arr, (1.0 - alpha) * 100))
    mean_bs_sr: float = float(np.mean(bs_arr))

    logger.info(
        f"Stationary Block Bootstrap (2,500 iterations): 95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]"
    )

    return {
        "mean_bootstrap_sr": mean_bs_sr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_distribution": bs_arr,
    }
