# Cross-Sectional Equity Momentum & Value Multi-Factor Pipeline

[![Quantitative Pipeline CI](https://github.com/nivrutti1113/multi-factor-alpha-research/actions/workflows/tests.yml/badge.svg)](https://github.com/nivrutti1113/multi-factor-alpha-research/actions)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-5%20passed-brightgreen.svg)](tests/test_no_lookahead.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An institutional-grade quantitative research pipeline built in Python evaluating **cross-sectional momentum (Jegadeesh-Titman 12-1)**, **Earnings Yield Value factor (E/P)**, and a **50/50 Combined Multi-Factor Portfolio** across US equities (S&P 500 universe + Historical Delistings, 2015–2024).

---

## 📌 Executive Summary & Out-of-Sample Results (2015–2024)

> [!NOTE]
> An early data-alignment issue in signal-to-return date mapping was identified and corrected during development; results below reflect the corrected, verified pipeline.

### Empirical Multi-Factor Comparison


| Quantitative Metric | Mode A (Survivorship Biased) | Mode B (Point-in-Time Corrected) | SPY Benchmark |
| :--- | :---: | :---: | :---: |
| **Annualized Return** | **1.71%** | **5.28%** | **14.25%** |
| **Annualized Volatility** | **19.55%** | **19.62%** | **17.63%** |
| **Sharpe Ratio** | **0.087** | **0.269** | **0.808** |
| **Sortino Ratio** | **0.117** | **0.361** | **1.129** |
| **Calmar Ratio** | **0.033** | **0.097** | **0.423** |
| **Max Drawdown** | **-51.35%** | **-54.35%** | **-33.72%** |
| **Daily Hit Rate** | **52.55%** | **53.63%** | **54.71%** |
| **Monthly Hit Rate** | **57.14%** | **63.03%** | **69.75%** |
| **CAPM Alpha (Annualized)** | **+0.93%** | **+3.47%** | 0.00% |
| **Alpha $t$-stat (Newey-West)** | **0.156** | **0.583** | N/A |
| **Market Beta ($\beta$)** | **0.055** | **0.127** | 1.000 |
| **Deflated Sharpe Ratio (DSR)** | **1.40%** | **5.29%** | N/A |


> [!TIP]
> **Key Multi-Factor Insight**: Momentum and Value daily returns exhibit strong negative correlation ($r = -0.3039$). Combining both factors in a 50/50 portfolio cuts strategy volatility nearly in half from **19.88%** down to **10.05%** and reduces maximum drawdown from **-52.75%** down to **-42.29%**, demonstrating structural factor diversification benefits.

---

## 🔬 Multi-Factor Framework & Methodology

1. **Signal 1: Jegadeesh-Titman (12-1) Momentum**:
   $$S_{\text{mom}, i}(t) = \frac{P_{i, t-21}}{P_{i, t-252}} - 1$$
   Skipping the most recent month ($t-21$ to $t$) removes short-term reversal contamination.

2. **Signal 2: Earnings Yield Value Factor ($E/P$)**:
   $$S_{\text{val}, i}(t) = \frac{1}{\text{P/E}_i(t)}$$
   Ranks cheap/undervalued stocks into top quintile (Q5, Long) and overvalued stocks into bottom quintile (Q1, Short).

3. **50/50 Combined Multi-Factor Allocation**:
   $$R_{\text{combined}, t} = 0.5 \times R_{\text{momentum}, t} + 0.5 \times R_{\text{value}, t}$$

4. **Point-in-Time Universe & Shumway (-30%) Delisting Rule**:
   - Restricts month-end rebalance selection to actual active S&P 500 constituents as of date $t$ (`src/universe_construction.py`).
   - Applies conservative **-30% delisting return** (Shumway 1997) on historical removal dates for delisted stocks.

---

## 🛡 Production Code Quality & Engineering Standards

- **Centralized Logging (`src/utils/logger.py`)**: Replaced raw print statements with standard Python `logging` module.
- **Type Annotations**: Complete static type hints (`typing.Dict`, `pd.DataFrame`, `pd.Series`, `List`, `Tuple`) across all modules.
- **Google-Style Docstrings**: Standardized `Args:`, `Returns:`, and `Raises:` sections on every function and class.
- **Tooling & Formatting (`pyproject.toml`)**: Formatted to Black line length 100 standards and Ruff lint checked.
- **Continuous Integration (`.github/workflows/tests.yml`)**: Automated GitHub Actions workflow running `pytest -v tests/` across Python 3.10, 3.11, and 3.12.

---

## 📁 Repository Structure

```
├── .github/workflows/
│   └── tests.yml                       # GitHub Actions CI workflow
├── config.yaml                         # Pipeline configuration & survivorship flag
├── data/
│   ├── sp500_historical_changes.csv    # Historical S&P 500 additions & removals (2014-2024)
│   ├── prices_adj_close.parquet        # Point-in-time price cache
│   └── prices_fundamentals.parquet     # Cached Earnings Yield fundamental metrics
├── notebooks/
│   └── momentum_reversal_research.ipynb  # Interactive multi-factor research report
├── pyproject.toml                      # Black, Ruff, and Pytest configuration
├── src/
│   ├── __init__.py
│   ├── universe_construction.py        # Point-in-time constituent mapping & mask generation
│   ├── data_loader.py                  # Ingestion, fundamental ratios, & Shumway -30% delisting
│   ├── signal_construction.py          # 12-1 Momentum & Earnings Yield Value factor signals
│   ├── ic_analysis.py                  # Spearman IC time-series & multi-horizon decay
│   ├── backtest_engine.py              # Walk-forward engine for Momentum, Value, & Combined
│   ├── risk_metrics.py                 # Sharpe, Sortino, Calmar, Max DD, Win Rate summary
│   ├── factor_regression.py            # CAPM Alpha/Beta with Newey-West HAC robust t-stats
│   ├── significance_test.py            # Deflated Sharpe Ratio (DSR) & block bootstrap CI
│   └── utils/
│       ├── __init__.py
│       └── logger.py                   # Centralized logging infrastructure
├── tests/
│   ├── __init__.py
│   └── test_no_lookahead.py            # Pytest assertions for zero lookahead & signals
├── scripts/
│   ├── run_multi_factor_comparison.py  # Momentum vs Value vs 50/50 Combined script
│   ├── compare_survivorship_bias.py    # Survivorship bias comparative backtest script
│   ├── build_notebook.py               # Notebook build script
│   └── run_pipeline.py                 # Main research pipeline runner
├── requirements.txt                    # Project dependencies
└── README.md                           # Recruiter documentation & research summary
```

---

## 🚀 Quickstart & Setup

### 1. Installation
```bash
git clone https://github.com/nivrutti1113/multi-factor-alpha-research.git
cd multi-factor-alpha-research

python -m venv venv
.\venv\Scripts\activate          # Windows (or source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
```

### 2. Run Pytest Suite
```bash
pytest -v tests/
```

### 3. Run Multi-Factor Comparison Script
```bash
python scripts/run_multi_factor_comparison.py
```

### 4. Run Survivorship Bias Comparison Script
```bash
python scripts/compare_survivorship_bias.py
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
