"""
Data Loader Module

Downloads historical price data from Yahoo Finance, cleans the data by removing
NAs and outliers, and prepares it for portfolio optimization.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


def download_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 10
) -> pd.DataFrame:
    """
    Download historical adjusted close prices from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        years: Number of years of historical data if dates not specified

    Returns:
        DataFrame with adjusted close prices for each ticker
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        start = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=years*365)
        start_date = start.strftime('%Y-%m-%d')

    print(f"Downloading data from {start_date} to {end_date}...")

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers

    return prices


def remove_missing_data(
    prices: pd.DataFrame,
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    Remove columns with too many missing values and forward-fill remaining NAs.

    Args:
        prices: DataFrame of prices
        threshold: Maximum fraction of missing values allowed per column

    Returns:
        Cleaned DataFrame
    """
    missing_pct = prices.isnull().sum() / len(prices)
    valid_columns = missing_pct[missing_pct <= threshold].index.tolist()

    removed = set(prices.columns) - set(valid_columns)
    if removed:
        print(f"Removed tickers with >={threshold*100}% missing data: {removed}")

    cleaned = prices[valid_columns].copy()
    cleaned = cleaned.ffill().bfill()
    cleaned = cleaned.dropna()

    return cleaned


def remove_outliers(
    returns: pd.DataFrame,
    z_threshold: float = 4.0
) -> pd.DataFrame:
    """
    Remove outliers from returns data using z-score method.
    Outliers are replaced with the column median.

    Args:
        returns: DataFrame of returns
        z_threshold: Z-score threshold for outlier detection

    Returns:
        DataFrame with outliers replaced
    """
    cleaned = returns.copy()

    for col in cleaned.columns:
        col_data = cleaned[col]
        mean = col_data.mean()
        std = col_data.std()

        if std > 0:
            z_scores = np.abs((col_data - mean) / std)
            outliers = z_scores > z_threshold

            if outliers.any():
                median = col_data.median()
                cleaned.loc[outliers, col] = median
                print(f"Replaced {outliers.sum()} outliers in {col}")

    return cleaned


def calculate_returns(
    prices: pd.DataFrame,
    method: str = 'log'
) -> pd.DataFrame:
    """
    Calculate returns from price data.

    Args:
        prices: DataFrame of prices
        method: 'log' for log returns, 'simple' for simple returns

    Returns:
        DataFrame of returns
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    returns = returns.dropna()
    return returns


def load_and_prepare_data(
    tickers: List[str],
    years: int = 10,
    outlier_threshold: float = 4.0,
    missing_threshold: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete data loading and preparation pipeline.

    Args:
        tickers: List of ticker symbols
        years: Number of years of historical data
        outlier_threshold: Z-score threshold for outlier removal
        missing_threshold: Maximum fraction of missing values

    Returns:
        Tuple of (prices, returns, cleaned_returns)
    """
    prices = download_data(tickers, years=years)
    prices = remove_missing_data(prices, threshold=missing_threshold)
    returns = calculate_returns(prices, method='log')
    cleaned_returns = remove_outliers(returns, z_threshold=outlier_threshold)

    print(f"\nData summary:")
    print(f"  Tickers loaded: {list(prices.columns)}")
    print(f"  Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Observations: {len(prices)}")

    return prices, returns, cleaned_returns


def get_default_tickers() -> List[str]:
    """Return the default universe of tickers."""
    return [
        # Thai Export
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        # Thai Domestic
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        # Global
        'WDC', 'THD',
        # Fixed Income
        'LEMB', 'VWOB', 'EMLC',
        # FX
        'THB=X'
    ]


def get_thai_tickers() -> List[str]:
    """Return tickers that are Thai-related (affected by tariffs)."""
    return [
        # Thai Export (directly affected)
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        # Thai Domestic (indirectly affected)
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        # Thailand ETF
        'THD',
        # Thai Baht
        'THB=X'
    ]


def download_benchmark(
    benchmark_ticker: str = '^GSPC',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 10
) -> Tuple[pd.Series, pd.Series]:
    """
    Download S&P 500 benchmark data.

    Args:
        benchmark_ticker: Ticker symbol for benchmark (default: ^GSPC for S&P 500)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        years: Number of years of historical data

    Returns:
        Tuple of (prices Series, returns Series)
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        start = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=years*365)
        start_date = start.strftime('%Y-%m-%d')

    print(f"Downloading benchmark {benchmark_ticker} from {start_date} to {end_date}...")

    data = yf.download(benchmark_ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'].iloc[:, 0] if len(data['Close'].shape) > 1 else data['Close']
    else:
        prices = data['Close']

    prices = prices.ffill().bfill()
    returns = np.log(prices / prices.shift(1)).dropna()

    return prices, returns


if __name__ == "__main__":
    tickers = get_default_tickers()
    prices, returns, cleaned_returns = load_and_prepare_data(tickers, years=10)
    print("\nSample of cleaned returns:")
    print(cleaned_returns.tail())
