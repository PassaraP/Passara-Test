"""
Data Loader Module
Downloads and cleans historical price data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple


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
    """Return tickers affected by Thai-related stress scenarios."""
    return [
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        'THD', 'THB=X'
    ]


def download_prices(tickers: List[str], years: int = 10) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    print(f"Downloading {len(tickers)} tickers from {start_date.date()} to {end_date.date()}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers

    return prices


def clean_data(prices: pd.DataFrame, missing_threshold: float = 0.1) -> pd.DataFrame:
    """Remove columns with too many NAs and forward-fill remaining."""
    missing_pct = prices.isnull().sum() / len(prices)
    valid_cols = missing_pct[missing_pct <= missing_threshold].index.tolist()

    removed = set(prices.columns) - set(valid_cols)
    if removed:
        print(f"Removed tickers with >{missing_threshold*100:.0f}% missing: {removed}")

    cleaned = prices[valid_cols].ffill().bfill().dropna()
    return cleaned


def remove_outliers(returns: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
    """Replace outliers with median using z-score method."""
    cleaned = returns.copy()

    for col in cleaned.columns:
        data = cleaned[col]
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > z_threshold
        if outliers.any():
            cleaned.loc[outliers, col] = data.median()
            print(f"  Replaced {outliers.sum()} outliers in {col}")

    return cleaned


def load_data(tickers: List[str] = None, years: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data loading pipeline.
    Returns: (prices, cleaned_returns)
    """
    if tickers is None:
        tickers = get_default_tickers()

    prices = download_prices(tickers, years)
    prices = clean_data(prices)

    # Calculate log returns
    returns = np.log(prices / prices.shift(1)).dropna()

    print("Cleaning outliers...")
    returns = remove_outliers(returns)

    print(f"\nLoaded {len(prices.columns)} tickers, {len(returns)} observations")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    return prices, returns
