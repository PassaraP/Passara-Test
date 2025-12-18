"""
Portfolio Optimizer Module
Mean-variance optimization using scipy with long-only constraints.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List


def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray,
                          cov_matrix: np.ndarray) -> tuple:
    """Calculate annualized return, volatility, and Sharpe ratio."""
    ret = np.sum(mean_returns * weights) * 252
    vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe


def neg_sharpe(weights: np.ndarray, mean_returns: np.ndarray,
               cov_matrix: np.ndarray) -> float:
    """Negative Sharpe ratio for minimization."""
    _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    return -sharpe


def optimize_portfolio(returns: pd.DataFrame) -> Dict:
    """
    Find optimal portfolio weights maximizing Sharpe ratio.
    Constraint: Long-only (no shorting).
    """
    mean_ret = returns.mean().values
    cov_mat = returns.cov().values
    n = len(returns.columns)

    # Initial equal weights
    w0 = np.ones(n) / n

    # Constraints: weights sum to 1, all weights >= 0 (no shorting)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))

    result = minimize(neg_sharpe, w0, args=(mean_ret, cov_mat),
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000})

    weights = result.x / result.x.sum()  # Normalize
    ret, vol, sharpe = portfolio_performance(weights, mean_ret, cov_mat)

    return {
        'weights': dict(zip(returns.columns, weights.tolist())),
        'expected_return': ret,
        'volatility': vol,
        'sharpe_ratio': sharpe
    }


def backtest_portfolio(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Run historical backtest with given weights."""
    w = pd.Series(weights).reindex(returns.columns).fillna(0)
    portfolio_returns = (returns * w.values).sum(axis=1)

    cumulative = portfolio_returns.cumsum()
    value = 100 * np.exp(cumulative)

    return pd.DataFrame({
        'daily_return': portfolio_returns,
        'cumulative_return': cumulative,
        'portfolio_value': value
    }, index=returns.index)


def calculate_metrics(backtest: pd.DataFrame) -> Dict:
    """Calculate backtest performance metrics."""
    rets = backtest['daily_return']
    values = backtest['portfolio_value']

    total_ret = np.exp(backtest['cumulative_return'].iloc[-1]) - 1
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    rolling_max = values.expanding().max()
    drawdown = values / rolling_max - 1
    max_dd = drawdown.min()

    # Sortino ratio
    neg_rets = rets[rets < 0]
    downside_vol = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else 0
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    return {
        'total_return': float(total_ret),
        'annualized_return': float(ann_ret),
        'annualized_volatility': float(ann_vol),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_dd),
        'start_date': str(backtest.index[0].date()),
        'end_date': str(backtest.index[-1].date()),
        'trading_days': len(backtest)
    }


def backtest_to_list(backtest: pd.DataFrame) -> List[Dict]:
    """Convert backtest DataFrame to list of dicts for JSON."""
    return [
        {
            'date': str(idx.date()),
            'daily_return': float(row['daily_return']),
            'cumulative_return': float(row['cumulative_return']),
            'portfolio_value': float(row['portfolio_value'])
        }
        for idx, row in backtest.iterrows()
    ]
