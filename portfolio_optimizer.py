"""
Portfolio Optimizer Module

Implements mean-variance portfolio optimization using scipy.
Supports long-only constraints and various optimization objectives.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, List


def calculate_portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252
) -> Tuple[float, float, float]:
    """
    Calculate annualized portfolio statistics.

    Args:
        weights: Portfolio weights
        mean_returns: Mean daily returns
        cov_matrix: Covariance matrix of returns
        trading_days: Number of trading days per year

    Returns:
        Tuple of (annualized_return, annualized_volatility, sharpe_ratio)
    """
    portfolio_return = np.sum(mean_returns * weights) * trading_days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0

    return portfolio_return, portfolio_std, sharpe_ratio


def negative_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252
) -> float:
    """Negative Sharpe ratio for minimization."""
    _, _, sharpe = calculate_portfolio_stats(weights, mean_returns, cov_matrix, trading_days)
    return -sharpe


def portfolio_volatility(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = 252
) -> float:
    """Portfolio volatility for minimization."""
    _, std, _ = calculate_portfolio_stats(weights, mean_returns, cov_matrix, trading_days)
    return std


def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = 'max_sharpe',
    target_return: Optional[float] = None,
    trading_days: int = 252
) -> Dict:
    """
    Optimize portfolio weights using mean-variance optimization.

    Args:
        returns: DataFrame of asset returns
        objective: 'max_sharpe' or 'min_variance'
        target_return: Target annual return (for efficient frontier)
        trading_days: Number of trading days per year

    Returns:
        Dictionary with optimal weights, returns, volatility, and Sharpe ratio
    """
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)

    initial_weights = np.array([1/n_assets] * n_assets)

    bounds = tuple((0, 1) for _ in range(n_assets))

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(mean_returns * w) * trading_days - target_return
        })

    if objective == 'max_sharpe':
        objective_func = negative_sharpe_ratio
    else:
        objective_func = portfolio_volatility

    result = minimize(
        objective_func,
        initial_weights,
        args=(mean_returns, cov_matrix, trading_days),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")

    optimal_weights = result.x
    optimal_weights = optimal_weights / optimal_weights.sum()

    ann_return, ann_vol, sharpe = calculate_portfolio_stats(
        optimal_weights, mean_returns, cov_matrix, trading_days
    )

    return {
        'weights': dict(zip(returns.columns, optimal_weights.tolist())),
        'expected_return': ann_return,
        'volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'success': result.success
    }


def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculate the efficient frontier.

    Args:
        returns: DataFrame of asset returns
        n_points: Number of points on the frontier
        trading_days: Number of trading days per year

    Returns:
        DataFrame with return and volatility for each frontier point
    """
    mean_returns = returns.mean().values
    min_return = mean_returns.min() * trading_days
    max_return = mean_returns.max() * trading_days

    target_returns = np.linspace(min_return * 1.1, max_return * 0.9, n_points)

    frontier = []
    for target in target_returns:
        try:
            result = optimize_portfolio(
                returns,
                objective='min_variance',
                target_return=target,
                trading_days=trading_days
            )
            if result['success']:
                frontier.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe': result['sharpe_ratio']
                })
        except Exception:
            continue

    return pd.DataFrame(frontier)


def backtest_portfolio(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    initial_value: float = 100.0
) -> pd.DataFrame:
    """
    Backtest portfolio performance on historical data.

    Args:
        returns: DataFrame of asset returns
        weights: Dictionary of asset weights
        initial_value: Starting portfolio value

    Returns:
        DataFrame with portfolio values and returns
    """
    weight_series = pd.Series(weights)
    aligned_weights = weight_series.reindex(returns.columns).fillna(0)

    portfolio_returns = (returns * aligned_weights.values).sum(axis=1)

    portfolio_value = initial_value * np.exp(portfolio_returns.cumsum())

    backtest = pd.DataFrame({
        'portfolio_return': portfolio_returns,
        'cumulative_return': portfolio_returns.cumsum(),
        'portfolio_value': portfolio_value
    }, index=returns.index)

    return backtest


def calculate_backtest_metrics(backtest: pd.DataFrame, trading_days: int = 252) -> Dict:
    """
    Calculate backtest performance metrics.

    Args:
        backtest: DataFrame from backtest_portfolio
        trading_days: Number of trading days per year

    Returns:
        Dictionary of performance metrics
    """
    returns = backtest['portfolio_return']

    total_return = backtest['cumulative_return'].iloc[-1]
    n_days = len(returns)
    ann_return = total_return * trading_days / n_days

    ann_vol = returns.std() * np.sqrt(trading_days)

    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    rolling_max = backtest['portfolio_value'].expanding().max()
    drawdowns = backtest['portfolio_value'] / rolling_max - 1
    max_drawdown = drawdowns.min()

    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(trading_days)
    sortino = ann_return / downside_std if downside_std > 0 else 0

    return {
        'total_return': float(np.exp(total_return) - 1),
        'annualized_return': float(ann_return),
        'annualized_volatility': float(ann_vol),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_drawdown),
        'start_date': backtest.index[0].strftime('%Y-%m-%d'),
        'end_date': backtest.index[-1].strftime('%Y-%m-%d'),
        'trading_days': n_days
    }


def get_portfolio_returns_series(
    backtest: pd.DataFrame
) -> List[Dict]:
    """
    Convert backtest to a list of date-value pairs for JSON output.

    Args:
        backtest: DataFrame from backtest_portfolio

    Returns:
        List of dictionaries with date and value
    """
    result = []
    for date, row in backtest.iterrows():
        result.append({
            'date': date.strftime('%Y-%m-%d'),
            'daily_return': float(row['portfolio_return']),
            'cumulative_return': float(row['cumulative_return']),
            'portfolio_value': float(row['portfolio_value'])
        })
    return result


def backtest_benchmark(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    initial_value: float = 100.0
) -> pd.DataFrame:
    """
    Backtest benchmark performance on the same period as portfolio returns.

    Args:
        returns: DataFrame of asset returns (used for date alignment)
        benchmark_returns: Series of benchmark returns
        initial_value: Starting portfolio value

    Returns:
        DataFrame with benchmark values and returns
    """
    aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
    benchmark_value = initial_value * np.exp(aligned_benchmark.cumsum())

    backtest = pd.DataFrame({
        'benchmark_return': aligned_benchmark,
        'cumulative_return': aligned_benchmark.cumsum(),
        'benchmark_value': benchmark_value
    }, index=returns.index)

    return backtest


def get_individual_stock_performance(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    initial_value: float = 100.0
) -> Dict[str, List[Dict]]:
    """
    Calculate individual stock performance based on portfolio allocation.

    Args:
        returns: DataFrame of asset returns
        weights: Dictionary of asset weights
        initial_value: Starting portfolio value

    Returns:
        Dictionary mapping ticker to list of date-value pairs
    """
    result = {}
    for ticker, weight in weights.items():
        if ticker in returns.columns and weight > 0.001:
            stock_returns = returns[ticker]
            allocated_value = initial_value * weight
            stock_value = allocated_value * np.exp(stock_returns.cumsum())

            stock_data = []
            for date, value in stock_value.items():
                stock_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'value': float(value),
                    'return': float(stock_returns.loc[date])
                })
            result[ticker] = stock_data

    return result


if __name__ == "__main__":
    from data_loader import load_and_prepare_data, get_default_tickers

    tickers = get_default_tickers()
    prices, returns, cleaned_returns = load_and_prepare_data(tickers, years=10)

    print("\n" + "="*60)
    print("Optimizing portfolio for maximum Sharpe ratio...")
    result = optimize_portfolio(cleaned_returns, objective='max_sharpe')

    print(f"\nOptimal Portfolio:")
    print(f"  Expected Annual Return: {result['expected_return']:.2%}")
    print(f"  Annual Volatility: {result['volatility']:.2%}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")

    print("\nOptimal Weights:")
    for ticker, weight in sorted(result['weights'].items(), key=lambda x: -x[1]):
        if weight > 0.001:
            print(f"  {ticker}: {weight:.2%}")

    print("\n" + "="*60)
    print("Running backtest...")
    backtest = backtest_portfolio(cleaned_returns, result['weights'])
    metrics = calculate_backtest_metrics(backtest)

    print(f"\nBacktest Results:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
