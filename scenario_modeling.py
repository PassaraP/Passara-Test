"""
Scenario Modeling Module

Implements stress testing and forward-looking scenario simulation using
Geometric Brownian Motion (GBM). Supports customizable stress scenarios
for specific asset subsets.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StressScenario:
    """Configuration for a stress scenario."""
    name: str
    affected_assets: List[str]
    expected_drop_mean: float  # Mean expected drop (e.g., -0.10 for -10%)
    expected_drop_std: float   # Std dev of drop (e.g., 0.02 for +-2%)
    volatility_increase: float # Relative increase in volatility (e.g., 0.10 for +10%)
    start_date: Optional[str] = None  # Start date of stress (YYYY-MM-DD)
    duration_days: int = 252   # Duration in trading days (1 year = 252)


def create_trump_tariff_scenario(
    thai_assets: List[str],
    expected_drop_mean: float = -0.10,
    expected_drop_std: float = 0.02,
    volatility_increase: float = 0.10,
    start_date: Optional[str] = None,
    duration_days: int = 252
) -> StressScenario:
    """
    Create a Trump tariffs stress scenario affecting Thai assets.

    Args:
        thai_assets: List of Thai-related ticker symbols
        expected_drop_mean: Mean expected drop for affected assets
        expected_drop_std: Standard deviation of drop
        volatility_increase: Relative volatility increase
        start_date: Start date of stress scenario
        duration_days: Duration in trading days

    Returns:
        StressScenario configuration
    """
    return StressScenario(
        name="Trump Tariffs Impact",
        affected_assets=thai_assets,
        expected_drop_mean=expected_drop_mean,
        expected_drop_std=expected_drop_std,
        volatility_increase=volatility_increase,
        start_date=start_date,
        duration_days=duration_days
    )


def simulate_gbm(
    initial_price: float,
    mu: float,
    sigma: float,
    n_steps: int,
    n_simulations: int = 1000,
    dt: float = 1/252
) -> np.ndarray:
    """
    Simulate asset prices using Geometric Brownian Motion.

    dS = mu * S * dt + sigma * S * dW

    Args:
        initial_price: Starting price
        mu: Drift (annualized expected return)
        sigma: Volatility (annualized)
        n_steps: Number of time steps
        n_simulations: Number of Monte Carlo paths
        dt: Time step size (1/252 for daily)

    Returns:
        Array of shape (n_simulations, n_steps+1) with simulated prices
    """
    prices = np.zeros((n_simulations, n_steps + 1))
    prices[:, 0] = initial_price

    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_simulations)
        prices[:, t] = prices[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return prices


def apply_stress_shock(
    initial_price: float,
    drop_mean: float,
    drop_std: float,
    n_simulations: int = 1000
) -> np.ndarray:
    """
    Apply initial stress shock to asset price.

    Args:
        initial_price: Pre-stress price
        drop_mean: Mean percentage drop
        drop_std: Standard deviation of drop
        n_simulations: Number of simulations

    Returns:
        Array of post-shock prices for each simulation
    """
    drops = np.random.normal(drop_mean, drop_std, n_simulations)
    drops = np.clip(drops, -0.99, 0.99)
    shocked_prices = initial_price * (1 + drops)
    return shocked_prices


def simulate_stressed_portfolio(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    scenario: StressScenario,
    n_simulations: int = 1000,
    initial_value: float = 100.0
) -> Dict:
    """
    Simulate portfolio performance under a stress scenario.

    Args:
        returns: Historical returns DataFrame
        weights: Portfolio weights
        scenario: Stress scenario configuration
        n_simulations: Number of Monte Carlo simulations
        initial_value: Starting portfolio value

    Returns:
        Dictionary with simulation results
    """
    assets = list(weights.keys())
    n_assets = len(assets)
    n_steps = scenario.duration_days

    historical_mean = returns.mean() * 252
    historical_vol = returns.std() * np.sqrt(252)

    portfolio_paths = np.zeros((n_simulations, n_steps + 1))

    asset_weights = np.array([weights.get(a, 0) for a in assets])
    asset_weights = asset_weights / asset_weights.sum()

    asset_paths = {}
    for i, asset in enumerate(assets):
        base_mu = historical_mean.get(asset, 0)
        base_sigma = historical_vol.get(asset, 0.2)

        if asset in scenario.affected_assets:
            shocked_initial = apply_stress_shock(
                initial_value * asset_weights[i],
                scenario.expected_drop_mean,
                scenario.expected_drop_std,
                n_simulations
            )
            adjusted_sigma = base_sigma * (1 + scenario.volatility_increase)
            adjusted_mu = base_mu * 0.5
        else:
            shocked_initial = np.full(n_simulations, initial_value * asset_weights[i])
            adjusted_sigma = base_sigma
            adjusted_mu = base_mu

        paths = simulate_gbm(
            1.0,
            adjusted_mu,
            adjusted_sigma,
            n_steps,
            n_simulations
        )

        paths = paths * shocked_initial[:, np.newaxis]
        asset_paths[asset] = paths

    for asset in assets:
        portfolio_paths += asset_paths[asset]

    return {
        'portfolio_paths': portfolio_paths,
        'asset_paths': asset_paths,
        'scenario': scenario,
        'n_simulations': n_simulations,
        'n_steps': n_steps
    }


def calculate_simulation_statistics(
    simulation_result: Dict,
    percentiles: List[float] = [5, 25, 50, 75, 95]
) -> Dict:
    """
    Calculate statistics from simulation results.

    Args:
        simulation_result: Output from simulate_stressed_portfolio
        percentiles: Percentiles to calculate

    Returns:
        Dictionary of statistics
    """
    paths = simulation_result['portfolio_paths']
    initial_value = paths[:, 0].mean()
    final_values = paths[:, -1]

    returns = (final_values - initial_value) / initial_value

    stats = {
        'initial_value': float(initial_value),
        'mean_final_value': float(final_values.mean()),
        'std_final_value': float(final_values.std()),
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'min_return': float(returns.min()),
        'max_return': float(returns.max()),
        'percentiles': {}
    }

    for p in percentiles:
        stats['percentiles'][f'p{p}'] = float(np.percentile(returns, p))

    prob_loss = (returns < 0).mean()
    prob_loss_10 = (returns < -0.10).mean()
    prob_loss_20 = (returns < -0.20).mean()

    stats['probability_of_loss'] = float(prob_loss)
    stats['probability_of_10pct_loss'] = float(prob_loss_10)
    stats['probability_of_20pct_loss'] = float(prob_loss_20)

    stats['var_95'] = float(np.percentile(returns, 5))
    stats['cvar_95'] = float(returns[returns <= np.percentile(returns, 5)].mean())

    return stats


def get_simulation_paths_summary(
    simulation_result: Dict,
    n_sample_paths: int = 100
) -> List[Dict]:
    """
    Get summary of simulation paths for JSON output.

    Args:
        simulation_result: Output from simulate_stressed_portfolio
        n_sample_paths: Number of paths to include

    Returns:
        List of path summaries
    """
    paths = simulation_result['portfolio_paths']
    n_steps = simulation_result['n_steps']
    n_sims = min(n_sample_paths, paths.shape[0])

    sample_indices = np.linspace(0, paths.shape[0]-1, n_sims, dtype=int)

    result = []
    for idx in sample_indices:
        path = paths[idx]
        result.append({
            'simulation_id': int(idx),
            'initial_value': float(path[0]),
            'final_value': float(path[-1]),
            'return': float((path[-1] - path[0]) / path[0]),
            'max_value': float(path.max()),
            'min_value': float(path.min()),
            'path_values': [float(v) for v in path[::max(1, n_steps//50)]]
        })

    return result


def get_percentile_paths(
    simulation_result: Dict,
    percentiles: List[float] = [5, 25, 50, 75, 95]
) -> Dict[str, List[float]]:
    """
    Get percentile paths across all simulations.

    Args:
        simulation_result: Output from simulate_stressed_portfolio
        percentiles: Percentiles to calculate

    Returns:
        Dictionary mapping percentile names to path values
    """
    paths = simulation_result['portfolio_paths']

    result = {}
    for p in percentiles:
        percentile_path = np.percentile(paths, p, axis=0)
        result[f'p{p}'] = [float(v) for v in percentile_path]

    result['mean'] = [float(v) for v in paths.mean(axis=0)]

    return result


def run_stress_test(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    scenario: StressScenario,
    n_simulations: int = 1000
) -> Dict:
    """
    Run complete stress test and return comprehensive results.

    Args:
        returns: Historical returns DataFrame
        weights: Portfolio weights
        scenario: Stress scenario configuration
        n_simulations: Number of simulations

    Returns:
        Complete stress test results
    """
    simulation = simulate_stressed_portfolio(
        returns, weights, scenario, n_simulations
    )

    statistics = calculate_simulation_statistics(simulation)
    percentile_paths = get_percentile_paths(simulation)
    sample_paths = get_simulation_paths_summary(simulation)

    return {
        'scenario_name': scenario.name,
        'scenario_config': {
            'affected_assets': scenario.affected_assets,
            'expected_drop_mean': scenario.expected_drop_mean,
            'expected_drop_std': scenario.expected_drop_std,
            'volatility_increase': scenario.volatility_increase,
            'duration_days': scenario.duration_days
        },
        'statistics': statistics,
        'percentile_paths': percentile_paths,
        'sample_paths': sample_paths,
        'n_simulations': n_simulations
    }


if __name__ == "__main__":
    from data_loader import load_and_prepare_data, get_default_tickers, get_thai_tickers
    from portfolio_optimizer import optimize_portfolio

    tickers = get_default_tickers()
    thai_tickers = get_thai_tickers()

    prices, returns, cleaned_returns = load_and_prepare_data(tickers, years=10)

    print("\nOptimizing portfolio...")
    opt_result = optimize_portfolio(cleaned_returns, objective='max_sharpe')

    print("\nCreating stress scenario...")
    scenario = create_trump_tariff_scenario(
        thai_assets=[t for t in thai_tickers if t in cleaned_returns.columns],
        expected_drop_mean=-0.10,
        expected_drop_std=0.02,
        volatility_increase=0.10,
        duration_days=252
    )

    print(f"Running stress test with {1000} simulations...")
    stress_results = run_stress_test(
        cleaned_returns,
        opt_result['weights'],
        scenario,
        n_simulations=1000
    )

    print("\n" + "="*60)
    print("Stress Test Results:")
    print(f"  Scenario: {stress_results['scenario_name']}")
    print(f"  Mean Return: {stress_results['statistics']['mean_return']:.2%}")
    print(f"  Std Return: {stress_results['statistics']['std_return']:.2%}")
    print(f"  VaR (95%): {stress_results['statistics']['var_95']:.2%}")
    print(f"  CVaR (95%): {stress_results['statistics']['cvar_95']:.2%}")
    print(f"  Probability of Loss: {stress_results['statistics']['probability_of_loss']:.2%}")
