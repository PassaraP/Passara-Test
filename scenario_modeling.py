"""
Scenario Modeling Module
Stress testing with customizable scenarios and GBM simulation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class StressScenario:
    """Configuration for a stress scenario."""
    name: str
    affected_assets: List[str]
    drop_mean: float        # e.g., -0.10 for -10%
    drop_std: float         # e.g., 0.02 for ±2%
    vol_increase: float     # e.g., 0.10 for +10% volatility
    start_date: Optional[str] = None
    duration_days: int = 252  # 1 year


def create_tariff_scenario(
    thai_assets: List[str],
    drop_mean: float = -0.10,
    drop_std: float = 0.02,
    vol_increase: float = 0.10,
    duration_days: int = 252
) -> StressScenario:
    """Create Trump tariffs stress scenario for Thai assets."""
    return StressScenario(
        name="Trump Tariffs Impact",
        affected_assets=thai_assets,
        drop_mean=drop_mean,
        drop_std=drop_std,
        vol_increase=vol_increase,
        duration_days=duration_days
    )


def simulate_gbm(s0: float, mu: float, sigma: float, n_steps: int,
                 n_sims: int, dt: float = 1/252) -> np.ndarray:
    """
    Simulate GBM paths: dS = mu*S*dt + sigma*S*dW
    Returns array of shape (n_sims, n_steps+1)
    """
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = s0

    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

    return paths


def run_stress_simulation(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    scenario: StressScenario,
    n_sims: int = 1000,
    initial_value: float = 100.0
) -> Dict:
    """
    Run Monte Carlo stress test simulation.

    For affected assets: apply initial shock + elevated volatility via GBM
    For unaffected assets: normal GBM with historical parameters
    """
    assets = list(weights.keys())
    n_steps = scenario.duration_days

    # Historical parameters (annualized)
    hist_mu = (returns.mean() * 252).to_dict()
    hist_sigma = (returns.std() * np.sqrt(252)).to_dict()

    # Normalize weights
    w = np.array([weights.get(a, 0) for a in assets])
    w = w / w.sum()

    # Simulate each asset
    portfolio_paths = np.zeros((n_sims, n_steps + 1))

    for i, asset in enumerate(assets):
        if w[i] < 1e-6:
            continue

        mu = hist_mu.get(asset, 0.05)
        sigma = hist_sigma.get(asset, 0.20)
        asset_value = initial_value * w[i]

        if asset in scenario.affected_assets:
            # Apply initial shock
            shocks = np.random.normal(scenario.drop_mean, scenario.drop_std, n_sims)
            shocks = np.clip(shocks, -0.5, 0.5)
            shocked_value = asset_value * (1 + shocks)

            # Elevated volatility, reduced drift
            stressed_sigma = sigma * (1 + scenario.vol_increase)
            stressed_mu = mu * 0.5

            # Simulate from shocked value
            for sim in range(n_sims):
                path = simulate_gbm(shocked_value[sim], stressed_mu, stressed_sigma,
                                    n_steps, 1)[0]
                portfolio_paths[sim] += path
        else:
            # Normal simulation
            paths = simulate_gbm(asset_value, mu, sigma, n_steps, n_sims)
            portfolio_paths += paths

    return {
        'paths': portfolio_paths,
        'scenario': scenario,
        'n_sims': n_sims,
        'n_steps': n_steps
    }


def calculate_stress_statistics(sim_result: Dict) -> Dict:
    """Calculate statistics from simulation results."""
    paths = sim_result['paths']
    initial = paths[:, 0].mean()
    final = paths[:, -1]
    returns = (final - initial) / initial

    percentiles = [5, 25, 50, 75, 95]
    pct_values = {f'p{p}': float(np.percentile(returns, p)) for p in percentiles}

    return {
        'initial_value': float(initial),
        'mean_final_value': float(final.mean()),
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'var_95': float(np.percentile(returns, 5)),
        'cvar_95': float(returns[returns <= np.percentile(returns, 5)].mean()),
        'probability_of_loss': float((returns < 0).mean()),
        'probability_of_10pct_loss': float((returns < -0.10).mean()),
        'probability_of_20pct_loss': float((returns < -0.20).mean()),
        'percentiles': pct_values
    }


def get_percentile_paths(sim_result: Dict) -> Dict[str, List[float]]:
    """Get percentile paths for visualization."""
    paths = sim_result['paths']
    result = {}

    for p in [5, 25, 50, 75, 95]:
        result[f'p{p}'] = np.percentile(paths, p, axis=0).tolist()

    result['mean'] = paths.mean(axis=0).tolist()
    return result


def get_sample_paths(sim_result: Dict, n_samples: int = 100) -> List[Dict]:
    """Get sample paths for JSON output."""
    paths = sim_result['paths']
    n_sims = paths.shape[0]
    indices = np.linspace(0, n_sims - 1, min(n_samples, n_sims), dtype=int)

    samples = []
    for idx in indices:
        path = paths[idx]
        samples.append({
            'id': int(idx),
            'initial': float(path[0]),
            'final': float(path[-1]),
            'return': float((path[-1] - path[0]) / path[0]),
            'values': path[::max(1, len(path)//50)].tolist()
        })

    return samples


def run_stress_test(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    scenario: StressScenario,
    n_sims: int = 1000
) -> Dict:
    """Complete stress test with all outputs."""
    sim = run_stress_simulation(returns, weights, scenario, n_sims)

    return {
        'scenario_name': scenario.name,
        'scenario_config': {
            'affected_assets': scenario.affected_assets,
            'drop_mean': scenario.drop_mean,
            'drop_std': scenario.drop_std,
            'vol_increase': scenario.vol_increase,
            'duration_days': scenario.duration_days
        },
        'statistics': calculate_stress_statistics(sim),
        'percentile_paths': get_percentile_paths(sim),
        'sample_paths': get_sample_paths(sim),
        'n_simulations': n_sims
    }
