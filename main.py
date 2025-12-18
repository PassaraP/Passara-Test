"""
Main Module
Orchestrates portfolio optimization and stress testing, outputs JSON.
"""

import json
import argparse
from datetime import datetime

from data_loader import load_data, get_default_tickers, get_thai_tickers
from portfolio_optimizer import (
    optimize_portfolio, backtest_portfolio,
    calculate_metrics, backtest_to_list
)
from scenario_modeling import create_tariff_scenario, run_stress_test


def run_analysis(
    years: int = 10,
    forward_years: int = 1,
    drop_mean: float = -0.10,
    drop_std: float = 0.02,
    vol_increase: float = 0.10,
    n_sims: int = 1000,
    output_file: str = 'portfolio_analysis.json'
) -> dict:
    """Run complete portfolio analysis."""

    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION WITH STRESS TESTING")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading data...")
    tickers = get_default_tickers()
    prices, returns = load_data(tickers, years)

    available = list(returns.columns)
    thai = [t for t in get_thai_tickers() if t in available]

    # 2. Optimize portfolio
    print("\n[2/4] Optimizing portfolio (max Sharpe, long-only)...")
    opt = optimize_portfolio(returns)

    print(f"  Expected Return: {opt['expected_return']:.2%}")
    print(f"  Volatility: {opt['volatility']:.2%}")
    print(f"  Sharpe Ratio: {opt['sharpe_ratio']:.2f}")

    print("\n  Top holdings:")
    for t, w in sorted(opt['weights'].items(), key=lambda x: -x[1])[:5]:
        if w > 0.01:
            print(f"    {t}: {w:.1%}")

    # 3. Backtest
    print("\n[3/4] Running historical backtest...")
    bt = backtest_portfolio(returns, opt['weights'])
    metrics = calculate_metrics(bt)

    print(f"  Total Return: {metrics['total_return']:.1%}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")

    # 4. Stress test
    print("\n[4/4] Running stress test simulation...")
    scenario = create_tariff_scenario(
        thai_assets=thai,
        drop_mean=drop_mean,
        drop_std=drop_std,
        vol_increase=vol_increase,
        duration_days=int(forward_years * 252)
    )

    print(f"  Scenario: {scenario.name}")
    print(f"  Affected: {len(thai)} Thai assets")
    print(f"  Drop: {drop_mean:.0%} ± {drop_std:.0%}")
    print(f"  Vol increase: {vol_increase:.0%}")
    print(f"  Simulations: {n_sims}")

    stress = run_stress_test(returns, opt['weights'], scenario, n_sims)

    stats = stress['statistics']
    print(f"\n  Results:")
    print(f"    Mean Return: {stats['mean_return']:.2%}")
    print(f"    VaR (95%): {stats['var_95']:.2%}")
    print(f"    P(Loss): {stats['probability_of_loss']:.1%}")

    # Build output
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'historical_years': years,
            'forward_years': forward_years,
            'tickers_available': available,
            'thai_assets_affected': thai
        },
        'optimal_weights': opt['weights'],
        'optimization_stats': {
            'expected_return': opt['expected_return'],
            'volatility': opt['volatility'],
            'sharpe_ratio': opt['sharpe_ratio']
        },
        'backtest': {
            'metrics': metrics,
            'returns': backtest_to_list(bt)
        },
        'stress_test': stress
    }

    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Portfolio Optimization with Stress Testing')
    parser.add_argument('--years', type=int, default=10, help='Historical years (default: 10)')
    parser.add_argument('--forward', type=int, default=1, help='Forward simulation years (default: 1)')
    parser.add_argument('--drop', type=float, default=-0.10, help='Stress drop mean (default: -0.10)')
    parser.add_argument('--drop-std', type=float, default=0.02, help='Stress drop std (default: 0.02)')
    parser.add_argument('--vol', type=float, default=0.10, help='Volatility increase (default: 0.10)')
    parser.add_argument('--sims', type=int, default=1000, help='Monte Carlo simulations (default: 1000)')
    parser.add_argument('--output', type=str, default='portfolio_analysis.json', help='Output file')

    args = parser.parse_args()

    run_analysis(
        years=args.years,
        forward_years=args.forward,
        drop_mean=args.drop,
        drop_std=args.drop_std,
        vol_increase=args.vol,
        n_sims=args.sims,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
