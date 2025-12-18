"""
Main Module

Orchestrates the complete portfolio optimization and stress testing workflow.
Generates JSON output with optimal weights, backtest returns, and stress scenario results.
"""

import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

from data_loader import (
    load_and_prepare_data,
    get_default_tickers,
    get_thai_tickers
)
from portfolio_optimizer import (
    optimize_portfolio,
    backtest_portfolio,
    calculate_backtest_metrics,
    get_portfolio_returns_series
)
from scenario_modeling import (
    create_trump_tariff_scenario,
    run_stress_test,
    StressScenario
)


def run_portfolio_analysis(
    tickers: Optional[List[str]] = None,
    thai_tickers: Optional[List[str]] = None,
    historical_years: int = 10,
    forward_years: int = 1,
    stress_drop_mean: float = -0.10,
    stress_drop_std: float = 0.02,
    stress_vol_increase: float = 0.10,
    n_simulations: int = 1000,
    output_file: Optional[str] = None
) -> Dict:
    """
    Run complete portfolio analysis with optimization and stress testing.

    Args:
        tickers: List of ticker symbols (uses default if None)
        thai_tickers: List of Thai-related tickers for stress scenario
        historical_years: Years of historical data
        forward_years: Years for forward simulation
        stress_drop_mean: Mean expected drop for Thai assets
        stress_drop_std: Std dev of drop
        stress_vol_increase: Volatility increase for Thai assets
        n_simulations: Number of Monte Carlo simulations
        output_file: Path to save JSON output (optional)

    Returns:
        Complete analysis results as dictionary
    """
    if tickers is None:
        tickers = get_default_tickers()

    if thai_tickers is None:
        thai_tickers = get_thai_tickers()

    print("="*70)
    print("PORTFOLIO OPTIMIZATION AND STRESS TESTING")
    print("="*70)

    print("\n[1/4] Loading and preparing data...")
    prices, returns, cleaned_returns = load_and_prepare_data(
        tickers,
        years=historical_years
    )

    available_tickers = list(cleaned_returns.columns)
    available_thai = [t for t in thai_tickers if t in available_tickers]

    print(f"\n[2/4] Optimizing portfolio (long-only constraint)...")
    optimization_result = optimize_portfolio(
        cleaned_returns,
        objective='max_sharpe'
    )

    print(f"  Expected Annual Return: {optimization_result['expected_return']:.2%}")
    print(f"  Annual Volatility: {optimization_result['volatility']:.2%}")
    print(f"  Sharpe Ratio: {optimization_result['sharpe_ratio']:.2f}")

    print("\n  Top holdings:")
    sorted_weights = sorted(
        optimization_result['weights'].items(),
        key=lambda x: -x[1]
    )
    for ticker, weight in sorted_weights[:5]:
        if weight > 0.01:
            print(f"    {ticker}: {weight:.2%}")

    print(f"\n[3/4] Running historical backtest...")
    backtest = backtest_portfolio(cleaned_returns, optimization_result['weights'])
    backtest_metrics = calculate_backtest_metrics(backtest)
    backtest_returns = get_portfolio_returns_series(backtest)

    print(f"  Total Return: {backtest_metrics['total_return']:.2%}")
    print(f"  Annualized Return: {backtest_metrics['annualized_return']:.2%}")
    print(f"  Max Drawdown: {backtest_metrics['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}")

    print(f"\n[4/4] Running stress test simulation...")
    print(f"  Scenario: Trump Tariffs Impact")
    print(f"  Affected assets: {len(available_thai)} Thai-related tickers")
    print(f"  Expected drop: {stress_drop_mean:.1%} ± {stress_drop_std:.1%}")
    print(f"  Volatility increase: {stress_vol_increase:.1%}")
    print(f"  Simulations: {n_simulations}")

    scenario = create_trump_tariff_scenario(
        thai_assets=available_thai,
        expected_drop_mean=stress_drop_mean,
        expected_drop_std=stress_drop_std,
        volatility_increase=stress_vol_increase,
        duration_days=int(forward_years * 252)
    )

    stress_results = run_stress_test(
        cleaned_returns,
        optimization_result['weights'],
        scenario,
        n_simulations=n_simulations
    )

    print(f"\n  Stress Test Results:")
    print(f"    Mean Return: {stress_results['statistics']['mean_return']:.2%}")
    print(f"    VaR (95%): {stress_results['statistics']['var_95']:.2%}")
    print(f"    CVaR (95%): {stress_results['statistics']['cvar_95']:.2%}")
    print(f"    P(Loss): {stress_results['statistics']['probability_of_loss']:.2%}")
    print(f"    P(Loss > 10%): {stress_results['statistics']['probability_of_10pct_loss']:.2%}")

    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'historical_years': historical_years,
            'forward_years': forward_years,
            'tickers_requested': tickers,
            'tickers_available': available_tickers,
            'thai_tickers_affected': available_thai
        },
        'optimal_weights': optimization_result['weights'],
        'optimization_stats': {
            'expected_return': optimization_result['expected_return'],
            'volatility': optimization_result['volatility'],
            'sharpe_ratio': optimization_result['sharpe_ratio']
        },
        'backtest': {
            'metrics': backtest_metrics,
            'returns': backtest_returns
        },
        'stress_test': stress_results
    }

    if output_file:
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("Done!")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return results


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Portfolio Optimization with Stress Testing'
    )
    parser.add_argument(
        '--historical-years',
        type=int,
        default=10,
        help='Years of historical data (default: 10)'
    )
    parser.add_argument(
        '--forward-years',
        type=int,
        default=1,
        help='Years for forward simulation (default: 1)'
    )
    parser.add_argument(
        '--stress-drop',
        type=float,
        default=-0.10,
        help='Mean expected drop for Thai assets (default: -0.10)'
    )
    parser.add_argument(
        '--stress-drop-std',
        type=float,
        default=0.02,
        help='Std dev of drop (default: 0.02)'
    )
    parser.add_argument(
        '--stress-vol',
        type=float,
        default=0.10,
        help='Volatility increase for Thai assets (default: 0.10)'
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='portfolio_analysis.json',
        help='Output JSON file path (default: portfolio_analysis.json)'
    )

    args = parser.parse_args()

    results = run_portfolio_analysis(
        historical_years=args.historical_years,
        forward_years=args.forward_years,
        stress_drop_mean=args.stress_drop,
        stress_drop_std=args.stress_drop_std,
        stress_vol_increase=args.stress_vol,
        n_simulations=args.simulations,
        output_file=args.output
    )

    return results


if __name__ == "__main__":
    main()
