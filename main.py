"""
Main Module

Orchestrates the complete portfolio optimization and stress testing workflow.
Generates JSON output with optimal weights, backtest returns, and stress scenario results.
"""

import json
import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional

from data_loader import (
    load_and_prepare_data,
    get_default_tickers,
    get_thai_tickers,
    download_benchmark
)
from portfolio_optimizer import (
    optimize_portfolio,
    backtest_portfolio,
    calculate_backtest_metrics,
    get_portfolio_returns_series,
    backtest_benchmark,
    get_individual_stock_performance
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

    print("\n[1/5] Loading and preparing data...")
    prices, returns, cleaned_returns = load_and_prepare_data(
        tickers,
        years=historical_years
    )

    available_tickers = list(cleaned_returns.columns)
    available_thai = [t for t in thai_tickers if t in available_tickers]

    # Download S&P500 benchmark
    print("\n  Downloading S&P 500 benchmark...")
    _, benchmark_returns = download_benchmark('^GSPC', years=historical_years)

    print(f"\n[2/5] Optimizing portfolio (long-only constraint)...")
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

    print(f"\n[3/5] Running historical backtest...")
    backtest = backtest_portfolio(cleaned_returns, optimization_result['weights'])
    backtest_metrics = calculate_backtest_metrics(backtest)
    backtest_returns_data = get_portfolio_returns_series(backtest)

    # Benchmark backtest
    benchmark_bt = backtest_benchmark(cleaned_returns, benchmark_returns)
    benchmark_metrics = calculate_backtest_metrics(benchmark_bt.rename(columns={
        'benchmark_return': 'portfolio_return',
        'benchmark_value': 'portfolio_value'
    }))

    print(f"  Portfolio Total Return: {backtest_metrics['total_return']:.2%}")
    print(f"  Benchmark Total Return: {benchmark_metrics['total_return']:.2%}")
    print(f"  Outperformance: {(backtest_metrics['total_return'] - benchmark_metrics['total_return']):.2%}")
    print(f"  Portfolio Sharpe: {backtest_metrics['sharpe_ratio']:.2f}")

    # Individual stock performance
    print(f"\n[4/5] Calculating individual stock performance...")
    individual_stock_backtest = get_individual_stock_performance(
        cleaned_returns, optimization_result['weights']
    )
    print(f"  Tracked {len(individual_stock_backtest)} stocks with significant weights")

    # Get benchmark returns series for JSON
    benchmark_returns_series = []
    for date, row in benchmark_bt.iterrows():
        benchmark_returns_series.append({
            'date': date.strftime('%Y-%m-%d'),
            'daily_return': float(row['benchmark_return']),
            'cumulative_return': float(row['cumulative_return']),
            'benchmark_value': float(row['benchmark_value'])
        })

    print(f"\n[5/5] Running stress test simulation...")
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

    # Calculate historical returns distribution
    portfolio_daily_returns = [d['daily_return'] for d in backtest_returns_data]
    historical_returns_dist = {
        'mean': float(sum(portfolio_daily_returns) / len(portfolio_daily_returns)),
        'std': float((sum((r - sum(portfolio_daily_returns)/len(portfolio_daily_returns))**2 for r in portfolio_daily_returns) / len(portfolio_daily_returns))**0.5),
        'min': float(min(portfolio_daily_returns)),
        'max': float(max(portfolio_daily_returns)),
        'daily_returns': portfolio_daily_returns
    }

    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'historical_years': historical_years,
            'forward_years': forward_years,
            'tickers_requested': tickers,
            'tickers_available': available_tickers,
            'thai_tickers_affected': available_thai,
            'benchmark': 'S&P 500 (^GSPC)'
        },
        'optimal_weights': optimization_result['weights'],
        'optimization_stats': {
            'expected_return': optimization_result['expected_return'],
            'volatility': optimization_result['volatility'],
            'sharpe_ratio': optimization_result['sharpe_ratio']
        },
        'backtest': {
            'metrics': backtest_metrics,
            'returns': backtest_returns_data,
            'individual_stocks': individual_stock_backtest,
            'returns_distribution': historical_returns_dist
        },
        'benchmark': {
            'metrics': benchmark_metrics,
            'returns': benchmark_returns_series
        },
        'stress_test': stress_results
    }

    if output_file:
        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
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
        default='reports/portfolio_analysis.json',
        help='Output JSON file path (default: reports/portfolio_analysis.json)'
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
