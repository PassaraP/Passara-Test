"""
Visualization Module

Creates visualizations from portfolio analysis JSON output.
Generates charts for portfolio weights, backtest performance, and stress test results.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict, Optional


def load_results(json_path: str) -> Dict:
    """Load portfolio analysis results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_optimal_weights(results: Dict, output_path: str) -> None:
    """
    Create a bar chart of optimal portfolio weights.

    Args:
        results: Portfolio analysis results
        output_path: Path to save the figure
    """
    weights = results['optimal_weights']

    # Filter out near-zero weights
    significant_weights = {k: v for k, v in weights.items() if v > 0.001}
    sorted_weights = sorted(significant_weights.items(), key=lambda x: -x[1])

    tickers = [w[0] for w in sorted_weights]
    values = [w[1] * 100 for w in sorted_weights]  # Convert to percentage

    # Color by asset type
    colors = []
    for ticker in tickers:
        if ticker.endswith('.BK'):
            colors.append('#e74c3c')  # Red for Thai stocks
        elif ticker in ['VWOB', 'EMLC', 'LEMB']:
            colors.append('#3498db')  # Blue for fixed income
        elif ticker == 'THB=X':
            colors.append('#f39c12')  # Orange for FX
        else:
            colors.append('#2ecc71')  # Green for global

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(tickers, values, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)

    ax.set_xlabel('Weight (%)', fontsize=12)
    ax.set_title('Optimal Portfolio Weights (Max Sharpe Ratio)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.15)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Thai Stocks'),
        Patch(facecolor='#3498db', label='Fixed Income'),
        Patch(facecolor='#2ecc71', label='Global'),
        Patch(facecolor='#f39c12', label='FX')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add stats annotation
    stats = results['optimization_stats']
    stats_text = f"Expected Return: {stats['expected_return']*100:.2f}%\n"
    stats_text += f"Volatility: {stats['volatility']*100:.2f}%\n"
    stats_text += f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_backtest_performance(results: Dict, output_path: str) -> None:
    """
    Create a chart showing historical backtest performance.

    Args:
        results: Portfolio analysis results
        output_path: Path to save the figure
    """
    backtest_data = results['backtest']['returns']

    dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in backtest_data]
    portfolio_values = [d['portfolio_value'] for d in backtest_data]
    cumulative_returns = [d['cumulative_return'] * 100 for d in backtest_data]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Portfolio value
    ax1 = axes[0]
    ax1.plot(dates, portfolio_values, color='#2c3e50', linewidth=1.5)
    ax1.fill_between(dates, portfolio_values, alpha=0.3, color='#3498db')
    ax1.set_ylabel('Portfolio Value', fontsize=12)
    ax1.set_title('Historical Backtest Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Calculate and plot drawdown
    portfolio_arr = np.array(portfolio_values)
    rolling_max = np.maximum.accumulate(portfolio_arr)
    drawdown = (portfolio_arr - rolling_max) / rolling_max * 100

    ax2 = axes[1]
    ax2.fill_between(dates, drawdown, 0, color='#e74c3c', alpha=0.5)
    ax2.plot(dates, drawdown, color='#c0392b', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # Add metrics annotation
    metrics = results['backtest']['metrics']
    metrics_text = f"Total Return: {metrics['total_return']*100:.1f}%\n"
    metrics_text += f"Ann. Return: {metrics['annualized_return']*100:.2f}%\n"
    metrics_text += f"Ann. Volatility: {metrics['annualized_volatility']*100:.2f}%\n"
    metrics_text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
    metrics_text += f"Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
    metrics_text += f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%"

    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stress_test_paths(results: Dict, output_path: str) -> None:
    """
    Create a chart showing stress test simulation paths.

    Args:
        results: Portfolio analysis results
        output_path: Path to save the figure
    """
    stress_test = results['stress_test']
    percentile_paths = stress_test['percentile_paths']
    n_steps = len(percentile_paths['mean'])

    days = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot percentile bands
    ax.fill_between(days, percentile_paths['p5'], percentile_paths['p95'],
                    alpha=0.2, color='#3498db', label='5th-95th percentile')
    ax.fill_between(days, percentile_paths['p25'], percentile_paths['p75'],
                    alpha=0.3, color='#3498db', label='25th-75th percentile')

    # Plot median and mean
    ax.plot(days, percentile_paths['p50'], color='#2980b9', linewidth=2,
            label='Median', linestyle='--')
    ax.plot(days, percentile_paths['mean'], color='#e74c3c', linewidth=2,
            label='Mean')

    # Starting value line
    initial_value = percentile_paths['mean'][0]
    ax.axhline(y=initial_value, color='gray', linestyle=':', alpha=0.7, label='Initial Value')

    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.set_title(f'Stress Test: {stress_test["scenario_name"]} - Portfolio Simulation Paths',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add scenario config annotation
    config = stress_test['scenario_config']
    config_text = f"Affected Assets: {len(config['affected_assets'])}\n"
    config_text += f"Expected Drop: {config['expected_drop_mean']*100:.0f}% ± {config['expected_drop_std']*100:.0f}%\n"
    config_text += f"Vol Increase: {config['volatility_increase']*100:.0f}%\n"
    config_text += f"Duration: {config['duration_days']} days"

    ax.text(0.98, 0.98, config_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stress_test_distribution(results: Dict, output_path: str) -> None:
    """
    Create a histogram of stress test return distribution.

    Args:
        results: Portfolio analysis results
        output_path: Path to save the figure
    """
    stress_test = results['stress_test']
    sample_paths = stress_test['sample_paths']
    statistics = stress_test['statistics']

    returns = [p['return'] * 100 for p in sample_paths]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create histogram
    n, bins, patches = ax.hist(returns, bins=30, color='#3498db',
                                edgecolor='white', alpha=0.7, density=True)

    # Color bars based on return (red for negative)
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#e74c3c')

    # Add VaR and CVaR lines
    var_95 = statistics['var_95'] * 100
    cvar_95 = statistics['cvar_95'] * 100
    mean_ret = statistics['mean_return'] * 100

    ax.axvline(x=var_95, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'VaR 95%: {var_95:.1f}%')
    ax.axvline(x=cvar_95, color='#c0392b', linestyle=':', linewidth=2,
               label=f'CVaR 95%: {cvar_95:.1f}%')
    ax.axvline(x=mean_ret, color='#27ae60', linestyle='-', linewidth=2,
               label=f'Mean: {mean_ret:.1f}%')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Return (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Stress Test Return Distribution ({stress_test["n_simulations"]} simulations)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics annotation
    stats_text = f"P(Loss): {statistics['probability_of_loss']*100:.1f}%\n"
    stats_text += f"P(Loss > 10%): {statistics['probability_of_10pct_loss']*100:.1f}%\n"
    stats_text += f"P(Loss > 20%): {statistics['probability_of_20pct_loss']*100:.1f}%\n"
    stats_text += f"Std Dev: {statistics['std_return']*100:.1f}%"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_dashboard(results: Dict, output_path: str) -> None:
    """
    Create a summary dashboard with multiple panels.

    Args:
        results: Portfolio analysis results
        output_path: Path to save the figure
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # 1. Portfolio Weights (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    weights = results['optimal_weights']
    significant = {k: v for k, v in weights.items() if v > 0.001}
    sorted_w = sorted(significant.items(), key=lambda x: -x[1])

    colors = ['#e74c3c' if t.endswith('.BK') else '#3498db' if t in ['VWOB','EMLC','LEMB']
              else '#2ecc71' for t, _ in sorted_w]

    wedges, texts, autotexts = ax1.pie(
        [w[1] for w in sorted_w],
        labels=[w[0] for w in sorted_w],
        autopct='%1.1f%%',
        colors=colors,
        pctdistance=0.75
    )
    ax1.set_title('Optimal Portfolio Weights', fontweight='bold')

    # 2. Backtest Performance (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    backtest = results['backtest']['returns']
    dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in backtest]
    values = [d['portfolio_value'] for d in backtest]
    ax2.plot(dates, values, color='#2c3e50', linewidth=1)
    ax2.fill_between(dates, values, alpha=0.3, color='#3498db')
    ax2.set_title('Historical Backtest', fontweight='bold')
    ax2.set_ylabel('Portfolio Value')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.grid(True, alpha=0.3)

    # 3. Stress Test Paths (middle, spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])
    percentile_paths = results['stress_test']['percentile_paths']
    n_steps = len(percentile_paths['mean'])
    days = np.arange(n_steps)

    ax3.fill_between(days, percentile_paths['p5'], percentile_paths['p95'],
                     alpha=0.2, color='#3498db')
    ax3.fill_between(days, percentile_paths['p25'], percentile_paths['p75'],
                     alpha=0.3, color='#3498db')
    ax3.plot(days, percentile_paths['p50'], color='#2980b9', linewidth=2, linestyle='--')
    ax3.plot(days, percentile_paths['mean'], color='#e74c3c', linewidth=2)
    ax3.set_title(f'Stress Test: {results["stress_test"]["scenario_name"]}', fontweight='bold')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Portfolio Value')
    ax3.grid(True, alpha=0.3)

    # 4. Return Distribution (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    sample_paths = results['stress_test']['sample_paths']
    returns = [p['return'] * 100 for p in sample_paths]
    n, bins, patches = ax4.hist(returns, bins=25, color='#3498db', edgecolor='white', alpha=0.7)
    for patch, left in zip(patches, bins[:-1]):
        if left < 0:
            patch.set_facecolor('#e74c3c')
    ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_title('Stress Test Return Distribution', fontweight='bold')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Frequency')

    # 5. Key Metrics Table (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    backtest_metrics = results['backtest']['metrics']
    stress_stats = results['stress_test']['statistics']
    opt_stats = results['optimization_stats']

    table_data = [
        ['Metric', 'Backtest', 'Stress Test'],
        ['Expected Return', f"{opt_stats['expected_return']*100:.2f}%", f"{stress_stats['mean_return']*100:.2f}%"],
        ['Volatility', f"{opt_stats['volatility']*100:.2f}%", f"{stress_stats['std_return']*100:.2f}%"],
        ['Sharpe Ratio', f"{backtest_metrics['sharpe_ratio']:.2f}", '-'],
        ['Max Drawdown', f"{backtest_metrics['max_drawdown']*100:.1f}%", '-'],
        ['VaR (95%)', '-', f"{stress_stats['var_95']*100:.2f}%"],
        ['CVaR (95%)', '-', f"{stress_stats['cvar_95']*100:.2f}%"],
        ['P(Loss)', '-', f"{stress_stats['probability_of_loss']*100:.1f}%"],
    ]

    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax5.set_title('Performance Metrics', fontweight='bold', pad=20)

    # Main title
    fig.suptitle('Portfolio Optimization & Stress Test Summary',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_visualizations(
    json_path: str,
    output_dir: str = 'result'
) -> None:
    """
    Generate all visualizations and save to output directory.

    Args:
        json_path: Path to portfolio analysis JSON file
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from {json_path}...")
    results = load_results(json_path)

    print(f"\nGenerating visualizations in '{output_dir}/'...")

    # Generate all plots
    plot_optimal_weights(results, os.path.join(output_dir, 'portfolio_weights.png'))
    plot_backtest_performance(results, os.path.join(output_dir, 'backtest_performance.png'))
    plot_stress_test_paths(results, os.path.join(output_dir, 'stress_test_paths.png'))
    plot_stress_test_distribution(results, os.path.join(output_dir, 'stress_test_distribution.png'))
    plot_summary_dashboard(results, os.path.join(output_dir, 'summary_dashboard.png'))

    # Copy JSON to result folder
    import shutil
    json_dest = os.path.join(output_dir, 'portfolio_analysis.json')
    shutil.copy(json_path, json_dest)
    print(f"Copied: {json_dest}")

    print(f"\nAll visualizations saved to '{output_dir}/'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate portfolio analysis visualizations')
    parser.add_argument('--input', type=str, default='portfolio_analysis.json',
                        help='Input JSON file path')
    parser.add_argument('--output-dir', type=str, default='result',
                        help='Output directory for visualizations')

    args = parser.parse_args()

    generate_all_visualizations(args.input, args.output_dir)
