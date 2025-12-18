"""
Visualization Module
Generate charts from portfolio analysis JSON output.
"""

import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Dict


def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def plot_weights(results: Dict, output: str):
    """Bar chart of portfolio weights."""
    weights = {k: v for k, v in results['optimal_weights'].items() if v > 0.001}
    sorted_w = sorted(weights.items(), key=lambda x: -x[1])

    tickers = [w[0] for w in sorted_w]
    values = [w[1] * 100 for w in sorted_w]

    colors = ['#e74c3c' if t.endswith('.BK') else
              '#3498db' if t in ['VWOB', 'EMLC', 'LEMB'] else
              '#f39c12' if t == 'THB=X' else '#2ecc71' for t in tickers]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(tickers, values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center')

    ax.set_xlabel('Weight (%)')
    ax.set_title('Optimal Portfolio Weights', fontweight='bold')
    ax.set_xlim(0, max(values) * 1.15)

    stats = results['optimization_stats']
    text = f"Return: {stats['expected_return']*100:.1f}%\nVol: {stats['volatility']*100:.1f}%\nSharpe: {stats['sharpe_ratio']:.2f}"
    ax.text(0.95, 0.05, text, transform=ax.transAxes, ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}")


def plot_backtest(results: Dict, output: str):
    """Historical backtest chart with drawdown."""
    data = results['backtest']['returns']
    dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in data]
    values = [d['portfolio_value'] for d in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Portfolio value
    ax1.plot(dates, values, color='#2c3e50', lw=1.5)
    ax1.fill_between(dates, values, alpha=0.3)
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title('Historical Backtest Performance', fontweight='bold')
    ax1.grid(alpha=0.3)

    # Drawdown
    arr = np.array(values)
    rolling_max = np.maximum.accumulate(arr)
    dd = (arr - rolling_max) / rolling_max * 100

    ax2.fill_between(dates, dd, 0, color='#e74c3c', alpha=0.5)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Metrics
    m = results['backtest']['metrics']
    text = f"Total: {m['total_return']*100:.0f}%\nAnn: {m['annualized_return']*100:.1f}%\nDD: {m['max_drawdown']*100:.0f}%\nSharpe: {m['sharpe_ratio']:.2f}"
    ax1.text(0.02, 0.95, text, transform=ax1.transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}")


def plot_stress_paths(results: Dict, output: str):
    """Stress test simulation paths with percentile bands."""
    stress = results['stress_test']
    paths = stress['percentile_paths']
    n = len(paths['mean'])
    days = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(days, paths['p5'], paths['p95'], alpha=0.2, color='#3498db', label='5-95%')
    ax.fill_between(days, paths['p25'], paths['p75'], alpha=0.3, color='#3498db', label='25-75%')
    ax.plot(days, paths['p50'], '--', color='#2980b9', lw=2, label='Median')
    ax.plot(days, paths['mean'], color='#e74c3c', lw=2, label='Mean')
    ax.axhline(paths['mean'][0], color='gray', ls=':', alpha=0.7)

    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Portfolio Value')
    ax.set_title(f"Stress Test: {stress['scenario_name']}", fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    cfg = stress['scenario_config']
    text = f"Drop: {cfg['drop_mean']*100:.0f}%±{cfg['drop_std']*100:.0f}%\nVol+: {cfg['vol_increase']*100:.0f}%"
    ax.text(0.98, 0.02, text, transform=ax.transAxes, ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}")


def plot_stress_distribution(results: Dict, output: str):
    """Return distribution histogram."""
    stress = results['stress_test']
    returns = [p['return'] * 100 for p in stress['sample_paths']]
    stats = stress['statistics']

    fig, ax = plt.subplots(figsize=(10, 6))

    n, bins, patches = ax.hist(returns, bins=30, color='#3498db', edgecolor='white', alpha=0.7)
    for patch, left in zip(patches, bins[:-1]):
        if left < 0:
            patch.set_facecolor('#e74c3c')

    ax.axvline(stats['var_95']*100, color='#c0392b', ls='--', lw=2, label=f"VaR 95%: {stats['var_95']*100:.1f}%")
    ax.axvline(stats['mean_return']*100, color='#27ae60', lw=2, label=f"Mean: {stats['mean_return']*100:.1f}%")
    ax.axvline(0, color='gray', alpha=0.5)

    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Stress Test Return Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    text = f"P(Loss): {stats['probability_of_loss']*100:.0f}%\nP(>10% Loss): {stats['probability_of_10pct_loss']*100:.0f}%"
    ax.text(0.02, 0.95, text, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}")


def plot_dashboard(results: Dict, output: str):
    """Combined summary dashboard."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # 1. Weights pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    weights = {k: v for k, v in results['optimal_weights'].items() if v > 0.001}
    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    colors = ['#e74c3c' if t.endswith('.BK') else '#3498db' if t in ['VWOB','EMLC','LEMB'] else '#2ecc71'
              for t, _ in sorted_w]
    ax1.pie([w[1] for w in sorted_w], labels=[w[0] for w in sorted_w],
            autopct='%1.0f%%', colors=colors, pctdistance=0.8)
    ax1.set_title('Portfolio Weights', fontweight='bold')

    # 2. Backtest line
    ax2 = fig.add_subplot(gs[0, 1])
    data = results['backtest']['returns']
    dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in data]
    values = [d['portfolio_value'] for d in data]
    ax2.plot(dates, values, color='#2c3e50')
    ax2.fill_between(dates, values, alpha=0.3)
    ax2.set_title('Historical Backtest', fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.grid(alpha=0.3)

    # 3. Stress paths
    ax3 = fig.add_subplot(gs[1, 0])
    paths = results['stress_test']['percentile_paths']
    days = np.arange(len(paths['mean']))
    ax3.fill_between(days, paths['p5'], paths['p95'], alpha=0.2, color='#3498db')
    ax3.fill_between(days, paths['p25'], paths['p75'], alpha=0.3, color='#3498db')
    ax3.plot(days, paths['mean'], color='#e74c3c', lw=2)
    ax3.set_title('Stress Test Paths', fontweight='bold')
    ax3.set_xlabel('Days')
    ax3.grid(alpha=0.3)

    # 4. Metrics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    bt = results['backtest']['metrics']
    st = results['stress_test']['statistics']
    opt = results['optimization_stats']

    table_data = [
        ['Expected Return', f"{opt['expected_return']*100:.1f}%", f"{st['mean_return']*100:.1f}%"],
        ['Volatility', f"{opt['volatility']*100:.1f}%", f"{st['std_return']*100:.1f}%"],
        ['Sharpe Ratio', f"{bt['sharpe_ratio']:.2f}", '-'],
        ['Max Drawdown', f"{bt['max_drawdown']*100:.0f}%", '-'],
        ['VaR (95%)', '-', f"{st['var_95']*100:.1f}%"],
        ['P(Loss)', '-', f"{st['probability_of_loss']*100:.0f}%"],
    ]

    table = ax4.table(cellText=table_data, colLabels=['Metric', 'Backtest', 'Stress'],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax4.set_title('Performance Metrics', fontweight='bold', pad=20)

    fig.suptitle('Portfolio Analysis Summary', fontsize=14, fontweight='bold')
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def generate_visualizations(json_path: str, output_dir: str = 'result'):
    """Generate all visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {json_path}...")
    results = load_results(json_path)

    print(f"Generating visualizations in {output_dir}/...")
    plot_weights(results, os.path.join(output_dir, 'portfolio_weights.png'))
    plot_backtest(results, os.path.join(output_dir, 'backtest_performance.png'))
    plot_stress_paths(results, os.path.join(output_dir, 'stress_test_paths.png'))
    plot_stress_distribution(results, os.path.join(output_dir, 'stress_test_distribution.png'))
    plot_dashboard(results, os.path.join(output_dir, 'summary_dashboard.png'))

    # Copy JSON
    shutil.copy(json_path, os.path.join(output_dir, 'portfolio_analysis.json'))
    print(f"Copied JSON to {output_dir}/")

    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='portfolio_analysis.json')
    parser.add_argument('--output-dir', default='result')
    args = parser.parse_args()

    generate_visualizations(args.input, args.output_dir)
