#!/usr/bin/env python3
"""
Run Full Benchmark on All 6 Scenarios

This script runs a complete benchmark across all traffic scenarios:
- HH_168h (High inbound, High outbound)
- MH_168h (Medium inbound, High outbound)
- MM_168h (Medium inbound, Medium outbound)
- LH_168h (Low inbound, High outbound)
- LM_168h (Low inbound, Medium outbound)
- LL_168h (Low inbound, Low outbound)

Usage:
    python3 run_full_benchmark.py

Options:
    python3 run_full_benchmark.py --quick    # Run 5 instances per scenario (faster)
    python3 run_full_benchmark.py --full     # Run all 10 instances per scenario (default)
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.benchmarking.pipeline import BenchmarkPipeline


def main():
    """Run the full benchmark."""

    # Parse command line arguments
    quick_mode = '--quick' in sys.argv
    max_instances = 5 if quick_mode else None  # None = all instances

    print("="*80)
    print("CROSS-DOCKING OPTIMIZATION - FULL BENCHMARK")
    print("="*80)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if quick_mode:
        print("Mode: QUICK (5 instances per scenario)")
        print("Estimated time: 10-15 minutes")
    else:
        print("Mode: FULL (10 instances per scenario)")
        print("Estimated time: 30-60 minutes")

    print()
    print("Scenarios to benchmark:")
    scenarios = ['HH_168h', 'MH_168h', 'MM_168h', 'LH_168h', 'LM_168h', 'LL_168h']
    for i, scenario in enumerate(scenarios, 1):
        instances = 5 if quick_mode else 10
        print(f"  {i}. {scenario:8s} - {instances} instances")

    print()
    print("Algorithms to test:")
    algorithms = ['EDD', 'First-Fit', 'Best-Fit']
    for i, algo in enumerate(algorithms, 1):
        print(f"  {i}. {algo}")

    instances_count = (5 if quick_mode else 10) * len(scenarios)
    total_experiments = instances_count * len(algorithms)
    print()
    print(f"Total experiments: {total_experiments} ({instances_count} instances × {len(algorithms)} algorithms)")
    print()

    # Confirm before running
    response = input("Start benchmark? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Benchmark cancelled.")
        return

    print()
    print("="*80)
    print("RUNNING BENCHMARK...")
    print("="*80)
    print()

    # Initialize pipeline
    pipeline = BenchmarkPipeline(output_dir=str(project_root / 'results' / 'benchmarks'))

    # Run benchmark
    start_time = datetime.now()

    results_df = pipeline.run_full_benchmark(
        scenarios=scenarios,
        algorithms=algorithms,
        max_instances_per_scenario=max_instances
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save results
    output_file = project_root / 'results' / 'benchmarks' / 'full_benchmark_results.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print()
    print("="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print()
    print(f"Total experiments completed: {len(results_df)}")
    print(f"Results saved to: {output_file}")
    print()

    # Summary statistics
    print("="*80)
    print("SUMMARY BY ALGORITHM")
    print("="*80)
    print()

    summary = results_df.groupby('algorithm').agg({
        'service_level': ['mean', 'std', 'min', 'max'],
        'avg_fill_rate': ['mean', 'std'],
        'num_late_pallets': ['mean', 'sum'],
        'solve_time': ['mean', 'max']
    }).round(4)

    print(summary)
    print()

    # Highlight best algorithm
    best_algo = results_df.groupby('algorithm')['service_level'].mean().idxmax()
    best_service = results_df.groupby('algorithm')['service_level'].mean().max()

    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\n🏆 Best Algorithm: {best_algo}")
    print(f"   Service Level: {best_service:.2%}")
    print()

    # EDD-specific stats
    if 'EDD' in results_df['algorithm'].unique():
        edd_data = results_df[results_df['algorithm'] == 'EDD']
        print("⭐ EDD Heuristic Performance:")
        print(f"   Average Service Level: {edd_data['service_level'].mean():.2%}")
        print(f"   Average Fill Rate: {edd_data['avg_fill_rate'].mean():.2%}")
        print(f"   Average Solve Time: {edd_data['solve_time'].mean():.4f} seconds")
        print(f"   Instances with >99% Service: {(edd_data['service_level'] > 0.99).sum()}/{len(edd_data)}")
        print()

    # Scenario breakdown
    print("="*80)
    print("PERFORMANCE BY SCENARIO")
    print("="*80)
    print()

    scenario_summary = results_df.groupby('scenario').agg({
        'service_level': 'mean',
        'avg_fill_rate': 'mean',
        'num_late_pallets': 'mean'
    }).round(4)

    print(scenario_summary)
    print()

    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. View detailed analysis in Jupyter notebook:")
    print("   jupyter notebook notebooks/03_results_dashboard.ipynb")
    print()
    print("2. View raw results:")
    print(f"   cat {output_file}")
    print()
    print("3. Generate visualizations:")
    print("   Run cells in notebooks/03_results_dashboard.ipynb")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
