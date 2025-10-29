"""
Automated Benchmarking Pipeline

This module provides automated benchmarking of optimization algorithms
across all 60 dataset instances, with result collection and analysis.

Features:
- Run multiple algorithms on multiple instances
- Collect comprehensive performance metrics
- Export results to CSV/Excel
- Generate comparison reports
- Statistical analysis

Author: Cross-Docking Optimization Project
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional
from pathlib import Path
from datetime import datetime
import time
import logging
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import DataLoader, CrossDockInstance
from models.pallet_assignment import optimize_pallet_assignment, AssignmentSolution
from models.heuristics import first_fit, best_fit, earliest_due_date, destination_balanced
from analysis.kpis import KPICalculator, KPIReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkPipeline:
    """
    Automated benchmarking pipeline for optimization algorithms.

    Runs experiments across multiple instances and algorithms,
    collecting comprehensive performance data.
    """

    def __init__(self, output_dir: str = "results/benchmarks"):
        """Initialize benchmarking pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DataLoader()
        self.calculator = KPICalculator()

        # Default algorithm registry
        self.algorithms = {
            'EDD': earliest_due_date,
            'First-Fit': first_fit,
            'Best-Fit': best_fit,
            'Dest-Balanced': destination_balanced
        }

        logger.info(f"Benchmark pipeline initialized. Output: {self.output_dir}")

    def register_algorithm(self, name: str, algorithm_func: Callable):
        """Register a new algorithm for benchmarking."""
        self.algorithms[name] = algorithm_func
        logger.info(f"Registered algorithm: {name}")

    def run_single_experiment(
        self,
        instance: CrossDockInstance,
        algorithm_name: str,
        algorithm_func: Callable
    ) -> Dict:
        """
        Run single experiment: one algorithm on one instance.

        Returns:
            Dictionary with all metrics
        """
        try:
            # Run algorithm
            solution = algorithm_func(instance)

            # Calculate KPIs
            report = self.calculator.calculate_kpis(
                solution, instance, algorithm_name
            )

            # Extract key metrics
            result = {
                'instance_name': instance.instance_name,
                'scenario': instance.instance_name.split('/')[0],
                'instance_number': int(instance.instance_name.split('instance')[1]),
                'algorithm': algorithm_name,
                'num_pallets': len(instance.pallets),
                'num_trucks': len(instance.outbound_trucks),
                **report.to_dict()
            }

            return result

        except Exception as e:
            logger.error(f"Error running {algorithm_name} on {instance.instance_name}: {str(e)}")
            return {
                'instance_name': instance.instance_name,
                'algorithm': algorithm_name,
                'status': 'ERROR',
                'error_message': str(e)
            }

    def run_scenario_benchmark(
        self,
        scenario: str,
        algorithms: Optional[List[str]] = None,
        max_instances: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Benchmark algorithms on all instances of a scenario.

        Args:
            scenario: Scenario name (e.g., 'MM_168h')
            algorithms: List of algorithm names (None = all registered)
            max_instances: Maximum number of instances to run (None = all 10)

        Returns:
            DataFrame with all results
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())

        logger.info(f"Benchmarking scenario {scenario} with {len(algorithms)} algorithms...")

        results = []
        instances = self.loader.load_all_instances(scenario)

        if max_instances:
            instances = instances[:max_instances]

        total_experiments = len(instances) * len(algorithms)
        pbar = tqdm(total=total_experiments, desc=f"{scenario}")

        for instance in instances:
            for algo_name in algorithms:
                algo_func = self.algorithms[algo_name]
                result = self.run_single_experiment(instance, algo_name, algo_func)
                results.append(result)
                pbar.update(1)

        pbar.close()

        df = pd.DataFrame(results)
        return df

    def run_full_benchmark(
        self,
        scenarios: Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None,
        max_instances_per_scenario: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run complete benchmark across all scenarios.

        Args:
            scenarios: List of scenarios (None = all 6)
            algorithms: List of algorithms (None = all registered)
            max_instances_per_scenario: Max instances per scenario

        Returns:
            DataFrame with all results
        """
        if scenarios is None:
            scenarios = ['HH_168h', 'MH_168h', 'MM_168h', 'LH_168h', 'LM_168h', 'LL_168h']

        if algorithms is None:
            algorithms = list(self.algorithms.keys())

        logger.info(f"\n{'='*80}")
        logger.info(f"FULL BENCHMARK")
        logger.info(f"  Scenarios: {len(scenarios)}")
        logger.info(f"  Algorithms: {len(algorithms)}")
        logger.info(f"  Max instances/scenario: {max_instances_per_scenario or 'all'}")
        logger.info(f"{'='*80}\n")

        all_results = []

        for scenario in scenarios:
            scenario_df = self.run_scenario_benchmark(
                scenario, algorithms, max_instances_per_scenario
            )
            all_results.append(scenario_df)

        # Combine all results
        full_df = pd.concat(all_results, ignore_index=True)

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"full_benchmark_{timestamp}.csv"
        full_df.to_csv(output_file, index=False)

        logger.info(f"\n✅ Benchmark complete! Results saved to: {output_file}")
        logger.info(f"Total experiments: {len(full_df)}")

        return full_df

    def generate_summary_report(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics from benchmark results.

        Args:
            results_df: Results from run_full_benchmark

        Returns:
            DataFrame with summary statistics
        """
        # Group by algorithm
        summary = results_df.groupby('algorithm').agg({
            'avg_fill_rate': ['mean', 'std'],
            'service_level': ['mean', 'std', 'min', 'max'],
            'num_late_pallets': ['mean', 'sum'],
            'unassigned_pallets': ['mean', 'sum'],
            'solve_time': ['mean', 'median', 'max'],
            'instance_name': 'count'  # Number of instances
        }).round(4)

        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"summary_report_{timestamp}.csv"
        summary.to_csv(output_file)

        logger.info(f"Summary report saved to: {output_file}")

        return summary

    def generate_scenario_comparison(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare algorithm performance across scenarios.

        Args:
            results_df: Results from run_full_benchmark

        Returns:
            DataFrame with scenario-level comparison
        """
        # Group by scenario and algorithm
        comparison = results_df.groupby(['scenario', 'algorithm']).agg({
            'service_level': 'mean',
            'avg_fill_rate': 'mean',
            'num_late_pallets': 'mean',
            'solve_time': 'mean',
            'num_pallets': 'mean'
        }).round(4)

        # Pivot for easier comparison
        pivoted = comparison.reset_index().pivot(
            index='scenario',
            columns='algorithm',
            values=['service_level', 'solve_time']
        )

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"scenario_comparison_{timestamp}.csv"
        pivoted.to_csv(output_file)

        logger.info(f"Scenario comparison saved to: {output_file}")

        return pivoted

    def find_best_algorithm(self, results_df: pd.DataFrame, metric: str = 'service_level') -> Dict:
        """
        Identify best algorithm based on specified metric.

        Args:
            results_df: Results DataFrame
            metric: Metric to optimize ('service_level', 'solve_time', etc.)

        Returns:
            Dictionary with best algorithm info
        """
        avg_performance = results_df.groupby('algorithm')[metric].agg(['mean', 'std', 'count'])

        if metric in ['service_level', 'avg_fill_rate']:
            # Higher is better
            best_algo = avg_performance['mean'].idxmax()
        else:
            # Lower is better (e.g., solve_time, late_pallets)
            best_algo = avg_performance['mean'].idxmin()

        best_stats = avg_performance.loc[best_algo]

        result = {
            'metric': metric,
            'best_algorithm': best_algo,
            'mean_value': best_stats['mean'],
            'std_value': best_stats['std'],
            'num_instances': int(best_stats['count'])
        }

        return result


# Convenience function for quick benchmarking
def quick_benchmark(
    scenario: str = 'LL_168h',
    max_instances: int = 3,
    output_dir: str = 'results/benchmarks'
) -> pd.DataFrame:
    """
    Quick benchmark for testing (runs on limited instances).

    Args:
        scenario: Scenario to test
        max_instances: Number of instances to run
        output_dir: Output directory

    Returns:
        Results DataFrame
    """
    pipeline = BenchmarkPipeline(output_dir=output_dir)

    results = pipeline.run_scenario_benchmark(
        scenario=scenario,
        max_instances=max_instances
    )

    print(f"\n{'='*80}")
    print(f"QUICK BENCHMARK RESULTS ({scenario}, {max_instances} instances)")
    print(f"{'='*80}")
    print(results[['algorithm', 'instance_name', 'service_level', 'solve_time']].to_string(index=False))
    print(f"{'='*80}\n")

    return results


# Testing
if __name__ == "__main__":
    import os

    # Change to project root
    os.chdir(Path(__file__).parent.parent.parent)

    print(f"\n{'='*80}")
    print(f"TESTING BENCHMARKING PIPELINE")
    print(f"{'='*80}\n")

    # Quick test
    print("Running quick benchmark (LL_168h, 3 instances, all heuristics)...")
    results = quick_benchmark(scenario='LL_168h', max_instances=3)

    # Generate summary
    pipeline = BenchmarkPipeline()
    summary = pipeline.generate_summary_report(results)

    print(f"\nSummary Statistics:")
    print(summary)

    # Find best algorithm
    best = pipeline.find_best_algorithm(results, metric='service_level')
    print(f"\nBest Algorithm (by service level): {best['best_algorithm']}")
    print(f"  Mean Service Level: {best['mean_value']:.2%}")

    print(f"\n✅ Benchmarking pipeline ready!")
    print(f"\nTo run full benchmark on all 60 instances:")
    print(f"  pipeline = BenchmarkPipeline()")
    print(f"  results = pipeline.run_full_benchmark()")
    print(f"  summary = pipeline.generate_summary_report(results)")
