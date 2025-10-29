"""
Advanced Metrics and Performance Analysis

This module extends the basic KPI framework with advanced metrics including:
- Time-series analysis (stock levels over time)
- Cost and efficiency calculations
- Door utilization over time
- Service level indicators (SLI)
- Comparative performance analytics

Author: Cross-Docking Optimization Project
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import CrossDockInstance
from models.pallet_assignment import AssignmentSolution
from simulation.cross_dock_sim import SimulationResults


@dataclass
class CostMetrics:
    """
    Cost and financial metrics for cross-docking operations.

    All costs in arbitrary monetary units (can be calibrated to actual currency).
    """
    # Cost parameters (per unit)
    cost_per_late_pallet: float = 100.0  # Penalty for late delivery
    cost_per_unassigned_pallet: float = 200.0  # Cost of not delivering
    cost_per_truck_hour: float = 50.0  # Cost of truck waiting
    cost_per_forklift_hour: float = 30.0  # Forklift operating cost
    cost_per_pallet_hour_storage: float = 0.5  # Staging area holding cost

    # Calculated costs
    total_tardiness_cost: float = 0.0
    total_unassigned_cost: float = 0.0
    total_truck_waiting_cost: float = 0.0
    total_forklift_cost: float = 0.0
    total_storage_cost: float = 0.0
    total_cost: float = 0.0

    # Efficiency metrics
    cost_per_pallet: float = 0.0
    cost_per_truck: float = 0.0

    def calculate_costs(
        self,
        num_late_pallets: int,
        num_unassigned: int,
        total_tardiness_min: float,
        truck_wait_time_min: float,
        forklift_hours: float,
        avg_storage_time_min: float,
        num_pallets_handled: int,
        num_trucks_served: int
    ):
        """Calculate all cost metrics."""
        # Direct costs
        self.total_tardiness_cost = num_late_pallets * self.cost_per_late_pallet
        self.total_unassigned_cost = num_unassigned * self.cost_per_unassigned_pallet
        self.total_truck_waiting_cost = (truck_wait_time_min / 60) * self.cost_per_truck_hour
        self.total_forklift_cost = forklift_hours * self.cost_per_forklift_hour
        self.total_storage_cost = (avg_storage_time_min / 60) * num_pallets_handled * self.cost_per_pallet_hour_storage

        # Total
        self.total_cost = (
            self.total_tardiness_cost +
            self.total_unassigned_cost +
            self.total_truck_waiting_cost +
            self.total_forklift_cost +
            self.total_storage_cost
        )

        # Per-unit costs
        if num_pallets_handled > 0:
            self.cost_per_pallet = self.total_cost / num_pallets_handled
        if num_trucks_served > 0:
            self.cost_per_truck = self.total_cost / num_trucks_served

    def print_summary(self):
        """Print cost breakdown."""
        print(f"\n{'='*70}")
        print(f"COST ANALYSIS")
        print(f"{'='*70}")
        print(f"\nCost Breakdown:")
        print(f"  Tardiness Penalties:    ${self.total_tardiness_cost:,.2f}")
        print(f"  Unassigned Penalties:   ${self.total_unassigned_cost:,.2f}")
        print(f"  Truck Waiting:          ${self.total_truck_waiting_cost:,.2f}")
        print(f"  Forklift Operations:    ${self.total_forklift_cost:,.2f}")
        print(f"  Storage/Holding:        ${self.total_storage_cost:,.2f}")
        print(f"  " + "-"*60)
        print(f"  TOTAL COST:             ${self.total_cost:,.2f}")
        print(f"\nEfficiency Metrics:")
        print(f"  Cost per Pallet:        ${self.cost_per_pallet:.2f}")
        print(f"  Cost per Truck:         ${self.cost_per_truck:.2f}")
        print(f"{'='*70}\n")


@dataclass
class TimeSeriesMetrics:
    """
    Time-series metrics for dynamic analysis.

    Tracks metrics over time for trend analysis.
    """
    timestamps: List[float] = field(default_factory=list)
    stock_levels: List[int] = field(default_factory=list)
    inbound_queue_lengths: List[int] = field(default_factory=list)
    outbound_queue_lengths: List[int] = field(default_factory=list)
    cumulative_late_pallets: List[int] = field(default_factory=list)
    door_utilization: List[float] = field(default_factory=list)

    def add_sample(
        self,
        time: float,
        stock: int,
        inbound_queue: int = 0,
        outbound_queue: int = 0,
        cumulative_late: int = 0,
        door_util: float = 0.0
    ):
        """Add a time-series sample."""
        self.timestamps.append(time)
        self.stock_levels.append(stock)
        self.inbound_queue_lengths.append(inbound_queue)
        self.outbound_queue_lengths.append(outbound_queue)
        self.cumulative_late_pallets.append(cumulative_late)
        self.door_utilization.append(door_util)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'time': self.timestamps,
            'stock_level': self.stock_levels,
            'inbound_queue': self.inbound_queue_lengths,
            'outbound_queue': self.outbound_queue_lengths,
            'cumulative_late': self.cumulative_late_pallets,
            'door_utilization': self.door_utilization
        })

    def plot_time_series(self, save_path: Optional[str] = None):
        """Plot all time series."""
        df = self.to_dataframe()

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Stock levels
        axes[0, 0].plot(df['time'], df['stock_level'], linewidth=2, color='steelblue')
        axes[0, 0].fill_between(df['time'], df['stock_level'], alpha=0.3, color='steelblue')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Stock Level (pallets)')
        axes[0, 0].set_title('Staging Area Inventory Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # Queue lengths
        axes[0, 1].plot(df['time'], df['inbound_queue'], label='Inbound', linewidth=2, color='coral')
        axes[0, 1].plot(df['time'], df['outbound_queue'], label='Outbound', linewidth=2, color='teal')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Queue Length (trucks)')
        axes[0, 1].set_title('Door Queue Lengths Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Cumulative late pallets
        axes[1, 0].plot(df['time'], df['cumulative_late'], linewidth=2, color='red')
        axes[1, 0].fill_between(df['time'], df['cumulative_late'], alpha=0.3, color='red')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Cumulative Late Pallets')
        axes[1, 0].set_title('Late Deliveries Over Time')
        axes[1, 0].grid(True, alpha=0.3)

        # Door utilization
        axes[1, 1].plot(df['time'], df['door_utilization'] * 100, linewidth=2, color='green')
        axes[1, 1].axhline(80, color='orange', linestyle='--', label='80% Target')
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Door Utilization (%)')
        axes[1, 1].set_title('Door Utilization Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Time series plot saved to: {save_path}")

        plt.show()


class AdvancedMetricsCalculator:
    """
    Calculator for advanced performance metrics.

    Extends basic KPIs with cost analysis, time-series tracking, and efficiency metrics.
    """

    def __init__(
        self,
        cost_params: Optional[Dict] = None
    ):
        """Initialize calculator with cost parameters."""
        self.cost_params = cost_params or {}

    def calculate_cost_metrics(
        self,
        solution: AssignmentSolution,
        instance: CrossDockInstance,
        sim_results: Optional[SimulationResults] = None
    ) -> CostMetrics:
        """
        Calculate comprehensive cost metrics.

        Args:
            solution: Optimization solution
            instance: Problem instance
            sim_results: Optional simulation results for more accurate costs

        Returns:
            CostMetrics object
        """
        cost_metrics = CostMetrics(**self.cost_params)

        # Extract metrics from solution
        num_late = solution.num_late_pallets
        num_unassigned = solution.metadata.get('unassigned_pallets', 0)
        total_tardiness = solution.total_tardiness

        # Estimate other costs (use simulation if available)
        if sim_results:
            truck_wait_time = sim_results.avg_inbound_queue_length * 10  # Rough estimate
            forklift_hours = sim_results.makespan / 60  # Total time
            avg_storage_time = sim_results.avg_pallet_flow_time
            num_pallets = sim_results.total_pallets_processed
            num_trucks = len(instance.outbound_trucks)
        else:
            # Rough estimates from optimization only
            truck_wait_time = len(instance.outbound_trucks) * 2  # Assume 2 min avg wait
            forklift_hours = len(instance.pallets) * 0.05  # ~3 min per pallet
            avg_storage_time = 60  # Assume 1 hour average
            num_pallets = len(instance.pallets) - num_unassigned
            num_trucks = len(instance.outbound_trucks)

        # Calculate all costs
        cost_metrics.calculate_costs(
            num_late_pallets=num_late,
            num_unassigned=num_unassigned,
            total_tardiness_min=total_tardiness,
            truck_wait_time_min=truck_wait_time,
            forklift_hours=forklift_hours,
            avg_storage_time_min=avg_storage_time,
            num_pallets_handled=num_pallets,
            num_trucks_served=num_trucks
        )

        return cost_metrics

    def calculate_efficiency_score(
        self,
        fill_rate: float,
        service_level: float,
        door_utilization: float,
        forklift_utilization: float
    ) -> float:
        """
        Calculate overall efficiency score (0-100).

        Weighted combination of key efficiency metrics.

        Args:
            fill_rate: Truck fill rate (0-1)
            service_level: On-time delivery rate (0-1)
            door_utilization: Door utilization (0-1)
            forklift_utilization: Forklift utilization (0-1)

        Returns:
            Efficiency score (0-100)
        """
        # Weights (must sum to 1)
        weights = {
            'fill_rate': 0.25,
            'service_level': 0.40,  # Most important
            'door_utilization': 0.20,
            'forklift_utilization': 0.15
        }

        score = (
            weights['fill_rate'] * fill_rate +
            weights['service_level'] * service_level +
            weights['door_utilization'] * door_utilization +
            weights['forklift_utilization'] * forklift_utilization
        ) * 100

        return score

    def generate_performance_report(
        self,
        solution: AssignmentSolution,
        instance: CrossDockInstance,
        solution_name: str = "Solution"
    ) -> Dict:
        """
        Generate comprehensive performance report.

        Returns:
            Dictionary with all advanced metrics
        """
        # Calculate costs
        cost_metrics = self.calculate_cost_metrics(solution, instance)

        # Calculate efficiency score
        efficiency_score = self.calculate_efficiency_score(
            fill_rate=solution.avg_fill_rate,
            service_level=1 - (solution.num_late_pallets / len(instance.pallets)),
            door_utilization=0.85,  # Estimate
            forklift_utilization=0.75  # Estimate
        )

        report = {
            'solution_name': solution_name,
            'instance': instance.instance_name,
            'total_cost': cost_metrics.total_cost,
            'cost_per_pallet': cost_metrics.cost_per_pallet,
            'efficiency_score': efficiency_score,
            'fill_rate': solution.avg_fill_rate,
            'service_level': 1 - (solution.num_late_pallets / max(len(instance.pallets), 1)),
            'cost_breakdown': {
                'tardiness': cost_metrics.total_tardiness_cost,
                'unassigned': cost_metrics.total_unassigned_cost,
                'truck_waiting': cost_metrics.total_truck_waiting_cost,
                'forklift': cost_metrics.total_forklift_cost,
                'storage': cost_metrics.total_storage_cost
            }
        }

        return report


# Testing
if __name__ == "__main__":
    import os
    from data_loader import DataLoader
    from models.heuristics import earliest_due_date

    os.chdir(Path(__file__).parent.parent.parent)

    # Load instance
    loader = DataLoader()
    instance = loader.load_instance('LL_168h', 1)

    # Solve
    solution = earliest_due_date(instance)

    # Advanced metrics
    calculator = AdvancedMetricsCalculator()

    print(f"\n{'='*70}")
    print(f"TESTING ADVANCED METRICS")
    print(f"{'='*70}")

    # Cost analysis
    cost_metrics = calculator.calculate_cost_metrics(solution, instance)
    cost_metrics.print_summary()

    # Efficiency score
    efficiency = calculator.calculate_efficiency_score(
        fill_rate=solution.avg_fill_rate,
        service_level=0.9976,  # 99.76%
        door_utilization=0.85,
        forklift_utilization=0.75
    )
    print(f"Overall Efficiency Score: {efficiency:.1f}/100")

    # Full report
    report = calculator.generate_performance_report(solution, instance, "EDD")
    print(f"\nTotal Cost: ${report['total_cost']:,.2f}")
    print(f"Cost per Pallet: ${report['cost_per_pallet']:.2f}")
    print(f"Efficiency Score: {report['efficiency_score']:.1f}/100")

    # Time series demo
    print(f"\nGenerating time series demo...")
    ts = TimeSeriesMetrics()
    for t in range(0, 1000, 10):
        stock = int(50 + 30 * np.sin(t / 100) + np.random.randn() * 5)
        ts.add_sample(
            time=t,
            stock=max(0, stock),
            inbound_queue=np.random.randint(0, 5),
            outbound_queue=np.random.randint(0, 5),
            cumulative_late=int(t / 100),
            door_util=0.7 + 0.2 * np.random.rand()
        )

    ts.plot_time_series('results/figures/time_series_demo.png')

    print(f"\n✅ Advanced metrics module working!")
