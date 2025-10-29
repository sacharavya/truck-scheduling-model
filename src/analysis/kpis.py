"""
KPI (Key Performance Indicator) Calculation Module

This module provides standardized performance metrics for evaluating
cross-docking terminal operations and optimization solutions.

KPIs include:
- Fill rate (truck capacity utilization)
- Tardiness (late deliveries)
- Service level
- Resource utilization
- Computational performance

Author: Cross-Docking Optimization Project
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import CrossDockInstance
from models.pallet_assignment import AssignmentSolution


@dataclass
class KPIReport:
    """
    Comprehensive KPI report for a solution.

    Attributes:
        solution_name: Name/identifier of the solution
        instance_name: Name of the instance

        # Capacity/Utilization KPIs
        avg_fill_rate: Average truck fill rate (0-1)
        min_fill_rate: Minimum truck fill rate
        max_fill_rate: Maximum truck fill rate
        std_fill_rate: Standard deviation of fill rates
        trucks_utilized: Number of trucks with at least one pallet
        total_truck_capacity: Total available capacity
        utilized_capacity: Total utilized capacity

        # Tardiness/Service Level KPIs
        num_late_pallets: Number of pallets delivered late
        pct_late_pallets: Percentage of pallets delivered late
        total_tardiness: Total tardiness in minutes
        avg_tardiness: Average tardiness per late pallet
        max_tardiness: Maximum tardiness of any pallet
        service_level: Percentage of on-time deliveries (1 - pct_late)

        # Assignment KPIs
        total_pallets: Total number of pallets
        assigned_pallets: Number of pallets assigned to trucks
        unassigned_pallets: Number of pallets not assigned
        pct_unassigned: Percentage of pallets unassigned

        # Performance KPIs
        solve_time: Computation time in seconds
        status: Solver status
        objective_value: Objective function value (if applicable)

        # By-destination breakdown
        fill_rate_by_dest: Dict mapping destination to fill rate
        late_pallets_by_dest: Dict mapping destination to late count
    """
    solution_name: str
    instance_name: str

    # Utilization
    avg_fill_rate: float = 0.0
    min_fill_rate: float = 0.0
    max_fill_rate: float = 0.0
    std_fill_rate: float = 0.0
    trucks_utilized: int = 0
    total_truck_capacity: int = 0
    utilized_capacity: int = 0

    # Tardiness
    num_late_pallets: int = 0
    pct_late_pallets: float = 0.0
    total_tardiness: float = 0.0
    avg_tardiness: float = 0.0
    max_tardiness: float = 0.0
    service_level: float = 0.0

    # Assignment
    total_pallets: int = 0
    assigned_pallets: int = 0
    unassigned_pallets: int = 0
    pct_unassigned: float = 0.0

    # Performance
    solve_time: float = 0.0
    status: str = "UNKNOWN"
    objective_value: float = 0.0

    # Breakdowns
    fill_rate_by_dest: Dict[int, float] = field(default_factory=dict)
    late_pallets_by_dest: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'solution_name': self.solution_name,
            'instance_name': self.instance_name,
            'avg_fill_rate': self.avg_fill_rate,
            'min_fill_rate': self.min_fill_rate,
            'max_fill_rate': self.max_fill_rate,
            'std_fill_rate': self.std_fill_rate,
            'trucks_utilized': self.trucks_utilized,
            'total_truck_capacity': self.total_truck_capacity,
            'utilized_capacity': self.utilized_capacity,
            'num_late_pallets': self.num_late_pallets,
            'pct_late_pallets': self.pct_late_pallets,
            'total_tardiness': self.total_tardiness,
            'avg_tardiness': self.avg_tardiness,
            'max_tardiness': self.max_tardiness,
            'service_level': self.service_level,
            'total_pallets': self.total_pallets,
            'assigned_pallets': self.assigned_pallets,
            'unassigned_pallets': self.unassigned_pallets,
            'pct_unassigned': self.pct_unassigned,
            'solve_time': self.solve_time,
            'status': self.status,
            'objective_value': self.objective_value
        }

    def print_report(self):
        """Print formatted KPI report."""
        print(f"\n{'='*80}")
        print(f"KPI REPORT: {self.solution_name}")
        print(f"Instance: {self.instance_name}")
        print(f"{'='*80}")

        print(f"\n📦 CAPACITY & UTILIZATION")
        print(f"  Average Fill Rate:        {self.avg_fill_rate:.2%}")
        print(f"  Min/Max Fill Rate:        {self.min_fill_rate:.2%} / {self.max_fill_rate:.2%}")
        print(f"  Std Dev Fill Rate:        {self.std_fill_rate:.4f}")
        print(f"  Trucks Utilized:          {self.trucks_utilized}")
        print(f"  Total Capacity:           {self.total_truck_capacity} pallets")
        print(f"  Utilized Capacity:        {self.utilized_capacity} pallets ({self.utilized_capacity/self.total_truck_capacity:.2%})")

        print(f"\n⏰ TARDINESS & SERVICE LEVEL")
        print(f"  Late Pallets:             {self.num_late_pallets} ({self.pct_late_pallets:.2%})")
        print(f"  Total Tardiness:          {self.total_tardiness:.2f} minutes")
        print(f"  Average Tardiness:        {self.avg_tardiness:.2f} minutes")
        print(f"  Maximum Tardiness:        {self.max_tardiness:.2f} minutes")
        print(f"  Service Level (On-Time):  {self.service_level:.2%}")

        print(f"\n📋 ASSIGNMENT")
        print(f"  Total Pallets:            {self.total_pallets}")
        print(f"  Assigned Pallets:         {self.assigned_pallets}")
        print(f"  Unassigned Pallets:       {self.unassigned_pallets} ({self.pct_unassigned:.2%})")

        print(f"\n⚡ PERFORMANCE")
        print(f"  Solve Time:               {self.solve_time:.3f} seconds")
        print(f"  Status:                   {self.status}")
        if self.objective_value != 0:
            print(f"  Objective Value:          {self.objective_value:.2f}")

        if self.fill_rate_by_dest:
            print(f"\n🎯 BY DESTINATION")
            for dest in sorted(self.fill_rate_by_dest.keys()):
                fill = self.fill_rate_by_dest[dest]
                late = self.late_pallets_by_dest.get(dest, 0)
                print(f"  Destination {dest}:          Fill Rate: {fill:.2%}, Late: {late}")

        print(f"{'='*80}\n")


class KPICalculator:
    """Calculator for performance metrics."""

    TRUCK_CAPACITY = 26

    @staticmethod
    def calculate_kpis(
        solution: AssignmentSolution,
        instance: CrossDockInstance,
        solution_name: str = "Solution"
    ) -> KPIReport:
        """
        Calculate comprehensive KPIs for a solution.

        Args:
            solution: Assignment solution to evaluate
            instance: Original instance data
            solution_name: Name for the solution

        Returns:
            KPIReport with all calculated metrics
        """
        truck_dict = {t.truck_id: t for t in instance.outbound_trucks}
        pallet_dict = {p.pallet_id: p for p in instance.pallets}

        report = KPIReport(
            solution_name=solution_name,
            instance_name=instance.instance_name,
            status=solution.status,
            solve_time=solution.solve_time,
            objective_value=solution.objective_value if hasattr(solution, 'objective_value') else 0.0
        )

        # Calculate fill rates
        fill_rates = []
        fill_by_dest = {1: [], 2: [], 3: []}

        for truck_id, pallet_ids in solution.truck_loads.items():
            truck = truck_dict[truck_id]
            fill_rate = len(pallet_ids) / KPICalculator.TRUCK_CAPACITY
            fill_rates.append(fill_rate)
            fill_by_dest[truck.destination].append(fill_rate)

        if fill_rates:
            report.avg_fill_rate = np.mean(fill_rates)
            report.min_fill_rate = np.min(fill_rates)
            report.max_fill_rate = np.max(fill_rates)
            report.std_fill_rate = np.std(fill_rates)

        # Fill rate by destination
        for dest, rates in fill_by_dest.items():
            if rates:
                report.fill_rate_by_dest[dest] = np.mean(rates)

        report.trucks_utilized = len(solution.truck_loads)
        report.total_truck_capacity = len(instance.outbound_trucks) * KPICalculator.TRUCK_CAPACITY
        report.utilized_capacity = sum(len(loads) for loads in solution.truck_loads.values())

        # Calculate tardiness
        late_by_dest = {1: 0, 2: 0, 3: 0}
        tardiness_values = []

        for truck_id, pallet_ids in solution.truck_loads.items():
            truck = truck_dict[truck_id]
            for pid in pallet_ids:
                pallet = pallet_dict[pid]
                if pallet.due_date < truck.due_date:
                    tardiness = truck.due_date - pallet.due_date
                    report.num_late_pallets += 1
                    report.total_tardiness += tardiness
                    tardiness_values.append(tardiness)
                    late_by_dest[pallet.destination] += 1

        report.late_pallets_by_dest = late_by_dest

        if tardiness_values:
            report.avg_tardiness = np.mean(tardiness_values)
            report.max_tardiness = np.max(tardiness_values)

        # Assignment metrics
        report.total_pallets = len(instance.pallets)
        report.assigned_pallets = report.utilized_capacity
        report.unassigned_pallets = solution.metadata.get('unassigned_pallets', 0)

        if report.total_pallets > 0:
            report.pct_late_pallets = report.num_late_pallets / report.assigned_pallets if report.assigned_pallets > 0 else 0
            report.pct_unassigned = report.unassigned_pallets / report.total_pallets
            report.service_level = 1.0 - report.pct_late_pallets

        return report

    @staticmethod
    def compare_solutions(
        solutions: List[tuple],  # List of (name, solution, instance) tuples
    ) -> pd.DataFrame:
        """
        Compare multiple solutions side-by-side.

        Args:
            solutions: List of (solution_name, solution, instance) tuples

        Returns:
            DataFrame with comparison of all solutions
        """
        reports = []
        for name, solution, instance in solutions:
            report = KPICalculator.calculate_kpis(solution, instance, name)
            reports.append(report.to_dict())

        df = pd.DataFrame(reports)
        return df

    @staticmethod
    def aggregate_results(
        results: List[KPIReport]
    ) -> Dict:
        """
        Aggregate KPI results across multiple instances.

        Args:
            results: List of KPIReport objects

        Returns:
            Dictionary with aggregated statistics
        """
        if not results:
            return {}

        aggregated = {
            'num_instances': len(results),
            'avg_fill_rate': np.mean([r.avg_fill_rate for r in results]),
            'avg_service_level': np.mean([r.service_level for r in results]),
            'avg_tardiness': np.mean([r.total_tardiness for r in results]),
            'avg_unassigned_pct': np.mean([r.pct_unassigned for r in results]),
            'avg_solve_time': np.mean([r.solve_time for r in results]),
            'total_late_pallets': sum([r.num_late_pallets for r in results]),
            'total_unassigned': sum([r.unassigned_pallets for r in results]),
            'total_pallets': sum([r.total_pallets for r in results])
        }

        return aggregated


# Testing
if __name__ == "__main__":
    import os
    from data_loader import DataLoader
    from models.heuristics import first_fit, earliest_due_date
    from models.pallet_assignment import optimize_pallet_assignment

    # Change to project root
    os.chdir(Path(__file__).parent.parent.parent)

    # Load test instance
    loader = DataLoader()
    instance = loader.load_instance('LL_168h', 1)

    print(f"\n{'='*80}")
    print(f"TESTING KPI CALCULATION")
    print(f"{'='*80}")
    print(f"Instance: {instance.instance_name}")
    print(f"Pallets: {len(instance.pallets)}, Trucks: {len(instance.outbound_trucks)}")

    # Test with multiple solutions
    print(f"\nRunning solutions...")
    ff_solution = first_fit(instance)
    edd_solution = earliest_due_date(instance)

    # Calculate KPIs
    calculator = KPICalculator()

    ff_report = calculator.calculate_kpis(ff_solution, instance, "First-Fit")
    ff_report.print_report()

    edd_report = calculator.calculate_kpis(edd_solution, instance, "EDD")
    edd_report.print_report()

    # Compare
    print(f"\n{'='*80}")
    print(f"SOLUTION COMPARISON")
    print(f"{'='*80}")
    comparison = calculator.compare_solutions([
        ("First-Fit", ff_solution, instance),
        ("EDD", edd_solution, instance)
    ])

    print(comparison[['solution_name', 'avg_fill_rate', 'service_level',
                      'num_late_pallets', 'unassigned_pallets', 'solve_time']].to_string(index=False))
