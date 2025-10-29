"""
Simulation-Optimization Integration

This module integrates optimization solutions with discrete-event simulation
to validate and evaluate optimization decisions under realistic conditions.

Key features:
- Feed optimization solutions into simulation
- Validate MILP/heuristic schedules
- Test robustness to uncertainties
- Compare optimization vs simulation KPIs

Author: Cross-Docking Optimization Project
"""

import simpy
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import CrossDockInstance
from models.pallet_assignment import AssignmentSolution
from simulation.cross_dock_sim import CrossDockSimulation, SimulationConfig, SimulationResults
from analysis.kpis import KPIReport, KPICalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """
    Results from validating optimization solution via simulation.

    Compares optimization predictions with simulation outcomes.
    """
    instance_name: str
    solution_method: str

    # Optimization metrics
    opt_fill_rate: float
    opt_service_level: float
    opt_late_pallets: int
    opt_solve_time: float

    # Simulation metrics
    sim_service_level: float
    sim_late_pallets: int
    sim_avg_flow_time: float
    sim_makespan: float
    sim_avg_staging_inventory: float
    sim_forklift_utilization: float

    # Comparison
    service_level_diff: float  # sim - opt
    late_pallets_diff: int

    def print_summary(self):
        """Print validation summary."""
        print(f"\n{'='*80}")
        print(f"VALIDATION RESULTS: {self.solution_method}")
        print(f"Instance: {self.instance_name}")
        print(f"{'='*80}")

        print(f"\n📊 OPTIMIZATION PREDICTIONS")
        print(f"  Fill Rate: {self.opt_fill_rate:.2%}")
        print(f"  Service Level: {self.opt_service_level:.2%}")
        print(f"  Late Pallets: {self.opt_late_pallets}")
        print(f"  Solve Time: {self.opt_solve_time:.3f}s")

        print(f"\n🔬 SIMULATION OUTCOMES")
        print(f"  Service Level: {self.sim_service_level:.2%}")
        print(f"  Late Pallets: {self.sim_late_pallets}")
        print(f"  Avg Flow Time: {self.sim_avg_flow_time:.2f} min")
        print(f"  Makespan: {self.sim_makespan:.2f} min")
        print(f"  Avg Staging Inventory: {self.sim_avg_staging_inventory:.2f} pallets")
        print(f"  Forklift Utilization: {self.sim_forklift_utilization:.2%}")

        print(f"\n🔍 COMPARISON (Simulation vs Optimization)")
        print(f"  Service Level Diff: {self.service_level_diff:+.2%}")
        print(f"  Late Pallets Diff: {self.late_pallets_diff:+d}")

        if abs(self.service_level_diff) < 0.01:
            print(f"  ✅ Excellent match! Simulation validates optimization.")
        elif abs(self.service_level_diff) < 0.05:
            print(f"  ✓ Good match. Minor differences acceptable.")
        else:
            print(f"  ⚠️  Significant difference. Further investigation needed.")

        print(f"{'='*80}\n")


class OptimizationValidator:
    """
    Validates optimization solutions through simulation.

    Runs simulation with optimized assignments and compares results.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize validator with simulation configuration."""
        self.config = config or SimulationConfig()
        self.kpi_calculator = KPICalculator()

    def validate_solution(
        self,
        instance: CrossDockInstance,
        solution: AssignmentSolution,
        solution_method: str = "Unknown"
    ) -> ValidationResults:
        """
        Validate optimization solution through simulation.

        Args:
            instance: Cross-dock instance
            solution: Optimization solution to validate
            solution_method: Name of optimization method

        Returns:
            ValidationResults with comparison
        """
        logger.info(f"Validating {solution_method} solution via simulation...")

        # Get optimization KPIs
        opt_report = self.kpi_calculator.calculate_kpis(
            solution, instance, solution_method
        )

        # Run simulation
        sim = CrossDockSimulation(instance, solution, self.config)
        sim_results = sim.run()

        # Calculate simulation service level
        total_processed = sim_results.total_pallets_processed
        if total_processed > 0:
            sim_service_level = sim_results.pallets_delivered_on_time / total_processed
        else:
            sim_service_level = 0.0

        # Create validation results
        validation = ValidationResults(
            instance_name=instance.instance_name,
            solution_method=solution_method,
            opt_fill_rate=opt_report.avg_fill_rate,
            opt_service_level=opt_report.service_level,
            opt_late_pallets=opt_report.num_late_pallets,
            opt_solve_time=opt_report.solve_time,
            sim_service_level=sim_service_level,
            sim_late_pallets=sim_results.pallets_delivered_late,
            sim_avg_flow_time=sim_results.avg_pallet_flow_time,
            sim_makespan=sim_results.makespan,
            sim_avg_staging_inventory=sim_results.avg_staging_inventory,
            sim_forklift_utilization=sim_results.avg_forklift_utilization,
            service_level_diff=sim_service_level - opt_report.service_level,
            late_pallets_diff=sim_results.pallets_delivered_late - opt_report.num_late_pallets
        )

        logger.info(f"Validation complete. Service level diff: {validation.service_level_diff:+.2%}")

        return validation

    def validate_multiple_solutions(
        self,
        instance: CrossDockInstance,
        solutions: Dict[str, AssignmentSolution]
    ) -> pd.DataFrame:
        """
        Validate multiple solutions and compare.

        Args:
            instance: Cross-dock instance
            solutions: Dict mapping method name to solution

        Returns:
            DataFrame with comparison of all solutions
        """
        results = []

        for method_name, solution in solutions.items():
            validation = self.validate_solution(instance, solution, method_name)
            results.append({
                'method': method_name,
                'opt_service_level': validation.opt_service_level,
                'sim_service_level': validation.sim_service_level,
                'service_diff': validation.service_level_diff,
                'opt_late': validation.opt_late_pallets,
                'sim_late': validation.sim_late_pallets,
                'sim_flow_time': validation.sim_avg_flow_time,
                'sim_makespan': validation.sim_makespan,
                'opt_solve_time': validation.opt_solve_time
            })

        df = pd.DataFrame(results)
        return df


class RobustnessAnalyzer:
    """
    Analyzes robustness of solutions to uncertainty.

    Tests solutions under various perturbations (delays, processing time variability, etc.)
    """

    def __init__(self, base_config: Optional[SimulationConfig] = None):
        """Initialize analyzer."""
        self.base_config = base_config or SimulationConfig()

    def test_robustness_to_delays(
        self,
        instance: CrossDockInstance,
        solution: AssignmentSolution,
        delay_percentage: float = 0.1
    ) -> Dict:
        """
        Test solution robustness to arrival delays.

        Args:
            instance: Cross-dock instance
            solution: Optimization solution
            delay_percentage: Percentage of trucks to delay (0-1)

        Returns:
            Dictionary with robustness metrics
        """
        # TODO: Implement arrival delay perturbations
        # This would require modifying truck arrival times in the instance
        pass

    def test_robustness_to_processing_variability(
        self,
        instance: CrossDockInstance,
        solution: AssignmentSolution,
        variability_factor: float = 0.2
    ) -> Dict:
        """
        Test solution robustness to processing time variability.

        Args:
            instance: Cross-dock instance
            solution: Optimization solution
            variability_factor: Coefficient of variation for processing times

        Returns:
            Dictionary with robustness metrics
        """
        # TODO: Implement processing time variability
        # Would modify pallet_unload_time, pallet_load_time in config
        pass


# Testing and demonstration
if __name__ == "__main__":
    import os
    from data_loader import DataLoader
    from models.heuristics import earliest_due_date, first_fit

    # Change to project root
    os.chdir(Path(__file__).parent.parent.parent)

    # Load instance
    loader = DataLoader()
    instance = loader.load_instance('LL_168h', 1)

    print(f"\n{'='*80}")
    print(f"TESTING OPTIMIZATION-SIMULATION INTEGRATION")
    print(f"{'='*80}")
    print(f"Instance: {instance.instance_name}")
    print(f"Pallets: {len(instance.pallets)}, Trucks: {len(instance.outbound_trucks)}")

    # Solve with two methods
    print(f"\nSolving with optimization methods...")
    edd_solution = earliest_due_date(instance)
    ff_solution = first_fit(instance)

    # Create validator
    validator = OptimizationValidator()

    # Validate EDD
    print(f"\n{'='*80}")
    print(f"VALIDATING EDD HEURISTIC")
    print(f"{'='*80}")
    edd_validation = validator.validate_solution(instance, edd_solution, "EDD")
    edd_validation.print_summary()

    # Validate First-Fit
    print(f"\n{'='*80}")
    print(f"VALIDATING FIRST-FIT HEURISTIC")
    print(f"{'='*80}")
    ff_validation = validator.validate_solution(instance, ff_solution, "First-Fit")
    ff_validation.print_summary()

    # Compare multiple solutions
    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE")
    print(f"{'='*80}")
    comparison = validator.validate_multiple_solutions(
        instance,
        {'EDD': edd_solution, 'First-Fit': ff_solution}
    )
    print(comparison.to_string(index=False))

    print(f"\n✅ Integration testing complete!")
