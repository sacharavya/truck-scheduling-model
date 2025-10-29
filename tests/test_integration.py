"""
Integration tests for end-to-end workflows.

Tests complete workflows from data loading through optimization to analysis.
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.models.heuristics import earliest_due_date, first_fit, best_fit, destination_balanced
from src.models.pallet_assignment import optimize_pallet_assignment
from src.analysis.kpis import KPICalculator
from src.analysis.advanced_metrics import AdvancedMetricsCalculator, CostMetrics


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_basic_workflow(self):
        """Test basic workflow: load → optimize → analyze."""
        # 1. Load data
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        assert instance is not None
        assert len(instance.pallets) > 0

        # 2. Optimize
        solution = earliest_due_date(instance)

        assert solution is not None
        assert len(solution.pallet_to_truck) > 0

        # 3. Analyze
        calculator = KPICalculator()
        report = calculator.calculate_kpis(solution, instance, "EDD")

        assert report is not None
        assert report.service_level > 0

    def test_multiple_algorithms_workflow(self):
        """Test workflow with multiple algorithms."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        algorithms = {
            'EDD': earliest_due_date,
            'First-Fit': first_fit,
            'Best-Fit': best_fit,
            'Dest-Balanced': destination_balanced
        }

        calculator = KPICalculator()
        reports = {}

        for name, algo_func in algorithms.items():
            solution = algo_func(instance)
            report = calculator.calculate_kpis(solution, instance, name)
            reports[name] = report

        # All algorithms should complete
        assert len(reports) == 4

        # EDD should be best
        edd_sl = reports['EDD'].service_level
        assert all(edd_sl >= r.service_level for r in reports.values())

    def test_batch_processing(self):
        """Test batch processing of multiple instances."""
        loader = DataLoader()
        calculator = KPICalculator()

        # Process first 3 instances
        results = []
        for i in range(1, 4):
            instance = loader.load_instance('LL_168h', i)
            solution = earliest_due_date(instance)
            report = calculator.calculate_kpis(solution, instance, "EDD")
            results.append(report)

        # All should succeed
        assert len(results) == 3
        assert all(r.service_level >= 0.95 for r in results)

    def test_cross_scenario_workflow(self):
        """Test workflow across different scenarios."""
        loader = DataLoader()
        calculator = KPICalculator()

        scenarios = ['LL_168h', 'MM_168h', 'HH_168h']
        results = {}

        for scenario in scenarios:
            instance = loader.load_instance(scenario, 1)
            solution = earliest_due_date(instance)
            report = calculator.calculate_kpis(solution, instance, "EDD")
            results[scenario] = report

        # All should succeed
        assert len(results) == 3

        # EDD should maintain high service level across all
        assert all(r.service_level >= 0.95 for r in results.values())


class TestOptimizationToAnalysis:
    """Test integration between optimization and analysis."""

    @pytest.fixture
    def setup(self):
        """Set up instance and solution."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)
        solution = earliest_due_date(instance)
        return instance, solution

    def test_solution_analysis_integration(self, setup):
        """Test that solution can be analyzed."""
        instance, solution = setup

        calculator = KPICalculator()
        report = calculator.calculate_kpis(solution, instance, "EDD")

        # Report should reflect solution quality
        assert report.num_late_pallets == solution.num_late_pallets
        assert report.solve_time == solution.solve_time

    def test_advanced_metrics_integration(self, setup):
        """Test integration with advanced metrics."""
        instance, solution = setup

        calculator = AdvancedMetricsCalculator()

        # Should be able to calculate all metrics
        cost_metrics = calculator.calculate_cost_metrics(solution, instance)

        assert cost_metrics is not None
        assert cost_metrics.total_cost >= 0


class TestDataFlowIntegrity:
    """Test data flow integrity through the system."""

    def test_pallet_tracking(self):
        """Test that pallets are tracked correctly through workflow."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        original_pallet_ids = {p.pallet_id for p in instance.pallets}

        solution = earliest_due_date(instance)

        solution_pallet_ids = set(solution.pallet_to_truck.keys())

        # All pallets should be in solution
        assert original_pallet_ids == solution_pallet_ids

    def test_truck_capacity_integrity(self):
        """Test that truck capacities are respected throughout."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        solution = earliest_due_date(instance)

        # Count pallets per truck
        truck_loads = {}
        for pallet_id, truck_id in solution.pallet_to_truck.items():
            if truck_id == 'UNASSIGNED':
                continue
            truck_loads[truck_id] = truck_loads.get(truck_id, 0) + 1

        # Check against original capacities
        truck_capacities = {t.truck_id: t.capacity for t in instance.outbound_trucks}

        for truck_id, load in truck_loads.items():
            assert load <= truck_capacities[truck_id]

    def test_destination_consistency(self):
        """Test destination consistency through workflow."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        # Get original pallet destinations
        pallet_destinations = {p.pallet_id: p.destination for p in instance.pallets}
        truck_destinations = {t.truck_id: t.destination for t in instance.outbound_trucks}

        solution = earliest_due_date(instance)

        # Check all assignments match destinations
        for pallet_id, truck_id in solution.pallet_to_truck.items():
            if truck_id == 'UNASSIGNED':
                continue

            pallet_dest = pallet_destinations[pallet_id]
            truck_dest = truck_destinations[truck_id]

            assert pallet_dest == truck_dest


class TestScalability:
    """Test system scalability across instance sizes."""

    def test_small_to_large_instances(self):
        """Test processing from small to large instances."""
        loader = DataLoader()
        calculator = KPICalculator()

        test_cases = [
            ('LL_168h', 1, 'small'),
            ('MM_168h', 1, 'medium'),
            ('HH_168h', 1, 'large')
        ]

        results = []
        for scenario, instance_num, size in test_cases:
            instance = loader.load_instance(scenario, instance_num)
            solution = earliest_due_date(instance)
            report = calculator.calculate_kpis(solution, instance, "EDD")

            results.append({
                'size': size,
                'pallets': len(instance.pallets),
                'service_level': report.service_level,
                'solve_time': report.solve_time
            })

        # All should succeed
        assert len(results) == 3

        # Service level should remain high
        assert all(r['service_level'] >= 0.95 for r in results)

        # Solve time should scale reasonably (sub-linear)
        small_time = results[0]['solve_time']
        large_time = results[2]['solve_time']

        # Large instance is 4× bigger but should be less than 4× slower
        assert large_time < small_time * 4


class TestErrorHandling:
    """Test error handling in integrated workflows."""

    def test_invalid_instance_handling(self):
        """Test handling of invalid instance."""
        loader = DataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_instance('INVALID', 1)

    def test_empty_solution_handling(self):
        """Test handling of edge cases."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        # Solution should always be valid even in edge cases
        solution = earliest_due_date(instance)

        assert solution is not None
        assert len(solution.pallet_to_truck) > 0


class TestReproducibility:
    """Test reproducibility of results."""

    def test_same_instance_reproducibility(self):
        """Test that same instance produces same results."""
        loader = DataLoader()

        # Load same instance twice
        instance1 = loader.load_instance('LL_168h', 1)
        instance2 = loader.load_instance('LL_168h', 1)

        solution1 = earliest_due_date(instance1)
        solution2 = earliest_due_date(instance2)

        # Should produce identical results
        assert solution1.num_late_pallets == solution2.num_late_pallets

        # Service level should be identical
        calculator = KPICalculator()
        report1 = calculator.calculate_kpis(solution1, instance1, "EDD")
        report2 = calculator.calculate_kpis(solution2, instance2, "EDD")

        assert report1.service_level == report2.service_level

    def test_algorithm_determinism(self):
        """Test that algorithms are deterministic."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        # Run EDD multiple times
        solutions = [earliest_due_date(instance) for _ in range(3)]

        # All should produce identical results
        service_levels = [s.num_late_pallets for s in solutions]
        assert len(set(service_levels)) == 1  # All identical


class TestPerformanceBenchmark:
    """Test performance characteristics."""

    def test_edd_performance_target(self):
        """Test that EDD meets performance targets."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        solution = earliest_due_date(instance)

        # Should achieve 99%+ service level
        calculator = KPICalculator()
        report = calculator.calculate_kpis(solution, instance, "EDD")

        assert report.service_level >= 0.99

        # Should complete in under 100ms
        assert solution.solve_time < 0.1

    def test_batch_processing_speed(self):
        """Test batch processing performance."""
        import time

        loader = DataLoader()
        calculator = KPICalculator()

        start_time = time.time()

        # Process 10 instances
        for i in range(1, 11):
            instance = loader.load_instance('LL_168h', i)
            solution = earliest_due_date(instance)
            report = calculator.calculate_kpis(solution, instance, "EDD")

        elapsed = time.time() - start_time

        # Should complete 10 instances in under 5 seconds
        assert elapsed < 5.0


class TestSystemIntegration:
    """Test overall system integration."""

    def test_complete_system_workflow(self):
        """Test complete system from start to finish."""
        # 1. Initialize all components
        loader = DataLoader()
        calculator = KPICalculator()
        advanced_calc = AdvancedMetricsCalculator()

        # 2. Load data
        instance = loader.load_instance('LL_168h', 1)
        assert instance is not None

        # 3. Run multiple algorithms
        algorithms = {
            'EDD': earliest_due_date,
            'First-Fit': first_fit
        }

        results = {}
        for name, algo_func in algorithms.items():
            solution = algo_func(instance)
            report = calculator.calculate_kpis(solution, instance, name)
            cost = advanced_calc.calculate_cost_metrics(solution, instance)

            results[name] = {
                'solution': solution,
                'report': report,
                'cost': cost
            }

        # 4. Verify all succeeded
        assert len(results) == 2

        # 5. Verify EDD is best
        edd_sl = results['EDD']['report'].service_level
        ff_sl = results['First-Fit']['report'].service_level

        assert edd_sl >= ff_sl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
