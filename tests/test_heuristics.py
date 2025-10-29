"""
Unit tests for heuristic algorithms.

Tests all four heuristic algorithms: EDD, First-Fit, Best-Fit, Destination-Balanced.
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.models.heuristics import (
    earliest_due_date,
    first_fit,
    best_fit,
    destination_balanced
)
from src.models.pallet_assignment import AssignmentSolution


class TestHeuristicAlgorithms:
    """Test suite for all heuristic algorithms."""

    @pytest.fixture
    def small_instance(self):
        """Load a small instance for testing."""
        loader = DataLoader()
        return loader.load_instance('LL_168h', 1)

    @pytest.fixture
    def medium_instance(self):
        """Load a medium instance for testing."""
        loader = DataLoader()
        return loader.load_instance('MM_168h', 1)

    def test_edd_basic(self, small_instance):
        """Test EDD heuristic basic functionality."""
        solution = earliest_due_date(small_instance)

        assert isinstance(solution, AssignmentSolution)
        assert len(solution.pallet_to_truck) > 0
        assert solution.solve_time >= 0

    def test_first_fit_basic(self, small_instance):
        """Test First-Fit heuristic basic functionality."""
        solution = first_fit(small_instance)

        assert isinstance(solution, AssignmentSolution)
        assert len(solution.pallet_to_truck) > 0
        assert solution.solve_time >= 0

    def test_best_fit_basic(self, small_instance):
        """Test Best-Fit heuristic basic functionality."""
        solution = best_fit(small_instance)

        assert isinstance(solution, AssignmentSolution)
        assert len(solution.pallet_to_truck) > 0
        assert solution.solve_time >= 0

    def test_destination_balanced_basic(self, small_instance):
        """Test Destination-Balanced heuristic basic functionality."""
        solution = destination_balanced(small_instance)

        assert isinstance(solution, AssignmentSolution)
        assert len(solution.pallet_to_truck) > 0
        assert solution.solve_time >= 0

    def test_all_algorithms_run(self, small_instance):
        """Test that all algorithms complete successfully."""
        algorithms = [
            ('EDD', earliest_due_date),
            ('First-Fit', first_fit),
            ('Best-Fit', best_fit),
            ('Dest-Balanced', destination_balanced)
        ]

        for name, algo_func in algorithms:
            solution = algo_func(small_instance)
            assert solution is not None, f"{name} returned None"
            assert isinstance(solution, AssignmentSolution), f"{name} returned wrong type"


class TestEDDHeuristic:
    """Test suite specifically for EDD heuristic."""

    @pytest.fixture
    def instance(self):
        """Load instance for testing."""
        loader = DataLoader()
        return loader.load_instance('LL_168h', 1)

    def test_edd_assigns_all_eligible_pallets(self, instance):
        """Test that EDD assigns all pallets that can be assigned."""
        solution = earliest_due_date(instance)

        # Should assign most pallets (allowing for capacity constraints)
        assigned_count = len([p for p in solution.pallet_to_truck.values() if p != 'UNASSIGNED'])
        total_pallets = len(instance.pallets)

        # At least 95% should be assigned
        assert assigned_count / total_pallets >= 0.95

    def test_edd_respects_destinations(self, instance):
        """Test that EDD only assigns pallets to matching destinations."""
        solution = earliest_due_date(instance)

        # Create destination map for trucks
        truck_destinations = {t.truck_id: t.destination for t in instance.outbound_trucks}
        pallet_destinations = {p.pallet_id: p.destination for p in instance.pallets}

        for pallet_id, truck_id in solution.pallet_to_truck.items():
            if truck_id == 'UNASSIGNED':
                continue
            assert pallet_destinations[pallet_id] == truck_destinations[truck_id]

    def test_edd_respects_capacity(self, instance):
        """Test that EDD respects truck capacity constraints."""
        solution = earliest_due_date(instance)

        # Count pallets per truck
        truck_loads = {}
        for pallet_id, truck_id in solution.pallet_to_truck.items():
            if truck_id == 'UNASSIGNED':
                continue
            truck_loads[truck_id] = truck_loads.get(truck_id, 0) + 1

        # Check against capacity
        truck_capacities = {t.truck_id: t.capacity for t in instance.outbound_trucks}

        for truck_id, load in truck_loads.items():
            assert load <= truck_capacities[truck_id]

    def test_edd_prioritizes_due_date(self, instance):
        """Test that EDD prioritizes pallets by due date."""
        solution = earliest_due_date(instance)

        # Get pallets sorted by due date
        sorted_pallets = sorted(instance.pallets, key=lambda p: p.due_date)

        # Check that early due date pallets are more likely to be assigned
        first_quarter = sorted_pallets[:len(sorted_pallets)//4]
        last_quarter = sorted_pallets[-len(sorted_pallets)//4:]

        first_assigned = sum(1 for p in first_quarter if solution.pallet_to_truck[p.pallet_id] != 'UNASSIGNED')
        last_assigned = sum(1 for p in last_quarter if solution.pallet_to_truck[p.pallet_id] != 'UNASSIGNED')

        # First quarter should have higher assignment rate
        first_rate = first_assigned / len(first_quarter)
        last_rate = last_assigned / len(last_quarter)

        # Allow some flexibility due to capacity constraints
        assert first_rate >= last_rate * 0.9

    def test_edd_performance(self, instance):
        """Test EDD performance metrics."""
        solution = earliest_due_date(instance)

        # Should have low number of late pallets
        assert solution.num_late_pallets < len(instance.pallets) * 0.05  # Less than 5% late

        # Should complete quickly (under 100ms for small instance)
        assert solution.solve_time < 0.1


class TestFirstFitHeuristic:
    """Test suite specifically for First-Fit heuristic."""

    @pytest.fixture
    def instance(self):
        """Load instance for testing."""
        loader = DataLoader()
        return loader.load_instance('LL_168h', 1)

    def test_first_fit_assigns_pallets(self, instance):
        """Test that First-Fit assigns pallets."""
        solution = first_fit(instance)

        assigned_count = len([p for p in solution.pallet_to_truck.values() if p != 'UNASSIGNED'])
        assert assigned_count > 0
        assert assigned_count >= len(instance.pallets) * 0.80  # At least 80%

    def test_first_fit_respects_constraints(self, instance):
        """Test that First-Fit respects all constraints."""
        solution = first_fit(instance)

        truck_destinations = {t.truck_id: t.destination for t in instance.outbound_trucks}
        truck_capacities = {t.truck_id: t.capacity for t in instance.outbound_trucks}
        pallet_destinations = {p.pallet_id: p.destination for p in instance.pallets}

        # Count pallets per truck
        truck_loads = {}
        for pallet_id, truck_id in solution.pallet_to_truck.items():
            if truck_id == 'UNASSIGNED':
                continue

            # Check destination match
            assert pallet_destinations[pallet_id] == truck_destinations[truck_id]

            # Track load
            truck_loads[truck_id] = truck_loads.get(truck_id, 0) + 1

        # Check capacity
        for truck_id, load in truck_loads.items():
            assert load <= truck_capacities[truck_id]

    def test_first_fit_speed(self, instance):
        """Test First-Fit execution speed."""
        solution = first_fit(instance)

        # Should be very fast (under 100ms)
        assert solution.solve_time < 0.1


class TestBestFitHeuristic:
    """Test suite specifically for Best-Fit heuristic."""

    @pytest.fixture
    def instance(self):
        """Load instance for testing."""
        loader = DataLoader()
        return loader.load_instance('LL_168h', 1)

    def test_best_fit_assigns_pallets(self, instance):
        """Test that Best-Fit assigns pallets."""
        solution = best_fit(instance)

        assigned_count = len([p for p in solution.pallet_to_truck.values() if p != 'UNASSIGNED'])
        assert assigned_count > 0

    def test_best_fit_respects_constraints(self, instance):
        """Test that Best-Fit respects all constraints."""
        solution = best_fit(instance)

        truck_destinations = {t.truck_id: t.destination for t in instance.outbound_trucks}
        truck_capacities = {t.truck_id: t.capacity for t in instance.outbound_trucks}
        pallet_destinations = {p.pallet_id: p.destination for p in instance.pallets}

        truck_loads = {}
        for pallet_id, truck_id in solution.pallet_to_truck.items():
            if truck_id == 'UNASSIGNED':
                continue

            assert pallet_destinations[pallet_id] == truck_destinations[truck_id]
            truck_loads[truck_id] = truck_loads.get(truck_id, 0) + 1

        for truck_id, load in truck_loads.items():
            assert load <= truck_capacities[truck_id]

    def test_best_fit_optimization(self, instance):
        """Test that Best-Fit attempts to minimize waste."""
        solution = best_fit(instance)

        # Calculate average fill rate
        truck_capacities = {t.truck_id: t.capacity for t in instance.outbound_trucks}

        truck_loads = {}
        for pallet_id, truck_id in solution.pallet_to_truck.items():
            if truck_id == 'UNASSIGNED':
                continue
            truck_loads[truck_id] = truck_loads.get(truck_id, 0) + 1

        # Calculate fill rates
        fill_rates = [truck_loads.get(t.truck_id, 0) / t.capacity for t in instance.outbound_trucks if t.truck_id in truck_loads]

        # Average fill rate should be reasonable
        if fill_rates:
            avg_fill_rate = sum(fill_rates) / len(fill_rates)
            assert avg_fill_rate > 0.5  # At least 50% on average


class TestDestinationBalanced:
    """Test suite specifically for Destination-Balanced heuristic."""

    @pytest.fixture
    def instance(self):
        """Load instance for testing."""
        loader = DataLoader()
        return loader.load_instance('LL_168h', 1)

    def test_destination_balanced_assigns_pallets(self, instance):
        """Test that Destination-Balanced assigns pallets."""
        solution = destination_balanced(instance)

        assigned_count = len([p for p in solution.pallet_to_truck.values() if p != 'UNASSIGNED'])
        assert assigned_count > 0

    def test_destination_balanced_respects_constraints(self, instance):
        """Test that Destination-Balanced respects all constraints."""
        solution = destination_balanced(instance)

        truck_destinations = {t.truck_id: t.destination for t in instance.outbound_trucks}
        truck_capacities = {t.truck_id: t.capacity for t in instance.outbound_trucks}
        pallet_destinations = {p.pallet_id: p.destination for p in instance.pallets}

        truck_loads = {}
        for pallet_id, truck_id in solution.pallet_to_truck.items():
            if truck_id == 'UNASSIGNED':
                continue

            assert pallet_destinations[pallet_id] == truck_destinations[truck_id]
            truck_loads[truck_id] = truck_loads.get(truck_id, 0) + 1

        for truck_id, load in truck_loads.items():
            assert load <= truck_capacities[truck_id]


class TestAlgorithmComparison:
    """Test suite comparing algorithm performance."""

    @pytest.fixture
    def instance(self):
        """Load instance for testing."""
        loader = DataLoader()
        return loader.load_instance('LL_168h', 1)

    def test_edd_vs_first_fit(self, instance):
        """Compare EDD and First-Fit performance."""
        edd_solution = earliest_due_date(instance)
        ff_solution = first_fit(instance)

        # EDD should have fewer late pallets
        assert edd_solution.num_late_pallets <= ff_solution.num_late_pallets

    def test_all_algorithms_complete(self, instance):
        """Test that all algorithms complete successfully."""
        solutions = {
            'EDD': earliest_due_date(instance),
            'First-Fit': first_fit(instance),
            'Best-Fit': best_fit(instance),
            'Dest-Balanced': destination_balanced(instance)
        }

        for name, solution in solutions.items():
            assert solution is not None, f"{name} returned None"
            assert solution.solve_time > 0, f"{name} has invalid solve time"

    def test_solution_validity(self, instance):
        """Test that all solutions are valid."""
        algorithms = [earliest_due_date, first_fit, best_fit, destination_balanced]

        for algo in algorithms:
            solution = algo(instance)

            # Check that assignments are valid
            assert all(isinstance(truck_id, (int, str)) for truck_id in solution.pallet_to_truck.values())

            # Check that truck loads are valid
            truck_loads = {}
            for pallet_id, truck_id in solution.pallet_to_truck.items():
                if truck_id == 'UNASSIGNED':
                    continue
                truck_loads[truck_id] = truck_loads.get(truck_id, 0) + 1

            # All loads should be positive
            assert all(load > 0 for load in truck_loads.values())


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.fixture
    def loader(self):
        """Create data loader."""
        return DataLoader()

    def test_different_instance_sizes(self, loader):
        """Test algorithms on different instance sizes."""
        instances = [
            loader.load_instance('LL_168h', 1),  # Small
            loader.load_instance('MM_168h', 1),  # Medium
            loader.load_instance('HH_168h', 1),  # Large
        ]

        for instance in instances:
            # All algorithms should work
            edd_sol = earliest_due_date(instance)
            ff_sol = first_fit(instance)

            assert edd_sol is not None
            assert ff_sol is not None
            assert edd_sol.num_late_pallets <= ff_sol.num_late_pallets

    def test_multiple_instances_same_scenario(self, loader):
        """Test consistency across multiple instances."""
        instances = [loader.load_instance('LL_168h', i) for i in range(1, 6)]

        for instance in instances:
            solution = earliest_due_date(instance)

            # EDD should consistently achieve high service level
            service_level = 1 - (solution.num_late_pallets / len(instance.pallets))
            assert service_level >= 0.95  # At least 95%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
