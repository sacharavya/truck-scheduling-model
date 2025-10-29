"""
Heuristic Algorithms for Cross-Docking Pallet Assignment

This module implements fast heuristic algorithms for assigning pallets to trucks.
These provide quick baseline solutions and can be used for real-time decision making.

Heuristics included:
- First-Fit (FF): Assign pallets to first available truck
- Best-Fit (BF): Assign to truck with best remaining capacity
- Earliest Due Date (EDD): Prioritize pallets by due date
- Destination-Balanced (DB): Balance loads across destinations

Author: Cross-Docking Optimization Project
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import logging
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import CrossDockInstance, Pallet, OutboundTruck
from models.pallet_assignment import AssignmentSolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeuristicSolver:
    """Base class for heuristic algorithms."""

    def __init__(self, instance: CrossDockInstance):
        """Initialize with a cross-dock instance."""
        self.instance = instance
        self.TRUCK_CAPACITY = 26

    def create_truck_assignment_state(self) -> Dict:
        """Create initial state for truck assignments."""
        state = {
            'truck_loads': {t.truck_id: [] for t in self.instance.outbound_trucks},
            'truck_remaining_capacity': {t.truck_id: self.TRUCK_CAPACITY
                                        for t in self.instance.outbound_trucks},
            'unassigned_pallets': [],
            'assignments': {}
        }
        return state

    def compute_kpis(self, state: Dict) -> Dict:
        """Compute KPIs from assignment state."""
        truck_loads = state['truck_loads']

        # Remove empty trucks
        used_trucks = {k: v for k, v in truck_loads.items() if v}

        # Calculate fill rates
        fill_rates = []
        for truck_id, pallet_ids in used_trucks.items():
            fill_rates.append(len(pallet_ids) / self.TRUCK_CAPACITY)

        avg_fill_rate = sum(fill_rates) / len(fill_rates) if fill_rates else 0.0

        # Calculate tardiness
        num_late = 0
        total_tardiness = 0.0
        truck_dict = {t.truck_id: t for t in self.instance.outbound_trucks}
        pallet_dict = {p.pallet_id: p for p in self.instance.pallets}

        for truck_id, pallet_ids in used_trucks.items():
            truck = truck_dict[truck_id]
            for pid in pallet_ids:
                pallet = pallet_dict[pid]
                if pallet.due_date < truck.due_date:
                    num_late += 1
                    total_tardiness += (truck.due_date - pallet.due_date)

        return {
            'avg_fill_rate': avg_fill_rate,
            'num_late_pallets': num_late,
            'total_tardiness': total_tardiness,
            'trucks_used': len(used_trucks),
            'unassigned_pallets': len(state['unassigned_pallets'])
        }

    def build_solution(self, state: Dict, solve_time: float) -> AssignmentSolution:
        """Build AssignmentSolution object from state."""
        kpis = self.compute_kpis(state)

        # Remove empty trucks
        truck_loads = {k: v for k, v in state['truck_loads'].items() if v}

        return AssignmentSolution(
            assignments=state['assignments'],
            truck_loads=truck_loads,
            objective_value=0.0,  # Not applicable for heuristics
            solve_time=solve_time,
            status="HEURISTIC",
            num_late_pallets=kpis['num_late_pallets'],
            avg_fill_rate=kpis['avg_fill_rate'],
            total_tardiness=kpis['total_tardiness'],
            metadata={
                'num_pallets': len(self.instance.pallets),
                'num_trucks': len(self.instance.outbound_trucks),
                'trucks_used': kpis['trucks_used'],
                'unassigned_pallets': kpis['unassigned_pallets'],
                'unassigned_pallet_ids': state['unassigned_pallets']
            }
        )


class FirstFitHeuristic(HeuristicSolver):
    """
    First-Fit (FF) Heuristic

    Assign each pallet to the first truck with available capacity
    going to the correct destination.
    """

    def solve(self) -> AssignmentSolution:
        """Solve using First-Fit heuristic."""
        start_time = time.time()
        logger.info("Running First-Fit heuristic...")

        state = self.create_truck_assignment_state()

        # Group trucks by destination
        trucks_by_dest = defaultdict(list)
        for truck in self.instance.outbound_trucks:
            trucks_by_dest[truck.destination].append(truck)

        # Process each pallet
        for pallet in self.instance.pallets:
            assigned = False
            destination_trucks = trucks_by_dest[pallet.destination]

            # Try to assign to first truck with capacity
            for truck in destination_trucks:
                if state['truck_remaining_capacity'][truck.truck_id] > 0:
                    # Assign pallet to this truck
                    state['truck_loads'][truck.truck_id].append(pallet.pallet_id)
                    state['truck_remaining_capacity'][truck.truck_id] -= 1
                    state['assignments'][(pallet.pallet_id, truck.truck_id)] = 1
                    assigned = True
                    break

            if not assigned:
                state['unassigned_pallets'].append(pallet.pallet_id)

        solve_time = time.time() - start_time
        solution = self.build_solution(state, solve_time)

        logger.info(f"First-Fit completed in {solve_time:.2f}s")
        return solution


class BestFitHeuristic(HeuristicSolver):
    """
    Best-Fit (BF) Heuristic

    Assign each pallet to the truck with the least remaining capacity
    (to maximize utilization) that still has space.
    """

    def solve(self) -> AssignmentSolution:
        """Solve using Best-Fit heuristic."""
        start_time = time.time()
        logger.info("Running Best-Fit heuristic...")

        state = self.create_truck_assignment_state()

        # Group trucks by destination
        trucks_by_dest = defaultdict(list)
        for truck in self.instance.outbound_trucks:
            trucks_by_dest[truck.destination].append(truck)

        # Process each pallet
        for pallet in self.instance.pallets:
            destination_trucks = trucks_by_dest[pallet.destination]

            # Find truck with minimum remaining capacity > 0 (best fit)
            best_truck = None
            min_remaining = self.TRUCK_CAPACITY + 1

            for truck in destination_trucks:
                remaining = state['truck_remaining_capacity'][truck.truck_id]
                if 0 < remaining < min_remaining:
                    best_truck = truck
                    min_remaining = remaining

            if best_truck:
                # Assign to best-fit truck
                state['truck_loads'][best_truck.truck_id].append(pallet.pallet_id)
                state['truck_remaining_capacity'][best_truck.truck_id] -= 1
                state['assignments'][(pallet.pallet_id, best_truck.truck_id)] = 1
            else:
                state['unassigned_pallets'].append(pallet.pallet_id)

        solve_time = time.time() - start_time
        solution = self.build_solution(state, solve_time)

        logger.info(f"Best-Fit completed in {solve_time:.2f}s")
        return solution


class EarliestDueDateHeuristic(HeuristicSolver):
    """
    Earliest Due Date (EDD) Heuristic

    Sort pallets by due date (earliest first) and assign to trucks
    going to correct destination with earliest due dates.
    """

    def solve(self) -> AssignmentSolution:
        """Solve using EDD heuristic."""
        start_time = time.time()
        logger.info("Running Earliest Due Date heuristic...")

        state = self.create_truck_assignment_state()

        # Group trucks by destination and sort by due date
        trucks_by_dest = defaultdict(list)
        for truck in self.instance.outbound_trucks:
            trucks_by_dest[truck.destination].append(truck)

        for dest in trucks_by_dest:
            trucks_by_dest[dest].sort(key=lambda t: t.due_date)

        # Sort pallets by due date (earliest first)
        sorted_pallets = sorted(self.instance.pallets, key=lambda p: p.due_date)

        # Process each pallet
        for pallet in sorted_pallets:
            destination_trucks = trucks_by_dest[pallet.destination]

            # Assign to first available truck with capacity
            assigned = False
            for truck in destination_trucks:
                if state['truck_remaining_capacity'][truck.truck_id] > 0:
                    state['truck_loads'][truck.truck_id].append(pallet.pallet_id)
                    state['truck_remaining_capacity'][truck.truck_id] -= 1
                    state['assignments'][(pallet.pallet_id, truck.truck_id)] = 1
                    assigned = True
                    break

            if not assigned:
                state['unassigned_pallets'].append(pallet.pallet_id)

        solve_time = time.time() - start_time
        solution = self.build_solution(state, solve_time)

        logger.info(f"EDD completed in {solve_time:.2f}s")
        return solution


class DestinationBalancedHeuristic(HeuristicSolver):
    """
    Destination-Balanced (DB) Heuristic

    Balance loads evenly across trucks going to each destination.
    Uses round-robin assignment within each destination.
    """

    def solve(self) -> AssignmentSolution:
        """Solve using Destination-Balanced heuristic."""
        start_time = time.time()
        logger.info("Running Destination-Balanced heuristic...")

        state = self.create_truck_assignment_state()

        # Group pallets and trucks by destination
        pallets_by_dest = defaultdict(list)
        for pallet in self.instance.pallets:
            pallets_by_dest[pallet.destination].append(pallet)

        trucks_by_dest = defaultdict(list)
        for truck in self.instance.outbound_trucks:
            trucks_by_dest[truck.destination].append(truck)

        # Process each destination separately
        for destination in [1, 2, 3]:
            pallets = pallets_by_dest[destination]
            trucks = trucks_by_dest[destination]

            truck_index = 0  # Round-robin index

            for pallet in pallets:
                assigned = False
                attempts = 0

                # Try round-robin assignment
                while attempts < len(trucks):
                    truck = trucks[truck_index]

                    if state['truck_remaining_capacity'][truck.truck_id] > 0:
                        # Assign pallet
                        state['truck_loads'][truck.truck_id].append(pallet.pallet_id)
                        state['truck_remaining_capacity'][truck.truck_id] -= 1
                        state['assignments'][(pallet.pallet_id, truck.truck_id)] = 1
                        assigned = True

                        # Move to next truck for next pallet
                        truck_index = (truck_index + 1) % len(trucks)
                        break

                    # This truck is full, try next
                    truck_index = (truck_index + 1) % len(trucks)
                    attempts += 1

                if not assigned:
                    state['unassigned_pallets'].append(pallet.pallet_id)

        solve_time = time.time() - start_time
        solution = self.build_solution(state, solve_time)

        logger.info(f"Destination-Balanced completed in {solve_time:.2f}s")
        return solution


# Convenience functions
def first_fit(instance: CrossDockInstance) -> AssignmentSolution:
    """Run First-Fit heuristic."""
    solver = FirstFitHeuristic(instance)
    return solver.solve()


def best_fit(instance: CrossDockInstance) -> AssignmentSolution:
    """Run Best-Fit heuristic."""
    solver = BestFitHeuristic(instance)
    return solver.solve()


def earliest_due_date(instance: CrossDockInstance) -> AssignmentSolution:
    """Run Earliest Due Date heuristic."""
    solver = EarliestDueDateHeuristic(instance)
    return solver.solve()


def destination_balanced(instance: CrossDockInstance) -> AssignmentSolution:
    """Run Destination-Balanced heuristic."""
    solver = DestinationBalancedHeuristic(instance)
    return solver.solve()


# Testing
if __name__ == "__main__":
    import os
    from data_loader import DataLoader

    # Change to project root
    os.chdir(Path(__file__).parent.parent.parent)

    # Load test instance
    loader = DataLoader()
    instance = loader.load_instance('LL_168h', 1)

    print(f"\n{'='*70}")
    print(f"TESTING HEURISTIC ALGORITHMS")
    print(f"{'='*70}")
    print(f"Instance: {instance.instance_name}")
    print(f"Pallets: {len(instance.pallets)}")
    print(f"Outbound Trucks: {len(instance.outbound_trucks)}")
    print(f"{'='*70}\n")

    # Test all heuristics
    heuristics = [
        ("First-Fit", first_fit),
        ("Best-Fit", best_fit),
        ("Earliest Due Date", earliest_due_date),
        ("Destination-Balanced", destination_balanced)
    ]

    results = []
    for name, heuristic_func in heuristics:
        print(f"\n--- {name} ---")
        solution = heuristic_func(instance)
        solution.print_summary()
        results.append((name, solution))

    # Comparison table
    print(f"\n\n{'='*90}")
    print(f"HEURISTIC COMPARISON")
    print(f"{'='*90}")
    print(f"{'Algorithm':<25} {'Time(s)':<12} {'Fill Rate':<12} {'Late':<8} {'Unassigned':<12}")
    print(f"{'-'*90}")
    for name, sol in results:
        print(f"{name:<25} {sol.solve_time:<12.3f} {sol.avg_fill_rate:<12.2%} "
              f"{sol.num_late_pallets:<8} {sol.metadata['unassigned_pallets']:<12}")
    print(f"{'='*90}")
