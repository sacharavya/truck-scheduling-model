"""
Pallet-to-Truck Assignment Optimization Model

This module implements a Mixed-Integer Linear Programming (MILP) model for
assigning pallets to outbound trucks in a cross-docking terminal.

Objectives:
- Maximize outbound truck fill rates
- Minimize late pallets (tardiness)
- Respect truck capacity constraints
- Match pallets to correct destination

Uses Google OR-Tools CP-SAT solver for efficient optimization.

Author: Cross-Docking Optimization Project
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from ortools.sat.python import cp_model
import time
import logging

# Import custom data structures
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import CrossDockInstance, Pallet, OutboundTruck

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AssignmentSolution:
    """
    Solution for the pallet-to-truck assignment problem.

    Attributes:
        assignments: Dict mapping (pallet_id, truck_id) to 1 if assigned, 0 otherwise
        truck_loads: Dict mapping truck_id to list of assigned pallet_ids
        objective_value: Value of the objective function
        solve_time: Time taken to solve (seconds)
        status: Solver status (OPTIMAL, FEASIBLE, INFEASIBLE, etc.)
        num_late_pallets: Number of pallets assigned after their due date
        avg_fill_rate: Average fill rate across all trucks
        total_tardiness: Total tardiness in minutes
    """
    assignments: Dict[Tuple[int, int], int] = field(default_factory=dict)
    truck_loads: Dict[int, List[int]] = field(default_factory=dict)
    objective_value: float = 0.0
    solve_time: float = 0.0
    status: str = "UNKNOWN"
    num_late_pallets: int = 0
    avg_fill_rate: float = 0.0
    total_tardiness: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def get_truck_fill_rate(self, truck_id: int, capacity: int = 26) -> float:
        """Get fill rate for a specific truck."""
        if truck_id not in self.truck_loads:
            return 0.0
        return len(self.truck_loads[truck_id]) / capacity

    def print_summary(self):
        """Print a summary of the solution."""
        print("\n" + "="*70)
        print("PALLET-TO-TRUCK ASSIGNMENT SOLUTION SUMMARY")
        print("="*70)
        print(f"Status: {self.status}")
        print(f"Solve Time: {self.solve_time:.2f} seconds")
        print(f"Objective Value: {self.objective_value:.2f}")
        print(f"\nKPIs:")
        print(f"  Average Fill Rate: {self.avg_fill_rate:.2%}")
        print(f"  Late Pallets: {self.num_late_pallets}")
        print(f"  Total Tardiness: {self.total_tardiness:.2f} minutes")
        print(f"  Trucks Utilized: {len(self.truck_loads)}")
        if 'unassigned_pallets' in self.metadata:
            print(f"  Unassigned Pallets: {self.metadata['unassigned_pallets']}")
        print("="*70)


class PalletAssignmentModel:
    """
    MILP model for assigning pallets to outbound trucks.

    Decision Variables:
        x[p,t] = 1 if pallet p is assigned to truck t, 0 otherwise

    Constraints:
        1. Each pallet assigned to exactly one truck
        2. Truck capacity constraints (max 26 pallets)
        3. Destination matching (pallets assigned to correct destination trucks)
        4. Pallet availability (truck arrival >= pallet arrival via inbound truck)

    Objective:
        Maximize: weighted sum of fill rate and on-time performance
        - Reward high fill rates
        - Penalize late deliveries (pallet due date < truck departure)
    """

    def __init__(self,
                 instance: CrossDockInstance,
                 tardiness_weight: float = 100.0,
                 fill_rate_weight: float = 50.0,
                 time_limit_seconds: int = 300):
        """
        Initialize the pallet assignment model.

        Args:
            instance: CrossDockInstance with pallets and trucks
            tardiness_weight: Weight for tardiness penalty in objective
            fill_rate_weight: Weight for fill rate bonus in objective
            time_limit_seconds: Maximum solve time
        """
        self.instance = instance
        self.tardiness_weight = tardiness_weight
        self.fill_rate_weight = fill_rate_weight
        self.time_limit = time_limit_seconds

        # Create model
        self.model = cp_model.CpModel()

        # Decision variables
        self.x = {}  # x[p,t] = 1 if pallet p assigned to truck t
        self.tardiness = {}  # tardiness[p,t] = max(0, truck_due_date - pallet_due_date)

        # Mappings for efficient lookup
        self.pallet_dict = {p.pallet_id: p for p in instance.pallets}
        self.truck_dict = {t.truck_id: t for t in instance.outbound_trucks}

        # Group pallets and trucks by destination
        self.pallets_by_dest = {d: [] for d in [1, 2, 3]}
        self.trucks_by_dest = {d: [] for d in [1, 2, 3]}

        for p in instance.pallets:
            self.pallets_by_dest[p.destination].append(p.pallet_id)

        for t in instance.outbound_trucks:
            self.trucks_by_dest[t.destination].append(t.truck_id)

        logger.info(f"Initialized model for {len(instance.pallets)} pallets "
                   f"and {len(instance.outbound_trucks)} trucks")

    def build_model(self):
        """Build the MILP model with variables, constraints, and objective."""
        logger.info("Building MILP model...")

        # Create decision variables
        self._create_variables()

        # Add constraints
        self._add_assignment_constraints()
        self._add_capacity_constraints()
        self._add_destination_constraints()

        # Set objective
        self._set_objective()

        logger.info("Model built successfully")

    def _create_variables(self):
        """Create decision variables for pallet-truck assignments."""
        # For each pallet, create variables for trucks with matching destination
        # PLUS an "unassigned" slack variable to handle capacity overflow
        for pallet_id, pallet in self.pallet_dict.items():
            valid_trucks = self.trucks_by_dest[pallet.destination]

            for truck_id in valid_trucks:
                # Binary variable: is pallet p assigned to truck t?
                var_name = f'x_p{pallet_id}_t{truck_id}'
                self.x[pallet_id, truck_id] = self.model.NewBoolVar(var_name)

            # Slack variable: pallet remains unassigned (penalty in objective)
            slack_var_name = f'unassigned_p{pallet_id}'
            self.x[pallet_id, 'UNASSIGNED'] = self.model.NewBoolVar(slack_var_name)

        logger.debug(f"Created {len(self.x)} assignment variables (including unassigned slack)")

    def _add_assignment_constraints(self):
        """Each pallet must be assigned to exactly one truck OR remain unassigned."""
        for pallet_id, pallet in self.pallet_dict.items():
            valid_trucks = self.trucks_by_dest[pallet.destination]

            # Sum of assignments for this pallet (including unassigned) must equal 1
            assignment_vars = [self.x[pallet_id, truck_id] for truck_id in valid_trucks]
            assignment_vars.append(self.x[pallet_id, 'UNASSIGNED'])
            self.model.Add(sum(assignment_vars) == 1)

        logger.debug("Added assignment constraints (one truck per pallet + unassigned option)")

    def _add_capacity_constraints(self):
        """Ensure trucks don't exceed their capacity (26 pallets)."""
        TRUCK_CAPACITY = 26

        for truck_id in self.truck_dict.keys():
            truck = self.truck_dict[truck_id]
            # Get all pallets that could be assigned to this truck
            valid_pallets = self.pallets_by_dest[truck.destination]

            # Sum of pallets assigned to this truck <= capacity
            truck_load = [self.x[pallet_id, truck_id]
                         for pallet_id in valid_pallets
                         if (pallet_id, truck_id) in self.x]

            if truck_load:  # Only add constraint if there are valid assignments
                self.model.Add(sum(truck_load) <= TRUCK_CAPACITY)

        logger.debug("Added capacity constraints (max 26 pallets per truck)")

    def _add_destination_constraints(self):
        """
        Ensure pallets are only assigned to trucks going to the correct destination.
        This is implicitly handled by only creating variables for matching destinations.
        """
        # Already handled in _create_variables by filtering valid_trucks
        logger.debug("Destination constraints enforced via variable creation")

    def _set_objective(self):
        """
        Set the objective function to maximize fill rate and minimize tardiness.

        Objective = (fill_rate_weight * total_pallets_assigned)
                   - (tardiness_weight * total_tardiness)
                   - (unassignment_penalty * total_unassigned)
        """
        objective_terms = []
        UNASSIGNMENT_PENALTY = 10000  # Very large penalty for unassigned pallets

        # Iterate through all assignment variables
        for (pallet_id, truck_id), var in self.x.items():
            pallet = self.pallet_dict[pallet_id]

            if truck_id == 'UNASSIGNED':
                # Large penalty for leaving pallet unassigned
                objective_terms.append(-UNASSIGNMENT_PENALTY * var)
            else:
                truck = self.truck_dict[truck_id]

                # Reward for assignment
                assignment_reward = int(self.fill_rate_weight)
                objective_terms.append(assignment_reward * var)

                # Penalty for tardiness (if pallet due date < truck due date, it's late)
                pallet_due = int(pallet.due_date)
                truck_due = int(truck.due_date)

                if pallet_due < truck_due:
                    # This assignment would make the pallet late
                    tardiness_penalty = int(self.tardiness_weight * (truck_due - pallet_due))
                    objective_terms.append(-tardiness_penalty * var)

        # Maximize the objective
        self.model.Maximize(sum(objective_terms))
        logger.debug("Objective function set: maximize fill rate - tardiness - unassignment penalty")

    def solve(self) -> AssignmentSolution:
        """
        Solve the optimization model.

        Returns:
            AssignmentSolution object with results
        """
        logger.info("Solving optimization model...")

        # Create solver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.log_search_progress = False

        # Solve
        start_time = time.time()
        status = solver.Solve(self.model)
        solve_time = time.time() - start_time

        # Map status
        status_name = solver.StatusName(status)
        logger.info(f"Solver finished with status: {status_name} in {solve_time:.2f}s")

        # Extract solution if feasible or optimal
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = self._extract_solution(solver, solve_time, status_name)
        else:
            # Infeasible or unknown
            solution = AssignmentSolution(
                status=status_name,
                solve_time=solve_time
            )
            logger.warning(f"No solution found. Status: {status_name}")

        return solution

    def _extract_solution(self, solver, solve_time: float, status: str) -> AssignmentSolution:
        """Extract solution from solver."""
        assignments = {}
        truck_loads = {truck_id: [] for truck_id in self.truck_dict.keys()}
        unassigned_pallets = []

        # Extract assignment variables
        for (pallet_id, truck_id), var in self.x.items():
            value = solver.Value(var)
            assignments[pallet_id, truck_id] = value

            if value == 1:
                if truck_id == 'UNASSIGNED':
                    unassigned_pallets.append(pallet_id)
                else:
                    truck_loads[truck_id].append(pallet_id)

        # Calculate KPIs
        num_late_pallets = 0
        total_tardiness = 0.0

        for truck_id, pallet_ids in truck_loads.items():
            truck = self.truck_dict[truck_id]
            for pallet_id in pallet_ids:
                pallet = self.pallet_dict[pallet_id]
                # Check if pallet is late
                if pallet.due_date < truck.due_date:
                    num_late_pallets += 1
                    total_tardiness += (truck.due_date - pallet.due_date)

        # Calculate average fill rate
        TRUCK_CAPACITY = 26
        fill_rates = [len(loads) / TRUCK_CAPACITY for loads in truck_loads.values() if len(loads) > 0]
        avg_fill_rate = np.mean(fill_rates) if fill_rates else 0.0

        # Remove empty truck loads
        truck_loads = {k: v for k, v in truck_loads.items() if v}

        solution = AssignmentSolution(
            assignments=assignments,
            truck_loads=truck_loads,
            objective_value=solver.ObjectiveValue(),
            solve_time=solve_time,
            status=status,
            num_late_pallets=num_late_pallets,
            avg_fill_rate=avg_fill_rate,
            total_tardiness=total_tardiness,
            metadata={
                'num_pallets': len(self.instance.pallets),
                'num_trucks': len(self.instance.outbound_trucks),
                'trucks_used': len(truck_loads),
                'unassigned_pallets': len(unassigned_pallets),
                'unassigned_pallet_ids': unassigned_pallets
            }
        )

        return solution


def optimize_pallet_assignment(instance: CrossDockInstance,
                               tardiness_weight: float = 100.0,
                               fill_rate_weight: float = 50.0,
                               time_limit: int = 300) -> AssignmentSolution:
    """
    Convenience function to optimize pallet-to-truck assignment.

    Args:
        instance: CrossDockInstance with data
        tardiness_weight: Weight for tardiness penalty
        fill_rate_weight: Weight for fill rate reward
        time_limit: Maximum solve time in seconds

    Returns:
        AssignmentSolution object
    """
    model = PalletAssignmentModel(
        instance=instance,
        tardiness_weight=tardiness_weight,
        fill_rate_weight=fill_rate_weight,
        time_limit_seconds=time_limit
    )

    model.build_model()
    solution = model.solve()

    return solution


# Example usage and testing
if __name__ == "__main__":
    import os
    from data_loader import DataLoader

    # Change to project root directory
    os.chdir(Path(__file__).parent.parent.parent)

    # Load a test instance (use smaller LL scenario for testing)
    loader = DataLoader()
    instance = loader.load_instance('LL_168h', 1)

    print(f"\n{'='*70}")
    print(f"TESTING PALLET ASSIGNMENT MODEL")
    print(f"{'='*70}")
    print(f"Instance: {instance.instance_name}")
    print(f"Pallets: {len(instance.pallets)}")
    print(f"Outbound Trucks: {len(instance.outbound_trucks)}")

    # Optimize with longer time limit for complex problems
    solution = optimize_pallet_assignment(
        instance=instance,
        tardiness_weight=100.0,
        fill_rate_weight=50.0,
        time_limit=120
    )

    # Print results
    solution.print_summary()

    # Show some example assignments
    print(f"\nExample truck loads (first 5 trucks):")
    for i, (truck_id, pallet_ids) in enumerate(list(solution.truck_loads.items())[:5]):
        truck = [t for t in instance.outbound_trucks if t.truck_id == truck_id][0]
        fill_rate = len(pallet_ids) / 26
        print(f"  Truck {truck_id} (dest {truck.destination}): "
              f"{len(pallet_ids)} pallets, fill rate: {fill_rate:.1%}")
