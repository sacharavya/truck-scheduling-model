"""
Door Assignment Model using OR-Tools MILP.

This module optimizes the assignment of trucks to dock doors to minimize
conflicts, waiting times, and makespan while respecting door capacity constraints.

Author: Cross-Docking Optimization System
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import time

from ortools.sat.python import cp_model

from src.data_loader import CrossDockInstance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DoorAssignment:
    """Result of door assignment optimization."""

    # Door assignments
    inbound_truck_to_door: Dict[int, int]  # truck_id -> door_id
    outbound_truck_to_door: Dict[int, int]  # truck_id -> door_id

    # Timing information
    truck_start_times: Dict[int, float]  # truck_id -> start time at assigned door
    truck_end_times: Dict[int, float]  # truck_id -> end time at assigned door

    # Door utilization
    door_utilization: Dict[int, float]  # door_id -> utilization (0-1)
    door_idle_time: Dict[int, float]  # door_id -> total idle time

    # Metrics
    total_waiting_time: float  # Total waiting time across all trucks
    max_waiting_time: float  # Maximum waiting time for any truck
    makespan: float  # Total time to process all trucks
    num_conflicts: int  # Number of door conflicts (should be 0)

    # Metadata
    solve_time: float
    solver_status: str
    objective_value: float


class DoorAssignmentModel:
    """
    MILP model for door assignment optimization.

    Assigns trucks to doors to minimize waiting times and conflicts
    while balancing door utilization.
    """

    def __init__(self, instance: CrossDockInstance, config: Optional[Dict] = None):
        """
        Initialize door assignment model.

        Args:
            instance: Cross-dock instance
            config: Optional configuration parameters
        """
        self.instance = instance
        self.config = config or {}

        # Door configuration
        self.num_inbound_doors = self.config.get('num_inbound_doors', 1)
        self.num_outbound_doors = self.config.get('num_outbound_doors', 1)

        # Processing time parameters (minutes)
        self.unload_time_per_pallet = self.config.get('unload_time_per_pallet', 1.0)
        self.load_time_per_pallet = self.config.get('load_time_per_pallet', 1.0)
        self.truck_setup_time = self.config.get('truck_setup_time', 5.0)

        # Time horizon
        self.time_horizon = self.config.get('time_horizon', 10080)  # 7 days

        # Weights
        self.waiting_weight = self.config.get('waiting_weight', 1.0)
        self.balance_weight = self.config.get('balance_weight', 0.1)

        # OR-Tools model
        self.model = cp_model.CpModel()

        # Decision variables
        self.inbound_door_assign = {}  # (truck_id, door_id) -> bool
        self.outbound_door_assign = {}  # (truck_id, door_id) -> bool

        self.truck_start = {}  # truck_id -> start time
        self.truck_end = {}  # truck_id -> end time
        self.truck_waiting = {}  # truck_id -> waiting time

        # Sequencing variables for each door
        self.door_sequence = {}  # (door_id, truck_i, truck_j) -> bool (i before j)

        logger.info(f"DoorAssignmentModel initialized: {self.num_inbound_doors} inbound doors, "
                   f"{self.num_outbound_doors} outbound doors")

    def build_model(self):
        """Build the complete MILP model."""
        logger.info("Building door assignment model...")

        self._create_variables()
        self._add_assignment_constraints()
        self._add_timing_constraints()
        self._add_non_overlap_constraints()
        self._set_objective()

        logger.info("Door assignment model building complete")

    def _create_variables(self):
        """Create decision variables."""

        # Inbound truck-to-door assignment variables
        for truck in self.instance.inbound_trucks:
            truck_id = truck.truck_id

            # Assignment variables (one door per truck)
            for door_id in range(self.num_inbound_doors):
                self.inbound_door_assign[(truck_id, door_id)] = self.model.NewBoolVar(
                    f'inbound_assign_t{truck_id}_d{door_id}'
                )

            # Timing variables
            self.truck_start[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time),
                self.time_horizon,
                f'truck_start_{truck_id}'
            )

            # Calculate processing time
            num_pallets = len([p for p in self.instance.pallets if p.inbound_truck_id == truck_id])
            processing_time = int(self.truck_setup_time + num_pallets * self.unload_time_per_pallet)

            self.truck_end[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time) + processing_time,
                self.time_horizon,
                f'truck_end_{truck_id}'
            )

            # Waiting time
            self.truck_waiting[truck_id] = self.model.NewIntVar(
                0,
                self.time_horizon,
                f'truck_wait_{truck_id}'
            )

        # Outbound truck-to-door assignment variables
        for truck in self.instance.outbound_trucks:
            truck_id = truck.truck_id

            # Assignment variables
            for door_id in range(self.num_outbound_doors):
                self.outbound_door_assign[(truck_id, door_id)] = self.model.NewBoolVar(
                    f'outbound_assign_t{truck_id}_d{door_id}'
                )

            # Timing variables
            self.truck_start[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time),
                self.time_horizon,
                f'truck_start_{truck_id}'
            )

            # Processing time
            estimated_pallets = min(truck.capacity, 20)
            processing_time = int(self.truck_setup_time + estimated_pallets * self.load_time_per_pallet)

            self.truck_end[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time) + processing_time,
                self.time_horizon,
                f'truck_end_{truck_id}'
            )

            # Waiting time
            self.truck_waiting[truck_id] = self.model.NewIntVar(
                0,
                self.time_horizon,
                f'truck_wait_{truck_id}'
            )

        # Sequencing variables for each door
        # For each door, create sequencing variables between all truck pairs
        for door_id in range(self.num_inbound_doors):
            inbound_trucks = [t.truck_id for t in self.instance.inbound_trucks]
            for i, truck_i in enumerate(inbound_trucks):
                for truck_j in inbound_trucks[i+1:]:
                    self.door_sequence[(door_id, truck_i, truck_j)] = self.model.NewBoolVar(
                        f'door{door_id}_seq_t{truck_i}_t{truck_j}'
                    )

        for door_id in range(self.num_outbound_doors):
            outbound_trucks = [t.truck_id for t in self.instance.outbound_trucks]
            for i, truck_i in enumerate(outbound_trucks):
                for truck_j in outbound_trucks[i+1:]:
                    self.door_sequence[(door_id, truck_i, truck_j)] = self.model.NewBoolVar(
                        f'door{door_id}_seq_t{truck_i}_t{truck_j}'
                    )

    def _add_assignment_constraints(self):
        """Add constraints ensuring each truck is assigned to exactly one door."""

        # Each inbound truck assigned to exactly one inbound door
        for truck in self.instance.inbound_trucks:
            truck_id = truck.truck_id
            door_vars = [
                self.inbound_door_assign[(truck_id, door_id)]
                for door_id in range(self.num_inbound_doors)
            ]
            self.model.Add(sum(door_vars) == 1)

        # Each outbound truck assigned to exactly one outbound door
        for truck in self.instance.outbound_trucks:
            truck_id = truck.truck_id
            door_vars = [
                self.outbound_door_assign[(truck_id, door_id)]
                for door_id in range(self.num_outbound_doors)
            ]
            self.model.Add(sum(door_vars) == 1)

    def _add_timing_constraints(self):
        """Add timing constraints."""

        for truck in list(self.instance.inbound_trucks) + list(self.instance.outbound_trucks):
            truck_id = truck.truck_id

            # Start time >= arrival time
            self.model.Add(self.truck_start[truck_id] >= int(truck.arrival_time))

            # Waiting time = start - arrival
            self.model.Add(
                self.truck_waiting[truck_id] == self.truck_start[truck_id] - int(truck.arrival_time)
            )

            # Calculate processing time
            if truck in self.instance.inbound_trucks:
                num_pallets = len([p for p in self.instance.pallets if p.inbound_truck_id == truck_id])
                processing_time = int(self.truck_setup_time + num_pallets * self.unload_time_per_pallet)
            else:
                estimated_pallets = min(truck.capacity, 20)
                processing_time = int(self.truck_setup_time + estimated_pallets * self.load_time_per_pallet)

            # End time = start + processing
            self.model.Add(
                self.truck_end[truck_id] == self.truck_start[truck_id] + processing_time
            )

    def _add_non_overlap_constraints(self):
        """
        Add non-overlap constraints for trucks assigned to the same door.

        Trucks assigned to the same door cannot overlap in time.
        """

        # Inbound doors
        for door_id in range(self.num_inbound_doors):
            inbound_trucks = [t.truck_id for t in self.instance.inbound_trucks]

            for i, truck_i in enumerate(inbound_trucks):
                for truck_j in inbound_trucks[i+1:]:

                    # Get assignment variables
                    assign_i = self.inbound_door_assign[(truck_i, door_id)]
                    assign_j = self.inbound_door_assign[(truck_j, door_id)]

                    # Get sequencing variable
                    seq_var = self.door_sequence[(door_id, truck_i, truck_j)]

                    # Only enforce non-overlap if both assigned to this door
                    # Create auxiliary variable: both_assigned = assign_i AND assign_j
                    both_assigned = self.model.NewBoolVar(f'both_d{door_id}_t{truck_i}_t{truck_j}')

                    # both_assigned = 1 IFF assign_i = 1 AND assign_j = 1
                    self.model.AddBoolAnd([assign_i, assign_j]).OnlyEnforceIf(both_assigned)
                    self.model.AddBoolOr([assign_i.Not(), assign_j.Not()]).OnlyEnforceIf(both_assigned.Not())

                    # If both assigned to this door, enforce non-overlap using sequencing
                    # If seq_var = 1 (i before j): end_i <= start_j
                    self.model.Add(
                        self.truck_end[truck_i] <= self.truck_start[truck_j]
                    ).OnlyEnforceIf([both_assigned, seq_var])

                    # If seq_var = 0 (j before i): end_j <= start_i
                    self.model.Add(
                        self.truck_end[truck_j] <= self.truck_start[truck_i]
                    ).OnlyEnforceIf([both_assigned, seq_var.Not()])

        # Outbound doors (same logic)
        for door_id in range(self.num_outbound_doors):
            outbound_trucks = [t.truck_id for t in self.instance.outbound_trucks]

            for i, truck_i in enumerate(outbound_trucks):
                for truck_j in outbound_trucks[i+1:]:

                    assign_i = self.outbound_door_assign[(truck_i, door_id)]
                    assign_j = self.outbound_door_assign[(truck_j, door_id)]
                    seq_var = self.door_sequence[(door_id, truck_i, truck_j)]

                    both_assigned = self.model.NewBoolVar(f'both_d{door_id}_t{truck_i}_t{truck_j}')

                    self.model.AddBoolAnd([assign_i, assign_j]).OnlyEnforceIf(both_assigned)
                    self.model.AddBoolOr([assign_i.Not(), assign_j.Not()]).OnlyEnforceIf(both_assigned.Not())

                    self.model.Add(
                        self.truck_end[truck_i] <= self.truck_start[truck_j]
                    ).OnlyEnforceIf([both_assigned, seq_var])

                    self.model.Add(
                        self.truck_end[truck_j] <= self.truck_start[truck_i]
                    ).OnlyEnforceIf([both_assigned, seq_var.Not()])

    def _set_objective(self):
        """Set multi-objective function."""

        objective_terms = []

        # Minimize total waiting time (primary)
        for truck_id, wait_var in self.truck_waiting.items():
            objective_terms.append(int(self.waiting_weight * 100) * wait_var)

        # Balance load across doors (secondary)
        # Minimize variance in door utilization
        if self.num_inbound_doors > 1:
            for door_id in range(self.num_inbound_doors):
                door_load = sum(
                    self.inbound_door_assign[(t.truck_id, door_id)]
                    for t in self.instance.inbound_trucks
                )
                objective_terms.append(int(self.balance_weight * 10) * door_load * door_load)

        if self.num_outbound_doors > 1:
            for door_id in range(self.num_outbound_doors):
                door_load = sum(
                    self.outbound_door_assign[(t.truck_id, door_id)]
                    for t in self.instance.outbound_trucks
                )
                objective_terms.append(int(self.balance_weight * 10) * door_load * door_load)

        # Minimize makespan (tertiary)
        for truck_id, end_var in self.truck_end.items():
            objective_terms.append(end_var)

        self.model.Minimize(sum(objective_terms))

    def solve(self, time_limit_seconds: int = 60) -> DoorAssignment:
        """
        Solve the door assignment model.

        Args:
            time_limit_seconds: Solver time limit

        Returns:
            DoorAssignment with optimized door assignments
        """
        logger.info(f"Solving door assignment model (time limit: {time_limit_seconds}s)...")

        start_time = time.time()

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.log_search_progress = False

        status = solver.Solve(self.model)
        solve_time = time.time() - start_time

        status_name = solver.StatusName(status)
        logger.info(f"Solver status: {status_name} (time: {solve_time:.2f}s)")

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._extract_solution(solver, solve_time, status_name)
        else:
            logger.warning(f"No feasible solution found: {status_name}")
            return self._create_empty_solution(solve_time, status_name)

    def _extract_solution(self, solver: cp_model.CpSolver, solve_time: float,
                         status: str) -> DoorAssignment:
        """Extract solution from solver."""

        # Extract assignments
        inbound_truck_to_door = {}
        for truck in self.instance.inbound_trucks:
            truck_id = truck.truck_id
            for door_id in range(self.num_inbound_doors):
                if solver.Value(self.inbound_door_assign[(truck_id, door_id)]) == 1:
                    inbound_truck_to_door[truck_id] = door_id
                    break

        outbound_truck_to_door = {}
        for truck in self.instance.outbound_trucks:
            truck_id = truck.truck_id
            for door_id in range(self.num_outbound_doors):
                if solver.Value(self.outbound_door_assign[(truck_id, door_id)]) == 1:
                    outbound_truck_to_door[truck_id] = door_id
                    break

        # Extract timing
        truck_start_times = {
            truck_id: float(solver.Value(var))
            for truck_id, var in self.truck_start.items()
        }
        truck_end_times = {
            truck_id: float(solver.Value(var))
            for truck_id, var in self.truck_end.items()
        }

        # Calculate door utilization
        door_utilization = {}
        door_idle_time = {}

        for door_id in range(self.num_inbound_doors):
            trucks_on_door = [
                tid for tid, did in inbound_truck_to_door.items() if did == door_id
            ]
            if trucks_on_door:
                total_busy_time = sum(
                    truck_end_times[tid] - truck_start_times[tid]
                    for tid in trucks_on_door
                )
                makespan = max(truck_end_times[tid] for tid in trucks_on_door)
                door_utilization[f'inbound_{door_id}'] = total_busy_time / makespan if makespan > 0 else 0
                door_idle_time[f'inbound_{door_id}'] = makespan - total_busy_time
            else:
                door_utilization[f'inbound_{door_id}'] = 0
                door_idle_time[f'inbound_{door_id}'] = 0

        for door_id in range(self.num_outbound_doors):
            trucks_on_door = [
                tid for tid, did in outbound_truck_to_door.items() if did == door_id
            ]
            if trucks_on_door:
                total_busy_time = sum(
                    truck_end_times[tid] - truck_start_times[tid]
                    for tid in trucks_on_door
                )
                makespan = max(truck_end_times[tid] for tid in trucks_on_door)
                door_utilization[f'outbound_{door_id}'] = total_busy_time / makespan if makespan > 0 else 0
                door_idle_time[f'outbound_{door_id}'] = makespan - total_busy_time
            else:
                door_utilization[f'outbound_{door_id}'] = 0
                door_idle_time[f'outbound_{door_id}'] = 0

        # Calculate metrics
        total_waiting = sum(
            float(solver.Value(var)) for var in self.truck_waiting.values()
        )
        max_waiting = max(
            float(solver.Value(var)) for var in self.truck_waiting.values()
        ) if self.truck_waiting else 0

        makespan = max(truck_end_times.values()) if truck_end_times else 0

        logger.info(f"Door assignment complete: total waiting {total_waiting:.1f} min, "
                   f"makespan {makespan:.1f} min")

        return DoorAssignment(
            inbound_truck_to_door=inbound_truck_to_door,
            outbound_truck_to_door=outbound_truck_to_door,
            truck_start_times=truck_start_times,
            truck_end_times=truck_end_times,
            door_utilization=door_utilization,
            door_idle_time=door_idle_time,
            total_waiting_time=total_waiting,
            max_waiting_time=max_waiting,
            makespan=makespan,
            num_conflicts=0,  # Model ensures no conflicts
            solve_time=solve_time,
            solver_status=status,
            objective_value=float(solver.ObjectiveValue())
        )

    def _create_empty_solution(self, solve_time: float, status: str) -> DoorAssignment:
        """Create empty solution when solver fails."""
        return DoorAssignment(
            inbound_truck_to_door={},
            outbound_truck_to_door={},
            truck_start_times={},
            truck_end_times={},
            door_utilization={},
            door_idle_time={},
            total_waiting_time=float('inf'),
            max_waiting_time=float('inf'),
            makespan=float('inf'),
            num_conflicts=0,
            solve_time=solve_time,
            solver_status=status,
            objective_value=float('inf')
        )


def optimize_door_assignment(instance: CrossDockInstance,
                             config: Optional[Dict] = None,
                             time_limit: int = 60) -> DoorAssignment:
    """
    Optimize door assignment for a cross-dock instance.

    Args:
        instance: Cross-dock instance
        config: Optional configuration
        time_limit: Solver time limit in seconds

    Returns:
        DoorAssignment with optimized assignments
    """
    model = DoorAssignmentModel(instance, config)
    model.build_model()
    return model.solve(time_limit)
