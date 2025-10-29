"""
Integrated Optimization Model for Cross-Docking.

This module implements a comprehensive MILP model that jointly optimizes:
1. Truck scheduling (sequence and timing)
2. Door assignment (truck-to-door allocation)
3. Pallet assignment (pallet-to-truck matching)

This integrated approach ensures global optimality across all three problems.

Author: Cross-Docking Optimization System
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

from ortools.sat.python import cp_model

from src.data_loader import CrossDockInstance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntegratedSolution:
    """Complete solution for integrated cross-docking optimization."""

    # Pallet assignments
    pallet_to_truck: Dict[int, int]  # pallet_id -> truck_id (or 'UNASSIGNED')
    truck_loads: Dict[int, List[int]]  # truck_id -> list of pallet_ids

    # Door assignments
    inbound_truck_to_door: Dict[int, int]  # truck_id -> door_id
    outbound_truck_to_door: Dict[int, int]  # truck_id -> door_id

    # Scheduling
    inbound_sequence: List[int]  # Ordered truck IDs
    outbound_sequence: List[int]
    truck_start_times: Dict[int, float]  # truck_id -> start time
    truck_end_times: Dict[int, float]  # truck_id -> end time

    # Metrics
    num_late_pallets: int
    service_level: float
    total_tardiness: float
    avg_fill_rate: float
    total_waiting_time: float
    makespan: float
    num_unassigned_pallets: int

    # Metadata
    solve_time: float
    solver_status: str
    objective_value: float


class IntegratedCrossDockModel:
    """
    Integrated MILP model for complete cross-docking optimization.

    Jointly optimizes truck scheduling, door assignment, and pallet assignment
    to achieve global optimality across all decisions.
    """

    def __init__(self, instance: CrossDockInstance, config: Optional[Dict] = None):
        """
        Initialize integrated model.

        Args:
            instance: Cross-dock instance
            config: Optional configuration parameters
        """
        self.instance = instance
        self.config = config or {}

        # Configuration parameters
        self.num_inbound_doors = self.config.get('num_inbound_doors', 1)
        self.num_outbound_doors = self.config.get('num_outbound_doors', 1)
        self.unload_time_per_pallet = self.config.get('unload_time_per_pallet', 1.0)
        self.load_time_per_pallet = self.config.get('load_time_per_pallet', 1.0)
        self.truck_setup_time = self.config.get('truck_setup_time', 5.0)
        self.time_horizon = self.config.get('time_horizon', 10080)

        # Objective weights
        self.tardiness_weight = self.config.get('tardiness_weight', 100.0)
        self.fill_rate_weight = self.config.get('fill_rate_weight', 10.0)
        self.waiting_weight = self.config.get('waiting_weight', 1.0)
        self.unassigned_penalty = self.config.get('unassigned_penalty', 1000.0)

        # Create data structures
        self._prepare_data()

        # OR-Tools model
        self.model = cp_model.CpModel()

        # Decision variables
        self.pallet_assign = {}  # (pallet_id, truck_id) -> bool
        self.door_assign = {}  # (truck_id, door_id) -> bool
        self.truck_start = {}  # truck_id -> start time
        self.truck_end = {}  # truck_id -> end time
        self.truck_waiting = {}  # truck_id -> waiting time
        self.pallet_tardiness = {}  # pallet_id -> tardiness
        self.truck_sequence = {}  # (door_id, truck_i, truck_j) -> bool

        logger.info(f"IntegratedModel initialized: {len(instance.pallets)} pallets, "
                   f"{len(instance.inbound_trucks)} inbound, {len(instance.outbound_trucks)} outbound trucks")

    def _prepare_data(self):
        """Prepare data structures for optimization."""

        # Group trucks by destination
        self.trucks_by_dest = {}
        for truck in self.instance.outbound_trucks:
            dest = truck.destination
            if dest not in self.trucks_by_dest:
                self.trucks_by_dest[dest] = []
            self.trucks_by_dest[dest].append(truck)

        # Group pallets by destination
        self.pallets_by_dest = {}
        for pallet in self.instance.pallets:
            dest = pallet.destination
            if dest not in self.pallets_by_dest:
                self.pallets_by_dest[dest] = []
            self.pallets_by_dest[dest].append(pallet)

        # Create pallet dictionary for easy lookup
        self.pallet_dict = {p.pallet_id: p for p in self.instance.pallets}
        self.outbound_truck_dict = {t.truck_id: t for t in self.instance.outbound_trucks}
        self.inbound_truck_dict = {t.truck_id: t for t in self.instance.inbound_trucks}

    def build_model(self):
        """Build the complete integrated model."""
        logger.info("Building integrated cross-docking model...")

        self._create_variables()
        self._add_pallet_assignment_constraints()
        self._add_door_assignment_constraints()
        self._add_scheduling_constraints()
        self._add_integration_constraints()
        self._set_objective()

        logger.info("Integrated model building complete")

    def _create_variables(self):
        """Create all decision variables."""

        # 1. Pallet assignment variables
        for pallet in self.instance.pallets:
            pallet_id = pallet.pallet_id
            dest = pallet.destination

            # Can assign to any truck going to same destination
            valid_trucks = [t.truck_id for t in self.trucks_by_dest.get(dest, [])]

            for truck_id in valid_trucks:
                self.pallet_assign[(pallet_id, truck_id)] = self.model.NewBoolVar(
                    f'pallet_{pallet_id}_truck_{truck_id}'
                )

            # Unassigned option
            self.pallet_assign[(pallet_id, 'UNASSIGNED')] = self.model.NewBoolVar(
                f'pallet_{pallet_id}_unassigned'
            )

            # Tardiness variable
            self.pallet_tardiness[pallet_id] = self.model.NewIntVar(
                0, self.time_horizon, f'tardiness_{pallet_id}'
            )

        # 2. Door assignment variables
        for truck in self.instance.inbound_trucks:
            for door_id in range(self.num_inbound_doors):
                self.door_assign[(truck.truck_id, door_id, 'inbound')] = self.model.NewBoolVar(
                    f'inbound_{truck.truck_id}_door_{door_id}'
                )

        for truck in self.instance.outbound_trucks:
            for door_id in range(self.num_outbound_doors):
                self.door_assign[(truck.truck_id, door_id, 'outbound')] = self.model.NewBoolVar(
                    f'outbound_{truck.truck_id}_door_{door_id}'
                )

        # 3. Scheduling variables
        for truck in list(self.instance.inbound_trucks) + list(self.instance.outbound_trucks):
            truck_id = truck.truck_id

            self.truck_start[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time),
                self.time_horizon,
                f'start_{truck_id}'
            )

            self.truck_end[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time),
                self.time_horizon,
                f'end_{truck_id}'
            )

            self.truck_waiting[truck_id] = self.model.NewIntVar(
                0, self.time_horizon, f'wait_{truck_id}'
            )

        # 4. Sequencing variables for each door
        for door_id in range(self.num_inbound_doors):
            trucks = [t.truck_id for t in self.instance.inbound_trucks]
            for i, ti in enumerate(trucks):
                for tj in trucks[i+1:]:
                    self.truck_sequence[(door_id, ti, tj, 'inbound')] = self.model.NewBoolVar(
                        f'seq_door{door_id}_in_{ti}_{tj}'
                    )

        for door_id in range(self.num_outbound_doors):
            trucks = [t.truck_id for t in self.instance.outbound_trucks]
            for i, ti in enumerate(trucks):
                for tj in trucks[i+1:]:
                    self.truck_sequence[(door_id, ti, tj, 'outbound')] = self.model.NewBoolVar(
                        f'seq_door{door_id}_out_{ti}_{tj}'
                    )

    def _add_pallet_assignment_constraints(self):
        """Add pallet assignment constraints."""

        # Each pallet assigned to exactly one truck (or unassigned)
        for pallet in self.instance.pallets:
            pallet_id = pallet.pallet_id
            dest = pallet.destination

            assignment_vars = []
            valid_trucks = [t.truck_id for t in self.trucks_by_dest.get(dest, [])]

            for truck_id in valid_trucks:
                assignment_vars.append(self.pallet_assign[(pallet_id, truck_id)])

            # Add unassigned option
            assignment_vars.append(self.pallet_assign[(pallet_id, 'UNASSIGNED')])

            # Exactly one assignment
            self.model.Add(sum(assignment_vars) == 1)

        # Capacity constraints for each truck
        for truck in self.instance.outbound_trucks:
            truck_id = truck.truck_id
            dest = truck.destination

            pallets_for_dest = [p.pallet_id for p in self.pallets_by_dest.get(dest, [])]

            assigned_pallets = [
                self.pallet_assign[(pid, truck_id)]
                for pid in pallets_for_dest
                if (pid, truck_id) in self.pallet_assign
            ]

            if assigned_pallets:
                self.model.Add(sum(assigned_pallets) <= truck.capacity)

    def _add_door_assignment_constraints(self):
        """Add door assignment constraints."""

        # Each truck assigned to exactly one door
        for truck in self.instance.inbound_trucks:
            door_vars = [
                self.door_assign[(truck.truck_id, d, 'inbound')]
                for d in range(self.num_inbound_doors)
            ]
            self.model.Add(sum(door_vars) == 1)

        for truck in self.instance.outbound_trucks:
            door_vars = [
                self.door_assign[(truck.truck_id, d, 'outbound')]
                for d in range(self.num_outbound_doors)
            ]
            self.model.Add(sum(door_vars) == 1)

    def _add_scheduling_constraints(self):
        """Add scheduling and timing constraints."""

        # Processing time and waiting time constraints
        for truck in list(self.instance.inbound_trucks) + list(self.instance.outbound_trucks):
            truck_id = truck.truck_id

            # Waiting time = start - arrival
            self.model.Add(
                self.truck_waiting[truck_id] == self.truck_start[truck_id] - int(truck.arrival_time)
            )

            # Processing time (simplified - will be adjusted based on actual assignments)
            if truck in self.instance.inbound_trucks:
                num_pallets = len([p for p in self.instance.pallets if p.inbound_truck_id == truck_id])
                proc_time = int(self.truck_setup_time + num_pallets * self.unload_time_per_pallet)
            else:
                # Outbound: estimate based on capacity
                proc_time = int(self.truck_setup_time + 20 * self.load_time_per_pallet)

            self.model.Add(self.truck_end[truck_id] == self.truck_start[truck_id] + proc_time)

        # Non-overlap constraints for trucks on same door
        self._add_door_non_overlap_constraints()

    def _add_door_non_overlap_constraints(self):
        """Ensure trucks on the same door don't overlap."""

        # Inbound doors
        for door_id in range(self.num_inbound_doors):
            trucks = [t.truck_id for t in self.instance.inbound_trucks]
            for i, ti in enumerate(trucks):
                for tj in trucks[i+1:]:
                    # Both assigned to this door?
                    both_here = self.model.NewBoolVar(f'both_door{door_id}_in_{ti}_{tj}')

                    assign_i = self.door_assign[(ti, door_id, 'inbound')]
                    assign_j = self.door_assign[(tj, door_id, 'inbound')]

                    self.model.AddBoolAnd([assign_i, assign_j]).OnlyEnforceIf(both_here)
                    self.model.AddBoolOr([assign_i.Not(), assign_j.Not()]).OnlyEnforceIf(both_here.Not())

                    # If both here, enforce sequencing
                    seq_var = self.truck_sequence[(door_id, ti, tj, 'inbound')]

                    # ti before tj
                    self.model.Add(
                        self.truck_end[ti] <= self.truck_start[tj]
                    ).OnlyEnforceIf([both_here, seq_var])

                    # tj before ti
                    self.model.Add(
                        self.truck_end[tj] <= self.truck_start[ti]
                    ).OnlyEnforceIf([both_here, seq_var.Not()])

        # Outbound doors (same logic)
        for door_id in range(self.num_outbound_doors):
            trucks = [t.truck_id for t in self.instance.outbound_trucks]
            for i, ti in enumerate(trucks):
                for tj in trucks[i+1:]:
                    both_here = self.model.NewBoolVar(f'both_door{door_id}_out_{ti}_{tj}')

                    assign_i = self.door_assign[(ti, door_id, 'outbound')]
                    assign_j = self.door_assign[(tj, door_id, 'outbound')]

                    self.model.AddBoolAnd([assign_i, assign_j]).OnlyEnforceIf(both_here)
                    self.model.AddBoolOr([assign_i.Not(), assign_j.Not()]).OnlyEnforceIf(both_here.Not())

                    seq_var = self.truck_sequence[(door_id, ti, tj, 'outbound')]

                    self.model.Add(
                        self.truck_end[ti] <= self.truck_start[tj]
                    ).OnlyEnforceIf([both_here, seq_var])

                    self.model.Add(
                        self.truck_end[tj] <= self.truck_start[ti]
                    ).OnlyEnforceIf([both_here, seq_var.Not()])

    def _add_integration_constraints(self):
        """
        Add constraints that link the three sub-problems.

        Key integration: outbound trucks can only start loading after
        their assigned pallets have been unloaded from inbound trucks.
        """

        for out_truck in self.instance.outbound_trucks:
            out_truck_id = out_truck.truck_id
            dest = out_truck.destination

            # For pallets that might be assigned to this truck
            pallets_for_dest = [p for p in self.pallets_by_dest.get(dest, [])]

            for pallet in pallets_for_dest:
                pallet_id = pallet.pallet_id
                inbound_truck_id = pallet.inbound_truck_id

                # If pallet assigned to this outbound truck,
                # outbound must start after inbound finishes
                if (pallet_id, out_truck_id) in self.pallet_assign:
                    is_assigned = self.pallet_assign[(pallet_id, out_truck_id)]

                    # Only enforce if assigned
                    self.model.Add(
                        self.truck_start[out_truck_id] >= self.truck_end[inbound_truck_id]
                    ).OnlyEnforceIf(is_assigned)

        # Pallet tardiness calculation
        for pallet in self.instance.pallets:
            pallet_id = pallet.pallet_id
            dest = pallet.destination

            valid_trucks = [t.truck_id for t in self.trucks_by_dest.get(dest, [])]

            for truck_id in valid_trucks:
                if (pallet_id, truck_id) in self.pallet_assign:
                    is_assigned = self.pallet_assign[(pallet_id, truck_id)]

                    # If assigned to this truck, tardiness = max(0, truck_end - pallet_due)
                    self.model.Add(
                        self.pallet_tardiness[pallet_id] >=
                        self.truck_end[truck_id] - int(pallet.due_date)
                    ).OnlyEnforceIf(is_assigned)

            # If unassigned, large tardiness
            is_unassigned = self.pallet_assign[(pallet_id, 'UNASSIGNED')]
            self.model.Add(
                self.pallet_tardiness[pallet_id] >= self.time_horizon // 2
            ).OnlyEnforceIf(is_unassigned)

    def _set_objective(self):
        """Set multi-objective function balancing all three problems."""

        objective_terms = []

        # 1. Minimize pallet tardiness (highest priority)
        for pallet_id, tard_var in self.pallet_tardiness.items():
            objective_terms.append(int(self.tardiness_weight) * tard_var)

        # 2. Penalize unassigned pallets
        for pallet in self.instance.pallets:
            pallet_id = pallet.pallet_id
            unassigned_var = self.pallet_assign[(pallet_id, 'UNASSIGNED')]
            objective_terms.append(int(self.unassigned_penalty) * unassigned_var)

        # 3. Maximize fill rate (encourage full trucks)
        for truck in self.instance.outbound_trucks:
            truck_id = truck.truck_id
            dest = truck.destination

            pallets_for_dest = [p.pallet_id for p in self.pallets_by_dest.get(dest, [])]
            assigned_to_this_truck = [
                self.pallet_assign[(pid, truck_id)]
                for pid in pallets_for_dest
                if (pid, truck_id) in self.pallet_assign
            ]

            if assigned_to_this_truck:
                # Reward for loading pallets
                num_assigned = sum(assigned_to_this_truck)
                objective_terms.append(-int(self.fill_rate_weight) * num_assigned)

        # 4. Minimize waiting time
        for truck_id, wait_var in self.truck_waiting.items():
            objective_terms.append(int(self.waiting_weight) * wait_var)

        # 5. Minimize makespan (lowest priority)
        for truck_id, end_var in self.truck_end.items():
            objective_terms.append(end_var)

        self.model.Minimize(sum(objective_terms))

    def solve(self, time_limit_seconds: int = 300) -> IntegratedSolution:
        """
        Solve the integrated model.

        Args:
            time_limit_seconds: Solver time limit (default 5 minutes for complex model)

        Returns:
            IntegratedSolution with complete optimization results
        """
        logger.info(f"Solving integrated model (time limit: {time_limit_seconds}s)...")

        start_time = time.time()

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = 4  # Parallel search

        status = solver.Solve(self.model)
        solve_time = time.time() - start_time

        status_name = solver.StatusName(status)
        logger.info(f"Integrated solver status: {status_name} (time: {solve_time:.2f}s)")

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._extract_solution(solver, solve_time, status_name)
        else:
            logger.warning(f"No feasible solution found: {status_name}")
            return self._create_empty_solution(solve_time, status_name)

    def _extract_solution(self, solver: cp_model.CpSolver, solve_time: float,
                         status: str) -> IntegratedSolution:
        """Extract complete solution from solver."""

        # Extract pallet assignments
        pallet_to_truck = {}
        truck_loads = {t.truck_id: [] for t in self.instance.outbound_trucks}

        for pallet in self.instance.pallets:
            pallet_id = pallet.pallet_id
            dest = pallet.destination

            assigned_truck = None
            valid_trucks = [t.truck_id for t in self.trucks_by_dest.get(dest, [])]

            for truck_id in valid_trucks:
                if (pallet_id, truck_id) in self.pallet_assign:
                    if solver.Value(self.pallet_assign[(pallet_id, truck_id)]) == 1:
                        assigned_truck = truck_id
                        truck_loads[truck_id].append(pallet_id)
                        break

            if assigned_truck is None:
                # Check if unassigned
                if solver.Value(self.pallet_assign[(pallet_id, 'UNASSIGNED')]) == 1:
                    assigned_truck = 'UNASSIGNED'

            pallet_to_truck[pallet_id] = assigned_truck

        # Extract door assignments
        inbound_truck_to_door = {}
        for truck in self.instance.inbound_trucks:
            for door_id in range(self.num_inbound_doors):
                if solver.Value(self.door_assign[(truck.truck_id, door_id, 'inbound')]) == 1:
                    inbound_truck_to_door[truck.truck_id] = door_id
                    break

        outbound_truck_to_door = {}
        for truck in self.instance.outbound_trucks:
            for door_id in range(self.num_outbound_doors):
                if solver.Value(self.door_assign[(truck.truck_id, door_id, 'outbound')]) == 1:
                    outbound_truck_to_door[truck.truck_id] = door_id
                    break

        # Extract scheduling
        truck_start_times = {tid: float(solver.Value(var)) for tid, var in self.truck_start.items()}
        truck_end_times = {tid: float(solver.Value(var)) for tid, var in self.truck_end.items()}

        inbound_sequence = sorted(
            [t.truck_id for t in self.instance.inbound_trucks],
            key=lambda tid: truck_start_times[tid]
        )
        outbound_sequence = sorted(
            [t.truck_id for t in self.instance.outbound_trucks],
            key=lambda tid: truck_start_times[tid]
        )

        # Calculate metrics
        num_unassigned = sum(1 for t in pallet_to_truck.values() if t == 'UNASSIGNED')
        num_late_pallets = sum(
            1 for pid in self.pallet_tardiness.keys()
            if solver.Value(self.pallet_tardiness[pid]) > 0
        )

        total_pallets = len(self.instance.pallets)
        service_level = (total_pallets - num_late_pallets) / total_pallets if total_pallets > 0 else 0

        total_tardiness = sum(
            float(solver.Value(var)) for var in self.pallet_tardiness.values()
        )

        # Calculate fill rate
        fill_rates = []
        for truck in self.instance.outbound_trucks:
            if truck.truck_id in truck_loads and len(truck_loads[truck.truck_id]) > 0:
                fill_rates.append(len(truck_loads[truck.truck_id]) / truck.capacity)

        avg_fill_rate = sum(fill_rates) / len(fill_rates) if fill_rates else 0

        total_waiting = sum(float(solver.Value(var)) for var in self.truck_waiting.values())
        makespan = max(truck_end_times.values()) if truck_end_times else 0

        logger.info(f"Integrated solution: {num_late_pallets} late pallets, "
                   f"{num_unassigned} unassigned, service level: {service_level:.2%}")

        return IntegratedSolution(
            pallet_to_truck=pallet_to_truck,
            truck_loads=truck_loads,
            inbound_truck_to_door=inbound_truck_to_door,
            outbound_truck_to_door=outbound_truck_to_door,
            inbound_sequence=inbound_sequence,
            outbound_sequence=outbound_sequence,
            truck_start_times=truck_start_times,
            truck_end_times=truck_end_times,
            num_late_pallets=num_late_pallets,
            service_level=service_level,
            total_tardiness=total_tardiness,
            avg_fill_rate=avg_fill_rate,
            total_waiting_time=total_waiting,
            makespan=makespan,
            num_unassigned_pallets=num_unassigned,
            solve_time=solve_time,
            solver_status=status,
            objective_value=float(solver.ObjectiveValue())
        )

    def _create_empty_solution(self, solve_time: float, status: str) -> IntegratedSolution:
        """Create empty solution when solver fails."""
        return IntegratedSolution(
            pallet_to_truck={},
            truck_loads={},
            inbound_truck_to_door={},
            outbound_truck_to_door={},
            inbound_sequence=[],
            outbound_sequence=[],
            truck_start_times={},
            truck_end_times={},
            num_late_pallets=len(self.instance.pallets),
            service_level=0.0,
            total_tardiness=float('inf'),
            avg_fill_rate=0.0,
            total_waiting_time=float('inf'),
            makespan=float('inf'),
            num_unassigned_pallets=len(self.instance.pallets),
            solve_time=solve_time,
            solver_status=status,
            objective_value=float('inf')
        )


def optimize_integrated(instance: CrossDockInstance,
                       config: Optional[Dict] = None,
                       time_limit: int = 300) -> IntegratedSolution:
    """
    Solve the integrated cross-docking optimization problem.

    Args:
        instance: Cross-dock instance
        config: Optional configuration
        time_limit: Solver time limit in seconds (default 5 min for complex model)

    Returns:
        IntegratedSolution with complete optimization results
    """
    model = IntegratedCrossDockModel(instance, config)
    model.build_model()
    return model.solve(time_limit)
