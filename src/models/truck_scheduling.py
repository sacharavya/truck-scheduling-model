"""
Truck Scheduling Model using OR-Tools MILP.

This module implements optimization for inbound and outbound truck sequencing
to minimize tardiness and maximize throughput while respecting dock capacity,
time windows, and precedence constraints.

Author: Cross-Docking Optimization System
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

from ortools.sat.python import cp_model

from src.data_loader import CrossDockInstance, InboundTruck, OutboundTruck

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TruckSchedule:
    """Result of truck scheduling optimization."""

    # Inbound truck schedule
    inbound_sequence: List[int]  # Ordered list of inbound truck IDs
    inbound_start_times: Dict[int, float]  # truck_id -> start time at door
    inbound_end_times: Dict[int, float]  # truck_id -> end time at door

    # Outbound truck schedule
    outbound_sequence: List[int]  # Ordered list of outbound truck IDs
    outbound_start_times: Dict[int, float]  # truck_id -> start time at door
    outbound_end_times: Dict[int, float]  # truck_id -> end time at door

    # Metrics
    total_tardiness: float  # Total tardiness across all trucks
    num_late_trucks: int  # Number of trucks that miss their due date
    makespan: float  # Total time to process all trucks
    avg_waiting_time: float  # Average waiting time per truck

    # Metadata
    solve_time: float
    solver_status: str
    objective_value: float


class TruckSchedulingModel:
    """
    MILP model for truck scheduling optimization.

    Optimizes the sequence and timing of inbound and outbound trucks
    to minimize tardiness while respecting capacity and time constraints.
    """

    def __init__(self, instance: CrossDockInstance, config: Optional[Dict] = None):
        """
        Initialize truck scheduling model.

        Args:
            instance: Cross-dock instance with truck data
            config: Optional configuration parameters
        """
        self.instance = instance
        self.config = config or {}

        # Processing time parameters (minutes)
        self.unload_time_per_pallet = self.config.get('unload_time_per_pallet', 1.0)
        self.load_time_per_pallet = self.config.get('load_time_per_pallet', 1.0)
        self.truck_setup_time = self.config.get('truck_setup_time', 5.0)

        # Door capacity
        self.num_inbound_doors = self.config.get('num_inbound_doors', 1)
        self.num_outbound_doors = self.config.get('num_outbound_doors', 1)

        # Time horizon
        self.time_horizon = self.config.get('time_horizon', 10080)  # 7 days in minutes

        # Weights for multi-objective
        self.tardiness_weight = self.config.get('tardiness_weight', 1.0)
        self.waiting_weight = self.config.get('waiting_weight', 0.1)

        # OR-Tools model
        self.model = cp_model.CpModel()

        # Decision variables
        self.inbound_start = {}  # truck_id -> start time variable
        self.inbound_end = {}  # truck_id -> end time variable
        self.outbound_start = {}  # truck_id -> start time variable
        self.outbound_end = {}  # truck_id -> end time variable

        self.inbound_tardiness = {}  # truck_id -> tardiness variable
        self.outbound_tardiness = {}  # truck_id -> tardiness variable

        self.inbound_sequence = {}  # (truck_i, truck_j) -> bool (i before j)
        self.outbound_sequence = {}  # (truck_i, truck_j) -> bool (i before j)

        logger.info(f"TruckSchedulingModel initialized for {len(instance.inbound_trucks)} inbound, "
                   f"{len(instance.outbound_trucks)} outbound trucks")

    def build_model(self):
        """Build the complete MILP model."""
        logger.info("Building truck scheduling model...")

        self._create_variables()
        self._add_timing_constraints()
        self._add_capacity_constraints()
        self._add_sequencing_constraints()
        self._add_precedence_constraints()
        self._set_objective()

        logger.info("Model building complete")

    def _create_variables(self):
        """Create decision variables for truck scheduling."""

        # Inbound truck variables
        for truck in self.instance.inbound_trucks:
            truck_id = truck.truck_id

            # Start and end time variables
            self.inbound_start[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time),
                self.time_horizon,
                f'inbound_start_{truck_id}'
            )

            # Calculate processing time (count pallets for this truck)
            num_pallets = len([p for p in self.instance.pallets if p.inbound_truck_id == truck_id])
            processing_time = int(self.truck_setup_time + num_pallets * self.unload_time_per_pallet)

            self.inbound_end[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time) + processing_time,
                self.time_horizon,
                f'inbound_end_{truck_id}'
            )

            # Tardiness variable (for outbound trucks with due dates)
            # Inbound trucks don't have due dates, so tardiness is 0
            self.inbound_tardiness[truck_id] = 0

        # Outbound truck variables
        for truck in self.instance.outbound_trucks:
            truck_id = truck.truck_id

            # Start and end time variables
            self.outbound_start[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time),
                self.time_horizon,
                f'outbound_start_{truck_id}'
            )

            # Processing time based on assigned pallets (estimated)
            estimated_pallets = min(truck.capacity, 20)  # Conservative estimate
            processing_time = int(self.truck_setup_time + estimated_pallets * self.load_time_per_pallet)

            self.outbound_end[truck_id] = self.model.NewIntVar(
                int(truck.arrival_time) + processing_time,
                self.time_horizon,
                f'outbound_end_{truck_id}'
            )

            # Tardiness variable
            self.outbound_tardiness[truck_id] = self.model.NewIntVar(
                0,
                self.time_horizon,
                f'outbound_tardiness_{truck_id}'
            )

        # Sequencing variables (for single door constraint)
        if self.num_inbound_doors == 1:
            inbound_trucks = [t.truck_id for t in self.instance.inbound_trucks]
            for i, truck_i in enumerate(inbound_trucks):
                for truck_j in inbound_trucks[i+1:]:
                    # Binary variable: 1 if truck_i processes before truck_j
                    self.inbound_sequence[(truck_i, truck_j)] = self.model.NewBoolVar(
                        f'inbound_seq_{truck_i}_{truck_j}'
                    )

        if self.num_outbound_doors == 1:
            outbound_trucks = [t.truck_id for t in self.instance.outbound_trucks]
            for i, truck_i in enumerate(outbound_trucks):
                for truck_j in outbound_trucks[i+1:]:
                    self.outbound_sequence[(truck_i, truck_j)] = self.model.NewBoolVar(
                        f'outbound_seq_{truck_i}_{truck_j}'
                    )

    def _add_timing_constraints(self):
        """Add timing constraints for truck processing."""

        # Inbound trucks
        for truck in self.instance.inbound_trucks:
            truck_id = truck.truck_id

            # Start time must be after arrival
            self.model.Add(self.inbound_start[truck_id] >= int(truck.arrival_time))

            # End time = start time + processing time
            num_pallets = len([p for p in self.instance.pallets if p.inbound_truck_id == truck_id])
            processing_time = int(self.truck_setup_time + num_pallets * self.unload_time_per_pallet)

            self.model.Add(
                self.inbound_end[truck_id] == self.inbound_start[truck_id] + processing_time
            )

        # Outbound trucks
        for truck in self.instance.outbound_trucks:
            truck_id = truck.truck_id

            # Start time must be after arrival
            self.model.Add(self.outbound_start[truck_id] >= int(truck.arrival_time))

            # End time = start time + processing time
            estimated_pallets = min(truck.capacity, 20)
            processing_time = int(self.truck_setup_time + estimated_pallets * self.load_time_per_pallet)

            self.model.Add(
                self.outbound_end[truck_id] == self.outbound_start[truck_id] + processing_time
            )

            # Tardiness calculation
            # tardiness = max(0, end_time - due_date)
            self.model.Add(
                self.outbound_tardiness[truck_id] >= self.outbound_end[truck_id] - int(truck.due_date)
            )
            self.model.Add(
                self.outbound_tardiness[truck_id] >= 0
            )

    def _add_capacity_constraints(self):
        """Add door capacity constraints."""

        # Single inbound door: no two trucks can overlap
        if self.num_inbound_doors == 1:
            inbound_trucks = [t.truck_id for t in self.instance.inbound_trucks]
            for i, truck_i in enumerate(inbound_trucks):
                for truck_j in inbound_trucks[i+1:]:
                    # If truck_i before truck_j: end_i <= start_j
                    # If truck_j before truck_i: end_j <= start_i

                    M = self.time_horizon  # Big-M
                    seq_var = self.inbound_sequence[(truck_i, truck_j)]

                    # If seq_var = 1 (i before j): end_i <= start_j
                    self.model.Add(
                        self.inbound_end[truck_i] <= self.inbound_start[truck_j]
                    ).OnlyEnforceIf(seq_var)

                    # If seq_var = 0 (j before i): end_j <= start_i
                    self.model.Add(
                        self.inbound_end[truck_j] <= self.inbound_start[truck_i]
                    ).OnlyEnforceIf(seq_var.Not())

        # Single outbound door: no two trucks can overlap
        if self.num_outbound_doors == 1:
            outbound_trucks = [t.truck_id for t in self.instance.outbound_trucks]
            for i, truck_i in enumerate(outbound_trucks):
                for truck_j in outbound_trucks[i+1:]:
                    seq_var = self.outbound_sequence[(truck_i, truck_j)]

                    # If seq_var = 1 (i before j): end_i <= start_j
                    self.model.Add(
                        self.outbound_end[truck_i] <= self.outbound_start[truck_j]
                    ).OnlyEnforceIf(seq_var)

                    # If seq_var = 0 (j before i): end_j <= start_i
                    self.model.Add(
                        self.outbound_end[truck_j] <= self.outbound_start[truck_i]
                    ).OnlyEnforceIf(seq_var.Not())

    def _add_sequencing_constraints(self):
        """Add sequencing logic constraints."""
        # Sequencing is handled implicitly through the binary variables
        # and capacity constraints in _add_capacity_constraints
        pass

    def _add_precedence_constraints(self):
        """
        Add precedence constraints.

        Pallets from inbound truck must be unloaded before they can be loaded
        onto outbound trucks.
        """
        # For each outbound truck, ensure it starts after all inbound trucks
        # that carry its pallets have finished unloading

        for out_truck in self.instance.outbound_trucks:
            out_truck_id = out_truck.truck_id

            # Find inbound trucks that carry pallets for this outbound truck
            # (This requires pallet assignment, which we don't have yet)
            # For now, ensure outbound starts after some inbound trucks finish

            # Simple heuristic: outbound truck should start after at least
            # some inbound trucks have finished (based on arrival times)
            relevant_inbound = [
                t for t in self.instance.inbound_trucks
                if t.arrival_time <= out_truck.arrival_time
            ]

            if relevant_inbound:
                # Outbound truck should start after earliest relevant inbound finishes
                earliest_inbound = min(relevant_inbound, key=lambda t: t.arrival_time)
                self.model.Add(
                    self.outbound_start[out_truck_id] >= self.inbound_end[earliest_inbound.truck_id]
                )

    def _set_objective(self):
        """Set the multi-objective function."""

        objective_terms = []

        # Minimize outbound truck tardiness (primary objective)
        for truck_id, tardiness_var in self.outbound_tardiness.items():
            objective_terms.append(int(self.tardiness_weight * 1000) * tardiness_var)

        # Minimize waiting time (secondary objective)
        for truck in self.instance.inbound_trucks:
            truck_id = truck.truck_id
            waiting_time = self.inbound_start[truck_id] - int(truck.arrival_time)
            objective_terms.append(int(self.waiting_weight * 10) * waiting_time)

        for truck in self.instance.outbound_trucks:
            truck_id = truck.truck_id
            waiting_time = self.outbound_start[truck_id] - int(truck.arrival_time)
            objective_terms.append(int(self.waiting_weight * 10) * waiting_time)

        # Minimize makespan (tertiary objective)
        # Add small weight to encourage early completion
        for truck_id, end_var in self.inbound_end.items():
            objective_terms.append(end_var)
        for truck_id, end_var in self.outbound_end.items():
            objective_terms.append(end_var)

        self.model.Minimize(sum(objective_terms))

    def solve(self, time_limit_seconds: int = 60) -> TruckSchedule:
        """
        Solve the truck scheduling model.

        Args:
            time_limit_seconds: Time limit for solver

        Returns:
            TruckSchedule with optimized schedule
        """
        logger.info(f"Solving truck scheduling model (time limit: {time_limit_seconds}s)...")

        start_time = time.time()

        # Create solver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.log_search_progress = False

        # Solve
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
                         status: str) -> TruckSchedule:
        """Extract solution from solver."""

        # Extract inbound schedule
        inbound_start_times = {
            truck_id: float(solver.Value(var))
            for truck_id, var in self.inbound_start.items()
        }
        inbound_end_times = {
            truck_id: float(solver.Value(var))
            for truck_id, var in self.inbound_end.items()
        }

        # Sort by start time to get sequence
        inbound_sequence = sorted(
            inbound_start_times.keys(),
            key=lambda tid: inbound_start_times[tid]
        )

        # Extract outbound schedule
        outbound_start_times = {
            truck_id: float(solver.Value(var))
            for truck_id, var in self.outbound_start.items()
        }
        outbound_end_times = {
            truck_id: float(solver.Value(var))
            for truck_id, var in self.outbound_end.items()
        }

        outbound_sequence = sorted(
            outbound_start_times.keys(),
            key=lambda tid: outbound_start_times[tid]
        )

        # Calculate metrics
        total_tardiness = sum(
            float(solver.Value(var))
            for var in self.outbound_tardiness.values()
        )

        num_late_trucks = sum(
            1 for var in self.outbound_tardiness.values()
            if solver.Value(var) > 0
        )

        makespan = max(
            list(inbound_end_times.values()) + list(outbound_end_times.values())
        )

        # Calculate average waiting time
        total_waiting = 0
        for truck in self.instance.inbound_trucks:
            waiting = inbound_start_times[truck.truck_id] - truck.arrival_time
            total_waiting += waiting
        for truck in self.instance.outbound_trucks:
            waiting = outbound_start_times[truck.truck_id] - truck.arrival_time
            total_waiting += waiting

        avg_waiting = total_waiting / (len(self.instance.inbound_trucks) +
                                       len(self.instance.outbound_trucks))

        logger.info(f"Schedule generated: {num_late_trucks} late trucks, "
                   f"total tardiness: {total_tardiness:.1f} min, "
                   f"makespan: {makespan:.1f} min")

        return TruckSchedule(
            inbound_sequence=inbound_sequence,
            inbound_start_times=inbound_start_times,
            inbound_end_times=inbound_end_times,
            outbound_sequence=outbound_sequence,
            outbound_start_times=outbound_start_times,
            outbound_end_times=outbound_end_times,
            total_tardiness=total_tardiness,
            num_late_trucks=num_late_trucks,
            makespan=makespan,
            avg_waiting_time=avg_waiting,
            solve_time=solve_time,
            solver_status=status,
            objective_value=float(solver.ObjectiveValue())
        )

    def _create_empty_solution(self, solve_time: float, status: str) -> TruckSchedule:
        """Create empty solution when solver fails."""
        return TruckSchedule(
            inbound_sequence=[],
            inbound_start_times={},
            inbound_end_times={},
            outbound_sequence=[],
            outbound_start_times={},
            outbound_end_times={},
            total_tardiness=float('inf'),
            num_late_trucks=len(self.instance.outbound_trucks),
            makespan=float('inf'),
            avg_waiting_time=float('inf'),
            solve_time=solve_time,
            solver_status=status,
            objective_value=float('inf')
        )


def optimize_truck_scheduling(instance: CrossDockInstance,
                              config: Optional[Dict] = None,
                              time_limit: int = 60) -> TruckSchedule:
    """
    Optimize truck scheduling for a cross-dock instance.

    Args:
        instance: Cross-dock instance
        config: Optional configuration parameters
        time_limit: Solver time limit in seconds

    Returns:
        TruckSchedule with optimized truck sequence and timing
    """
    model = TruckSchedulingModel(instance, config)
    model.build_model()
    return model.solve(time_limit)
