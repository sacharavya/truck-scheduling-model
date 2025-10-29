"""
Discrete-Event Simulation for Cross-Docking Terminal

This module implements a SimPy-based discrete-event simulation to validate
optimization solutions and analyze terminal operations under realistic conditions.

Features:
- Truck arrival/departure events
- Forklift resource management
- Pallet unloading/loading processes
- Queue management
- Performance tracking

Author: Cross-Docking Optimization Project
"""

import simpy
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import CrossDockInstance, InboundTruck, OutboundTruck, Pallet
from models.pallet_assignment import AssignmentSolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    num_forklifts: int = 15
    num_inbound_doors: int = 1
    num_outbound_doors: int = 1

    # Processing times (minutes)
    pallet_unload_time: float = 2.0  # Time to unload one pallet from inbound truck
    pallet_load_time: float = 2.0    # Time to load one pallet onto outbound truck
    truck_positioning_time: float = 5.0  # Time for truck to dock/undock
    pallet_transfer_time: float = 1.5   # Time to move pallet to staging area

    # Simulation settings
    warmup_period: float = 0.0  # Warm-up period (minutes)
    random_seed: Optional[int] = 42


@dataclass
class SimulationResults:
    """Results from simulation run."""
    instance_name: str
    solution_name: str

    # Completion metrics
    total_pallets_processed: int = 0
    pallets_delivered_on_time: int = 0
    pallets_delivered_late: int = 0
    total_tardiness: float = 0.0

    # Resource utilization
    avg_forklift_utilization: float = 0.0
    avg_inbound_door_utilization: float = 0.0
    avg_outbound_door_utilization: float = 0.0

    # Queue statistics
    avg_inbound_queue_length: float = 0.0
    avg_outbound_queue_length: float = 0.0
    max_inbound_queue_length: int = 0
    max_outbound_queue_length: int = 0

    # Inventory
    avg_staging_inventory: float = 0.0
    max_staging_inventory: int = 0

    # Time statistics
    avg_pallet_flow_time: float = 0.0  # Time from arrival to departure
    makespan: float = 0.0  # Total simulation time

    # Events log
    events: List[Dict] = field(default_factory=list)

    def print_summary(self):
        """Print simulation results summary."""
        print(f"\n{'='*70}")
        print(f"SIMULATION RESULTS")
        print(f"{'='*70}")
        print(f"Instance: {self.instance_name}")
        print(f"Solution: {self.solution_name}")
        print(f"Makespan: {self.makespan:.2f} minutes")

        print(f"\n📦 PALLET FLOW")
        print(f"  Total Processed: {self.total_pallets_processed}")
        print(f"  On-Time: {self.pallets_delivered_on_time}")
        print(f"  Late: {self.pallets_delivered_late}")
        if self.total_pallets_processed > 0:
            on_time_pct = (self.pallets_delivered_on_time / self.total_pallets_processed) * 100
            print(f"  Service Level: {on_time_pct:.2f}%")
        print(f"  Total Tardiness: {self.total_tardiness:.2f} minutes")
        print(f"  Avg Flow Time: {self.avg_pallet_flow_time:.2f} minutes")

        print(f"\n🔧 RESOURCE UTILIZATION")
        print(f"  Forklifts: {self.avg_forklift_utilization:.2%}")
        print(f"  Inbound Doors: {self.avg_inbound_door_utilization:.2%}")
        print(f"  Outbound Doors: {self.avg_outbound_door_utilization:.2%}")

        print(f"\n📊 QUEUE STATISTICS")
        print(f"  Avg Inbound Queue: {self.avg_inbound_queue_length:.2f} trucks")
        print(f"  Max Inbound Queue: {self.max_inbound_queue_length} trucks")
        print(f"  Avg Outbound Queue: {self.avg_outbound_queue_length:.2f} trucks")
        print(f"  Max Outbound Queue: {self.max_outbound_queue_length} trucks")

        print(f"\n📦 STAGING INVENTORY")
        print(f"  Average: {self.avg_staging_inventory:.2f} pallets")
        print(f"  Maximum: {self.max_staging_inventory} pallets")

        print(f"{'='*70}\n")


class CrossDockSimulation:
    """Discrete-event simulation of cross-docking terminal."""

    def __init__(
        self,
        instance: CrossDockInstance,
        solution: AssignmentSolution,
        config: Optional[SimulationConfig] = None
    ):
        """Initialize simulation."""
        self.instance = instance
        self.solution = solution
        self.config = config or SimulationConfig()

        # Create SimPy environment
        self.env = simpy.Environment()

        # Resources
        self.forklifts = simpy.Resource(self.env, capacity=self.config.num_forklifts)
        self.inbound_doors = simpy.Resource(self.env, capacity=self.config.num_inbound_doors)
        self.outbound_doors = simpy.Resource(self.env, capacity=self.config.num_outbound_doors)

        # State tracking
        self.staging_area = {}  # pallet_id -> arrival_time in staging
        self.pallet_assignments = {}  # pallet_id -> truck_id (from solution)
        self.truck_pallets = defaultdict(list)  # truck_id -> list of pallet_ids

        # Statistics
        self.events = []
        self.staging_inventory_samples = []
        self.queue_samples = {'inbound': [], 'outbound': []}
        self.pallet_flow_times = []
        self.tardiness_values = []
        self.on_time_count = 0
        self.late_count = 0

        # Build assignment mapping from solution
        self._build_assignment_mapping()

        # Lookup dictionaries
        self.truck_dict = {t.truck_id: t for t in instance.outbound_trucks}
        self.pallet_dict = {p.pallet_id: p for p in instance.pallets}
        self.inbound_dict = {t.truck_id: t for t in instance.inbound_trucks}

        logger.info(f"Simulation initialized for {instance.instance_name}")

    def _build_assignment_mapping(self):
        """Build pallet-to-truck assignment mapping from solution."""
        for truck_id, pallet_ids in self.solution.truck_loads.items():
            if truck_id != 'UNASSIGNED':
                for pallet_id in pallet_ids:
                    self.pallet_assignments[pallet_id] = truck_id
                    self.truck_pallets[truck_id].append(pallet_id)

    def _log_event(self, time: float, event_type: str, details: Dict):
        """Log simulation event."""
        self.events.append({
            'time': time,
            'event_type': event_type,
            **details
        })

    def run(self) -> SimulationResults:
        """Run the simulation."""
        logger.info("Starting simulation...")

        # Schedule inbound truck arrivals
        for truck in self.instance.inbound_trucks:
            self.env.process(self.inbound_truck_process(truck))

        # Schedule outbound truck arrivals
        for truck in self.instance.outbound_trucks:
            self.env.process(self.outbound_truck_process(truck))

        # Start monitoring processes
        self.env.process(self.monitor_inventory())
        self.env.process(self.monitor_queues())

        # Run simulation
        self.env.run()

        # Compile results
        results = self._compile_results()

        logger.info("Simulation completed")
        return results

    def inbound_truck_process(self, truck: InboundTruck):
        """Process inbound truck arrival and unloading."""
        # Wait until arrival time
        yield self.env.timeout(truck.arrival_time)

        self._log_event(self.env.now, 'inbound_arrival', {'truck_id': truck.truck_id})

        # Request inbound door
        with self.inbound_doors.request() as door_request:
            yield door_request

            # Positioning time
            yield self.env.timeout(self.config.truck_positioning_time)

            # Get pallets on this truck
            truck_pallets = [p for p in self.instance.pallets if p.inbound_truck_id == truck.truck_id]

            # Unload each pallet
            for pallet in truck_pallets:
                # Request forklift
                with self.forklifts.request() as forklift:
                    yield forklift

                    # Unload pallet
                    yield self.env.timeout(self.config.pallet_unload_time)

                    # Transfer to staging area
                    yield self.env.timeout(self.config.pallet_transfer_time)

                    # Place in staging
                    self.staging_area[pallet.pallet_id] = self.env.now

                    self._log_event(self.env.now, 'pallet_staged', {
                        'pallet_id': pallet.pallet_id,
                        'inbound_truck': truck.truck_id
                    })

            # Truck departs
            yield self.env.timeout(self.config.truck_positioning_time)

            self._log_event(self.env.now, 'inbound_departure', {'truck_id': truck.truck_id})

    def outbound_truck_process(self, truck: OutboundTruck):
        """Process outbound truck arrival and loading."""
        # Wait until arrival time
        yield self.env.timeout(truck.arrival_time)

        self._log_event(self.env.now, 'outbound_arrival', {'truck_id': truck.truck_id})

        # Request outbound door
        with self.outbound_doors.request() as door_request:
            yield door_request

            # Positioning time
            yield self.env.timeout(self.config.truck_positioning_time)

            # Get pallets assigned to this truck
            assigned_pallets = self.truck_pallets.get(truck.truck_id, [])

            loaded_pallets = []

            # Load each assigned pallet (when available in staging)
            for pallet_id in assigned_pallets:
                # Wait until pallet is in staging area
                while pallet_id not in self.staging_area:
                    yield self.env.timeout(1)  # Check every minute

                # Request forklift
                with self.forklifts.request() as forklift:
                    yield forklift

                    # Remove from staging
                    staging_arrival_time = self.staging_area.pop(pallet_id)

                    # Transfer and load
                    yield self.env.timeout(self.config.pallet_transfer_time)
                    yield self.env.timeout(self.config.pallet_load_time)

                    loaded_pallets.append(pallet_id)

                    # Calculate flow time
                    pallet = self.pallet_dict[pallet_id]
                    inbound_truck = self.inbound_dict[pallet.inbound_truck_id]
                    flow_time = self.env.now - inbound_truck.arrival_time
                    self.pallet_flow_times.append(flow_time)

                    # Check if on-time
                    if self.env.now <= pallet.due_date:
                        self.on_time_count += 1
                    else:
                        self.late_count += 1
                        tardiness = self.env.now - pallet.due_date
                        self.tardiness_values.append(tardiness)

                    self._log_event(self.env.now, 'pallet_loaded', {
                        'pallet_id': pallet_id,
                        'outbound_truck': truck.truck_id,
                        'flow_time': flow_time
                    })

            # Truck departs
            yield self.env.timeout(self.config.truck_positioning_time)

            self._log_event(self.env.now, 'outbound_departure', {
                'truck_id': truck.truck_id,
                'pallets_loaded': len(loaded_pallets),
                'departure_time': self.env.now,
                'due_date': truck.due_date,
                'late': self.env.now > truck.due_date
            })

    def monitor_inventory(self):
        """Monitor staging area inventory."""
        while True:
            self.staging_inventory_samples.append((self.env.now, len(self.staging_area)))
            yield self.env.timeout(10)  # Sample every 10 minutes

    def monitor_queues(self):
        """Monitor queue lengths."""
        while True:
            inbound_queue = len(self.inbound_doors.queue)
            outbound_queue = len(self.outbound_doors.queue)

            self.queue_samples['inbound'].append((self.env.now, inbound_queue))
            self.queue_samples['outbound'].append((self.env.now, outbound_queue))

            yield self.env.timeout(5)  # Sample every 5 minutes

    def _compile_results(self) -> SimulationResults:
        """Compile simulation statistics into results object."""
        # Calculate averages
        avg_staging = np.mean([inv for _, inv in self.staging_inventory_samples]) if self.staging_inventory_samples else 0
        max_staging = max([inv for _, inv in self.staging_inventory_samples]) if self.staging_inventory_samples else 0

        avg_inbound_queue = np.mean([q for _, q in self.queue_samples['inbound']]) if self.queue_samples['inbound'] else 0
        max_inbound_queue = max([q for _, q in self.queue_samples['inbound']]) if self.queue_samples['inbound'] else 0

        avg_outbound_queue = np.mean([q for _, q in self.queue_samples['outbound']]) if self.queue_samples['outbound'] else 0
        max_outbound_queue = max([q for _, q in self.queue_samples['outbound']]) if self.queue_samples['outbound'] else 0

        avg_flow_time = np.mean(self.pallet_flow_times) if self.pallet_flow_times else 0

        total_tardiness = sum(self.tardiness_values)

        # TODO: Calculate resource utilization properly
        # For now, use placeholder values
        avg_forklift_util = 0.0
        avg_inbound_door_util = 0.0
        avg_outbound_door_util = 0.0

        results = SimulationResults(
            instance_name=self.instance.instance_name,
            solution_name=self.solution.status,
            total_pallets_processed=self.on_time_count + self.late_count,
            pallets_delivered_on_time=self.on_time_count,
            pallets_delivered_late=self.late_count,
            total_tardiness=total_tardiness,
            avg_forklift_utilization=avg_forklift_util,
            avg_inbound_door_utilization=avg_inbound_door_util,
            avg_outbound_door_utilization=avg_outbound_door_util,
            avg_inbound_queue_length=avg_inbound_queue,
            avg_outbound_queue_length=avg_outbound_queue,
            max_inbound_queue_length=max_inbound_queue,
            max_outbound_queue_length=max_outbound_queue,
            avg_staging_inventory=avg_staging,
            max_staging_inventory=max_staging,
            avg_pallet_flow_time=avg_flow_time,
            makespan=self.env.now,
            events=self.events
        )

        return results


# Testing
if __name__ == "__main__":
    import os
    from data_loader import DataLoader
    from models.heuristics import earliest_due_date

    # Change to project root
    os.chdir(Path(__file__).parent.parent.parent)

    # Load small instance for testing
    loader = DataLoader()
    instance = loader.load_instance('LL_168h', 1)

    print(f"\n{'='*70}")
    print(f"TESTING CROSS-DOCK SIMULATION")
    print(f"{'='*70}")
    print(f"Instance: {instance.instance_name}")
    print(f"Pallets: {len(instance.pallets)}")

    # Solve with EDD heuristic
    print("\nSolving with EDD heuristic...")
    solution = earliest_due_date(instance)
    print(f"Solution: {solution.status}, Fill Rate: {solution.avg_fill_rate:.2%}")

    # Run simulation
    print("\nRunning simulation...")
    config = SimulationConfig(num_forklifts=15, random_seed=42)
    sim = CrossDockSimulation(instance, solution, config)
    results = sim.run()

    # Print results
    results.print_summary()

    print(f"\nFirst 10 events:")
    events_df = pd.DataFrame(results.events[:10])
    print(events_df.to_string())
