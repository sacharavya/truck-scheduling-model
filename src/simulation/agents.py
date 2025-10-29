"""
Multi-Agent Components for Cross-Docking Simulation

This module implements intelligent agents for the cross-docking simulation,
including truck agents, forklift agents, and a terminal coordinator.

Agents have autonomous decision-making capabilities and can adapt to
real-time conditions in the terminal.

Author: Cross-Docking Optimization Project
"""

import simpy
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import InboundTruck, OutboundTruck, Pallet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TruckState(Enum):
    """States for truck agents."""
    EN_ROUTE = "en_route"
    WAITING_FOR_DOOR = "waiting_for_door"
    AT_DOOR = "at_door"
    LOADING = "loading"
    UNLOADING = "unloading"
    DEPARTING = "departing"
    COMPLETED = "completed"


class ForkliftState(Enum):
    """States for forklift agents."""
    IDLE = "idle"
    MOVING_TO_INBOUND = "moving_to_inbound"
    UNLOADING = "unloading"
    MOVING_TO_STAGING = "moving_to_staging"
    MOVING_TO_OUTBOUND = "moving_to_outbound"
    LOADING = "loading"


@dataclass
class InboundTruckAgent:
    """
    Autonomous agent representing an inbound truck.

    The agent manages its own lifecycle from arrival through unloading to departure.
    """
    truck: InboundTruck
    env: simpy.Environment
    terminal_coordinator: 'TerminalCoordinator'

    state: TruckState = TruckState.EN_ROUTE
    door_assigned: Optional[int] = None
    pallets_to_unload: List[int] = None
    arrival_actual: float = 0.0
    departure_time: float = 0.0

    def run(self):
        """Main process for inbound truck agent."""
        # Wait until scheduled arrival
        yield self.env.timeout(self.truck.arrival_time)
        self.arrival_actual = self.env.now
        self.state = TruckState.WAITING_FOR_DOOR

        logger.debug(f"Inbound truck {self.truck.truck_id} arrived at {self.env.now:.1f}")

        # Request door assignment from coordinator
        self.state = TruckState.WAITING_FOR_DOOR
        door = yield self.env.process(
            self.terminal_coordinator.request_inbound_door(self)
        )
        self.door_assigned = door
        self.state = TruckState.AT_DOOR

        logger.debug(f"Inbound truck {self.truck.truck_id} assigned to door {door}")

        # Positioning time
        yield self.env.timeout(5)  # 5 minutes to position

        # Unload pallets (coordinator manages forklift assignment)
        self.state = TruckState.UNLOADING
        yield self.env.process(
            self.terminal_coordinator.unload_truck(self)
        )

        # Departure
        self.state = TruckState.DEPARTING
        yield self.env.timeout(5)  # 5 minutes to undock

        self.departure_time = self.env.now
        self.state = TruckState.COMPLETED

        # Release door
        self.terminal_coordinator.release_inbound_door(door)

        logger.debug(f"Inbound truck {self.truck.truck_id} departed at {self.env.now:.1f}")


@dataclass
class OutboundTruckAgent:
    """
    Autonomous agent representing an outbound truck.

    Manages loading of assigned pallets and departure.
    """
    truck: OutboundTruck
    env: simpy.Environment
    terminal_coordinator: 'TerminalCoordinator'
    assigned_pallets: List[int] = None

    state: TruckState = TruckState.EN_ROUTE
    door_assigned: Optional[int] = None
    pallets_loaded: List[int] = None
    arrival_actual: float = 0.0
    departure_time: float = 0.0
    is_late: bool = False

    def __post_init__(self):
        if self.pallets_loaded is None:
            self.pallets_loaded = []

    def run(self):
        """Main process for outbound truck agent."""
        # Wait until scheduled arrival
        yield self.env.timeout(self.truck.arrival_time)
        self.arrival_actual = self.env.now
        self.state = TruckState.WAITING_FOR_DOOR

        logger.debug(f"Outbound truck {self.truck.truck_id} arrived at {self.env.now:.1f}")

        # Request door assignment
        door = yield self.env.process(
            self.terminal_coordinator.request_outbound_door(self)
        )
        self.door_assigned = door
        self.state = TruckState.AT_DOOR

        logger.debug(f"Outbound truck {self.truck.truck_id} assigned to door {door}")

        # Positioning time
        yield self.env.timeout(5)

        # Load assigned pallets
        self.state = TruckState.LOADING
        if self.assigned_pallets:
            yield self.env.process(
                self.terminal_coordinator.load_truck(self)
            )

        # Departure
        self.state = TruckState.DEPARTING
        yield self.env.timeout(5)

        self.departure_time = self.env.now
        self.is_late = self.departure_time > self.truck.due_date
        self.state = TruckState.COMPLETED

        # Release door
        self.terminal_coordinator.release_outbound_door(door)

        logger.debug(f"Outbound truck {self.truck.truck_id} departed at {self.env.now:.1f}, "
                    f"late={self.is_late}")


@dataclass
class ForkliftAgent:
    """
    Autonomous forklift agent that handles pallet movement.

    Can be assigned tasks by the terminal coordinator.
    """
    forklift_id: int
    env: simpy.Environment
    resource: simpy.Resource

    state: ForkliftState = ForkliftState.IDLE
    current_task: Optional[Dict] = None
    total_moves: int = 0
    total_busy_time: float = 0.0

    def execute_task(self, task: Dict):
        """
        Execute a pallet movement task.

        Args:
            task: Dictionary with 'type' ('unload' or 'load'), 'pallet_id', and times
        """
        start_time = self.env.now
        self.current_task = task

        if task['type'] == 'unload':
            self.state = ForkliftState.MOVING_TO_INBOUND
            yield self.env.timeout(1)  # Move to inbound truck

            self.state = ForkliftState.UNLOADING
            yield self.env.timeout(task.get('unload_time', 2))

            self.state = ForkliftState.MOVING_TO_STAGING
            yield self.env.timeout(task.get('transfer_time', 1.5))

        elif task['type'] == 'load':
            self.state = ForkliftState.MOVING_TO_STAGING
            yield self.env.timeout(1)  # Move to staging

            self.state = ForkliftState.MOVING_TO_OUTBOUND
            yield self.env.timeout(task.get('transfer_time', 1.5))

            self.state = ForkliftState.LOADING
            yield self.env.timeout(task.get('load_time', 2))

        self.state = ForkliftState.IDLE
        self.current_task = None
        self.total_moves += 1
        self.total_busy_time += (self.env.now - start_time)

        logger.debug(f"Forklift {self.forklift_id} completed {task['type']} of pallet {task['pallet_id']}")


class TerminalCoordinator:
    """
    Central coordinator agent that manages terminal operations.

    Responsibilities:
    - Door assignment and scheduling
    - Forklift task assignment
    - Staging area management
    - Performance monitoring
    """

    def __init__(
        self,
        env: simpy.Environment,
        num_inbound_doors: int = 1,
        num_outbound_doors: int = 1,
        num_forklifts: int = 15
    ):
        """Initialize terminal coordinator."""
        self.env = env

        # Resources
        self.inbound_doors = simpy.Resource(env, capacity=num_inbound_doors)
        self.outbound_doors = simpy.Resource(env, capacity=num_outbound_doors)

        # Create forklift agents
        self.forklifts = []
        for i in range(num_forklifts):
            forklift = ForkliftAgent(
                forklift_id=i + 1,
                env=env,
                resource=simpy.Resource(env, capacity=1)
            )
            self.forklifts.append(forklift)

        # Staging area
        self.staging_area = {}  # pallet_id -> arrival_time

        # Statistics
        self.stats = {
            'inbound_trucks_processed': 0,
            'outbound_trucks_processed': 0,
            'pallets_staged': 0,
            'pallets_loaded': 0,
            'total_door_wait_time': 0.0
        }

        logger.info(f"Terminal coordinator initialized: {num_inbound_doors} inbound doors, "
                   f"{num_outbound_doors} outbound doors, {num_forklifts} forklifts")

    def request_inbound_door(self, truck_agent: InboundTruckAgent):
        """Request assignment of inbound door to truck."""
        wait_start = self.env.now

        with self.inbound_doors.request() as request:
            yield request

            wait_time = self.env.now - wait_start
            self.stats['total_door_wait_time'] += wait_time

            logger.debug(f"Inbound truck {truck_agent.truck.truck_id} got door "
                        f"(waited {wait_time:.1f} min)")

            return 1  # Door ID (simplified - only 1 door)

    def release_inbound_door(self, door_id: int):
        """Release inbound door."""
        self.stats['inbound_trucks_processed'] += 1
        logger.debug(f"Inbound door {door_id} released")

    def request_outbound_door(self, truck_agent: OutboundTruckAgent):
        """Request assignment of outbound door to truck."""
        wait_start = self.env.now

        with self.outbound_doors.request() as request:
            yield request

            wait_time = self.env.now - wait_start
            self.stats['total_door_wait_time'] += wait_time

            logger.debug(f"Outbound truck {truck_agent.truck.truck_id} got door "
                        f"(waited {wait_time:.1f} min)")

            return 1  # Door ID

    def release_outbound_door(self, door_id: int):
        """Release outbound door."""
        self.stats['outbound_trucks_processed'] += 1
        logger.debug(f"Outbound door {door_id} released")

    def unload_truck(self, truck_agent: InboundTruckAgent):
        """Coordinate unloading of inbound truck using forklifts."""
        if not truck_agent.pallets_to_unload:
            return

        for pallet_id in truck_agent.pallets_to_unload:
            # Find available forklift (simple round-robin)
            forklift = self._get_available_forklift()

            with forklift.resource.request() as request:
                yield request

                task = {
                    'type': 'unload',
                    'pallet_id': pallet_id,
                    'truck_id': truck_agent.truck.truck_id,
                    'unload_time': 2.0,
                    'transfer_time': 1.5
                }

                yield self.env.process(forklift.execute_task(task))

                # Add to staging area
                self.staging_area[pallet_id] = self.env.now
                self.stats['pallets_staged'] += 1

    def load_truck(self, truck_agent: OutboundTruckAgent):
        """Coordinate loading of outbound truck."""
        if not truck_agent.assigned_pallets:
            return

        for pallet_id in truck_agent.assigned_pallets:
            # Wait until pallet is in staging
            while pallet_id not in self.staging_area:
                yield self.env.timeout(1)

            # Find available forklift
            forklift = self._get_available_forklift()

            with forklift.resource.request() as request:
                yield request

                task = {
                    'type': 'load',
                    'pallet_id': pallet_id,
                    'truck_id': truck_agent.truck.truck_id,
                    'load_time': 2.0,
                    'transfer_time': 1.5
                }

                yield self.env.process(forklift.execute_task(task))

                # Remove from staging
                del self.staging_area[pallet_id]
                truck_agent.pallets_loaded.append(pallet_id)
                self.stats['pallets_loaded'] += 1

    def _get_available_forklift(self) -> ForkliftAgent:
        """Get available forklift (simple selection - first idle or least busy)."""
        # Prefer idle forklifts
        for forklift in self.forklifts:
            if forklift.state == ForkliftState.IDLE:
                return forklift

        # Otherwise, return least busy
        return min(self.forklifts, key=lambda f: f.total_moves)

    def get_statistics(self) -> Dict:
        """Get coordinator statistics."""
        total_busy_time = sum(f.total_busy_time for f in self.forklifts)
        total_moves = sum(f.total_moves for f in self.forklifts)

        return {
            **self.stats,
            'total_forklift_moves': total_moves,
            'total_forklift_busy_time': total_busy_time,
            'avg_forklift_utilization': total_busy_time / (len(self.forklifts) * self.env.now) if self.env.now > 0 else 0,
            'current_staging_inventory': len(self.staging_area)
        }


# Testing
if __name__ == "__main__":
    print("Multi-agent components module ready!")
    print("Use in conjunction with cross_dock_sim.py for full simulation")
