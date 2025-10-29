"""
Resource Models for Cross-Docking Simulation

This module provides detailed resource management including:
- Dock door resources with scheduling
- Forklift pool management
- Staging area with capacity constraints
- Queue management systems

Author: Cross-Docking Optimization Project
"""

import simpy
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DoorResource:
    """
    Represents a dock door resource with scheduling capabilities.

    Tracks occupancy, queue, and utilization statistics.
    """
    door_id: int
    door_type: str  # 'inbound' or 'outbound'
    env: simpy.Environment
    resource: simpy.Resource

    # Statistics
    total_busy_time: float = 0.0
    total_trucks_served: int = 0
    queue_wait_times: List[float] = field(default_factory=list)
    current_truck: Optional[int] = None
    occupancy_start: Optional[float] = None

    def request(self, truck_id: int):
        """Request door for a truck."""
        queue_entry_time = self.env.now

        with self.resource.request() as request:
            yield request

            # Calculate wait time
            wait_time = self.env.now - queue_entry_time
            self.queue_wait_times.append(wait_time)

            # Mark as occupied
            self.current_truck = truck_id
            self.occupancy_start = self.env.now

            logger.debug(f"{self.door_type.capitalize()} door {self.door_id} assigned to truck {truck_id} "
                        f"(wait: {wait_time:.1f} min)")

            return self

    def release(self):
        """Release door."""
        if self.occupancy_start is not None:
            busy_time = self.env.now - self.occupancy_start
            self.total_busy_time += busy_time

        self.total_trucks_served += 1
        self.current_truck = None
        self.occupancy_start = None

        logger.debug(f"{self.door_type.capitalize()} door {self.door_id} released")

    @property
    def utilization(self) -> float:
        """Calculate door utilization rate."""
        if self.env.now == 0:
            return 0.0
        return self.total_busy_time / self.env.now

    @property
    def avg_queue_wait(self) -> float:
        """Average queue wait time."""
        if not self.queue_wait_times:
            return 0.0
        return sum(self.queue_wait_times) / len(self.queue_wait_times)

    def get_stats(self) -> Dict:
        """Get door statistics."""
        return {
            'door_id': self.door_id,
            'type': self.door_type,
            'utilization': self.utilization,
            'trucks_served': self.total_trucks_served,
            'avg_wait_time': self.avg_queue_wait,
            'max_wait_time': max(self.queue_wait_times) if self.queue_wait_times else 0,
            'total_busy_time': self.total_busy_time
        }


@dataclass
class ForkliftPool:
    """
    Manages a pool of forklifts as resources.

    Provides intelligent assignment based on workload balancing.
    """
    env: simpy.Environment
    num_forklifts: int

    forklifts: List[simpy.Resource] = field(default_factory=list)
    forklift_stats: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize forklift pool."""
        for i in range(self.num_forklifts):
            forklift = simpy.Resource(self.env, capacity=1)
            self.forklifts.append(forklift)
            self.forklift_stats.append({
                'id': i + 1,
                'total_moves': 0,
                'total_busy_time': 0.0,
                'last_start_time': None
            })

        logger.info(f"Forklift pool initialized with {self.num_forklifts} forklifts")

    def request_forklift(self, strategy: str = 'round_robin') -> Tuple[int, simpy.Request]:
        """
        Request a forklift from the pool.

        Args:
            strategy: 'round_robin', 'least_busy', or 'random'

        Returns:
            Tuple of (forklift_id, request)
        """
        if strategy == 'least_busy':
            forklift_id = self._get_least_busy_forklift()
        else:  # round_robin (default)
            forklift_id = len([s for s in self.forklift_stats if s['last_start_time'] is not None]) % self.num_forklifts

        forklift = self.forklifts[forklift_id]
        request = forklift.request()

        return forklift_id, request

    def start_task(self, forklift_id: int):
        """Mark start of forklift task."""
        self.forklift_stats[forklift_id]['last_start_time'] = self.env.now

    def complete_task(self, forklift_id: int):
        """Mark completion of forklift task."""
        stats = self.forklift_stats[forklift_id]
        if stats['last_start_time'] is not None:
            task_time = self.env.now - stats['last_start_time']
            stats['total_busy_time'] += task_time
            stats['total_moves'] += 1
            stats['last_start_time'] = None

    def _get_least_busy_forklift(self) -> int:
        """Get ID of least busy forklift."""
        return min(range(self.num_forklifts),
                  key=lambda i: self.forklift_stats[i]['total_moves'])

    def get_statistics(self) -> Dict:
        """Get pool statistics."""
        total_moves = sum(s['total_moves'] for s in self.forklift_stats)
        total_busy_time = sum(s['total_busy_time'] for s in self.forklift_stats)

        return {
            'num_forklifts': self.num_forklifts,
            'total_moves': total_moves,
            'avg_moves_per_forklift': total_moves / self.num_forklifts,
            'total_busy_time': total_busy_time,
            'avg_utilization': total_busy_time / (self.num_forklifts * self.env.now) if self.env.now > 0 else 0,
            'individual_stats': self.forklift_stats
        }


@dataclass
class StagingArea:
    """
    Staging area for pallets awaiting outbound loading.

    Tracks inventory levels, dwell times, and capacity constraints.
    """
    env: simpy.Environment
    capacity: int = 10000  # Effectively unlimited

    pallets: Dict[int, float] = field(default_factory=dict)  # pallet_id -> arrival_time
    inventory_samples: List[Tuple[float, int]] = field(default_factory=list)
    sample_interval: float = 10.0  # Sample every 10 minutes

    def __post_init__(self):
        """Start monitoring process."""
        self.env.process(self._monitor_inventory())

    def add_pallet(self, pallet_id: int):
        """Add pallet to staging area."""
        if len(self.pallets) >= self.capacity:
            logger.warning(f"Staging area at capacity! Cannot add pallet {pallet_id}")
            return False

        self.pallets[pallet_id] = self.env.now
        logger.debug(f"Pallet {pallet_id} added to staging (inventory: {len(self.pallets)})")
        return True

    def remove_pallet(self, pallet_id: int) -> Optional[float]:
        """
        Remove pallet from staging area.

        Returns:
            Dwell time (time spent in staging) or None if pallet not found
        """
        if pallet_id not in self.pallets:
            logger.warning(f"Pallet {pallet_id} not in staging area")
            return None

        arrival_time = self.pallets.pop(pallet_id)
        dwell_time = self.env.now - arrival_time

        logger.debug(f"Pallet {pallet_id} removed from staging (dwell: {dwell_time:.1f} min)")
        return dwell_time

    def is_pallet_available(self, pallet_id: int) -> bool:
        """Check if pallet is in staging area."""
        return pallet_id in self.pallets

    def get_dwell_time(self, pallet_id: int) -> Optional[float]:
        """Get current dwell time for pallet."""
        if pallet_id not in self.pallets:
            return None
        return self.env.now - self.pallets[pallet_id]

    def _monitor_inventory(self):
        """Monitor inventory levels periodically."""
        while True:
            self.inventory_samples.append((self.env.now, len(self.pallets)))
            yield self.env.timeout(self.sample_interval)

    @property
    def current_inventory(self) -> int:
        """Current number of pallets in staging."""
        return len(self.pallets)

    @property
    def avg_inventory(self) -> float:
        """Average inventory level."""
        if not self.inventory_samples:
            return 0.0
        return sum(inv for _, inv in self.inventory_samples) / len(self.inventory_samples)

    @property
    def max_inventory(self) -> int:
        """Maximum inventory level observed."""
        if not self.inventory_samples:
            return 0
        return max(inv for _, inv in self.inventory_samples)

    def get_statistics(self) -> Dict:
        """Get staging area statistics."""
        return {
            'capacity': self.capacity,
            'current_inventory': self.current_inventory,
            'avg_inventory': self.avg_inventory,
            'max_inventory': self.max_inventory,
            'utilization': self.current_inventory / self.capacity if self.capacity > 0 else 0,
            'num_samples': len(self.inventory_samples)
        }


@dataclass
class QueueManager:
    """
    Manages queues for different resources.

    Tracks queue lengths, wait times, and provides statistics.
    """
    env: simpy.Environment

    queues: Dict[str, deque] = field(default_factory=dict)
    queue_samples: Dict[str, List[Tuple[float, int]]] = field(default_factory=dict)
    wait_times: Dict[str, List[float]] = field(default_factory=dict)
    sample_interval: float = 5.0

    def __post_init__(self):
        """Start monitoring."""
        self.env.process(self._monitor_queues())

    def register_queue(self, queue_name: str):
        """Register a new queue for monitoring."""
        self.queues[queue_name] = deque()
        self.queue_samples[queue_name] = []
        self.wait_times[queue_name] = []

    def enter_queue(self, queue_name: str, entity_id: int):
        """Record entity entering queue."""
        if queue_name not in self.queues:
            self.register_queue(queue_name)

        self.queues[queue_name].append((entity_id, self.env.now))
        logger.debug(f"Entity {entity_id} entered {queue_name} queue (length: {len(self.queues[queue_name])})")

    def exit_queue(self, queue_name: str, entity_id: int):
        """Record entity exiting queue."""
        if queue_name not in self.queues:
            return

        # Find and remove entity
        for i, (eid, entry_time) in enumerate(self.queues[queue_name]):
            if eid == entity_id:
                wait_time = self.env.now - entry_time
                self.wait_times[queue_name].append(wait_time)
                del self.queues[queue_name][i]
                logger.debug(f"Entity {entity_id} exited {queue_name} queue (wait: {wait_time:.1f} min)")
                return

    def get_queue_length(self, queue_name: str) -> int:
        """Get current queue length."""
        return len(self.queues.get(queue_name, []))

    def _monitor_queues(self):
        """Monitor all queues periodically."""
        while True:
            for queue_name, queue in self.queues.items():
                self.queue_samples[queue_name].append((self.env.now, len(queue)))
            yield self.env.timeout(self.sample_interval)

    def get_statistics(self, queue_name: str) -> Dict:
        """Get statistics for a specific queue."""
        if queue_name not in self.queues:
            return {}

        samples = self.queue_samples[queue_name]
        waits = self.wait_times[queue_name]

        return {
            'queue_name': queue_name,
            'current_length': len(self.queues[queue_name]),
            'avg_length': sum(l for _, l in samples) / len(samples) if samples else 0,
            'max_length': max((l for _, l in samples), default=0),
            'avg_wait_time': sum(waits) / len(waits) if waits else 0,
            'max_wait_time': max(waits, default=0),
            'total_entities_served': len(waits)
        }

    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all queues."""
        return {name: self.get_statistics(name) for name in self.queues.keys()}


# Testing
if __name__ == "__main__":
    import simpy

    print("Testing resource models...")

    env = simpy.Environment()

    # Test door resource
    door = DoorResource(
        door_id=1,
        door_type='inbound',
        env=env,
        resource=simpy.Resource(env, capacity=1)
    )

    # Test forklift pool
    pool = ForkliftPool(env, num_forklifts=15)

    # Test staging area
    staging = StagingArea(env, capacity=1000)

    # Test queue manager
    queue_mgr = QueueManager(env)
    queue_mgr.register_queue('inbound_door')
    queue_mgr.register_queue('outbound_door')

    print("✅ All resource models initialized successfully!")
    print(f"Door utilization: {door.utilization:.2%}")
    print(f"Forklift pool: {pool.num_forklifts} forklifts")
    print(f"Staging capacity: {staging.capacity} pallets")
    print(f"Queue manager tracking {len(queue_mgr.queues)} queues")
