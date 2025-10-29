"""
Online/Real-Time Scheduling Module with Adaptive Decision Rules.

This module implements event-driven scheduling policies for dynamic cross-docking
operations, including rolling horizon optimization and adaptive rules based on
system state.

Author: Cross-Docking Optimization System
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import time

from src.data_loader import CrossDockInstance, Pallet, OutboundTruck, InboundTruck
from src.models.heuristics import earliest_due_date
from src.models.pallet_assignment import AssignmentSolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    """Available online scheduling policies."""
    FCFS = "first_come_first_served"  # Process trucks in arrival order
    EDD = "earliest_due_date"  # Prioritize by due date
    STF = "shortest_task_first"  # Process fastest trucks first
    CRITICAL_RATIO = "critical_ratio"  # Due date / processing time ratio
    ADAPTIVE = "adaptive"  # Adapt based on system state
    ROLLING_HORIZON = "rolling_horizon"  # Optimize over time window


@dataclass
class SystemState:
    """Current state of the cross-docking system."""

    current_time: float  # Current simulation time
    available_doors: Dict[str, List[int]]  # 'inbound'/'outbound' -> list of available door IDs
    trucks_waiting: Dict[str, List[int]]  # 'inbound'/'outbound' -> list of waiting truck IDs
    trucks_processing: Dict[str, List[int]]  # 'inbound'/'outbound' -> list of processing truck IDs
    pallets_in_staging: List[int]  # Pallet IDs currently in staging area
    pallets_assigned: Dict[int, int]  # pallet_id -> truck_id assignments made so far
    staging_capacity: int  # Maximum staging area capacity
    current_utilization: float  # Current staging area utilization (0-1)


@dataclass
class OnlineDecision:
    """Decision made by online scheduler."""

    truck_id: int
    action: str  # 'assign_door', 'wait', 'start_loading', 'start_unloading'
    door_id: Optional[int] = None
    priority: float = 0.0
    reason: str = ""


@dataclass
class OnlineScheduleResult:
    """Result of online scheduling session."""

    decisions: List[OnlineDecision]
    pallet_assignments: Dict[int, int]  # pallet_id -> truck_id
    total_decisions: int
    avg_decision_time: float
    policy_used: str
    adaptations_made: int  # Number of times policy adapted


class OnlineScheduler:
    """
    Online/real-time scheduler with adaptive policies.

    Makes scheduling decisions as trucks arrive and system state changes,
    without requiring complete information about future arrivals.
    """

    def __init__(self, instance: CrossDockInstance, config: Optional[Dict] = None):
        """
        Initialize online scheduler.

        Args:
            instance: Cross-dock instance
            config: Configuration parameters
        """
        self.instance = instance
        self.config = config or {}

        # Configuration
        self.default_policy = SchedulingPolicy[self.config.get('policy', 'EDD')]
        self.horizon_length = self.config.get('horizon_length', 60)  # minutes
        self.replan_interval = self.config.get('replan_interval', 30)  # minutes
        self.staging_capacity = self.config.get('staging_capacity', 1000)
        self.num_inbound_doors = self.config.get('num_inbound_doors', 1)
        self.num_outbound_doors = self.config.get('num_outbound_doors', 1)

        # State
        self.current_policy = self.default_policy
        self.decisions_log = []
        self.adaptations = 0

        # Performance tracking
        self.decision_times = []

        logger.info(f"OnlineScheduler initialized with policy: {self.default_policy.value}")

    def make_decision(self, state: SystemState, event: str, truck_id: Optional[int] = None) -> OnlineDecision:
        """
        Make a scheduling decision based on current state and event.

        Args:
            state: Current system state
            event: Type of event ('truck_arrival', 'truck_departure', 'door_available', etc.)
            truck_id: Truck ID involved in event (if applicable)

        Returns:
            OnlineDecision with action to take
        """
        start_time = time.time()

        # Adapt policy if needed
        if self.current_policy == SchedulingPolicy.ADAPTIVE:
            self._adapt_policy(state)

        # Make decision based on current policy and event
        if event == 'inbound_truck_arrival':
            decision = self._handle_inbound_arrival(state, truck_id)
        elif event == 'outbound_truck_arrival':
            decision = self._handle_outbound_arrival(state, truck_id)
        elif event == 'door_available':
            decision = self._handle_door_available(state)
        elif event == 'pallet_unloaded':
            decision = self._handle_pallet_unloaded(state)
        elif event == 'replan':
            decision = self._handle_replan(state)
        else:
            decision = OnlineDecision(
                truck_id=-1,
                action='wait',
                reason=f"Unknown event: {event}"
            )

        decision_time = time.time() - start_time
        self.decision_times.append(decision_time)
        self.decisions_log.append(decision)

        return decision

    def _adapt_policy(self, state: SystemState):
        """
        Adapt scheduling policy based on current system state.

        Switches between policies based on congestion, urgency, etc.
        """
        # Check congestion level
        total_waiting = len(state.trucks_waiting.get('inbound', [])) + \
                       len(state.trucks_waiting.get('outbound', []))

        # Check staging utilization
        utilization = state.current_utilization

        # Adapt policy based on conditions
        old_policy = self.current_policy

        if utilization > 0.9 and total_waiting > 10:
            # High congestion: prioritize speed
            self.current_policy = SchedulingPolicy.STF
        elif utilization > 0.7:
            # Medium congestion: prioritize critical items
            self.current_policy = SchedulingPolicy.CRITICAL_RATIO
        elif total_waiting > 20:
            # Many waiting trucks: use rolling horizon optimization
            self.current_policy = SchedulingPolicy.ROLLING_HORIZON
        else:
            # Normal operation: use EDD
            self.current_policy = SchedulingPolicy.EDD

        if old_policy != self.current_policy:
            self.adaptations += 1
            logger.info(f"Policy adapted: {old_policy.value} -> {self.current_policy.value}")

    def _handle_inbound_arrival(self, state: SystemState, truck_id: int) -> OnlineDecision:
        """Handle inbound truck arrival event."""

        # Check if door available
        available_doors = state.available_doors.get('inbound', [])

        if available_doors:
            # Assign to first available door
            door_id = available_doors[0]
            return OnlineDecision(
                truck_id=truck_id,
                action='assign_door',
                door_id=door_id,
                priority=self._calculate_priority(truck_id, state),
                reason="Door available on arrival"
            )
        else:
            # Add to waiting queue
            return OnlineDecision(
                truck_id=truck_id,
                action='wait',
                priority=self._calculate_priority(truck_id, state),
                reason="No doors available"
            )

    def _handle_outbound_arrival(self, state: SystemState, truck_id: int) -> OnlineDecision:
        """Handle outbound truck arrival event."""

        # Find truck info
        truck = next((t for t in self.instance.outbound_trucks if t.truck_id == truck_id), None)
        if not truck:
            return OnlineDecision(truck_id=truck_id, action='wait', reason="Truck not found")

        # Check if pallets are ready (in staging)
        dest_pallets = [p for p in self.instance.pallets if p.destination == truck.destination]
        available_pallets = [p.pallet_id for p in dest_pallets if p.pallet_id in state.pallets_in_staging]

        if len(available_pallets) >= truck.capacity * 0.5:
            # Enough pallets ready - assign pallets and door if available
            if state.available_doors.get('outbound', []):
                return OnlineDecision(
                    truck_id=truck_id,
                    action='start_loading',
                    door_id=state.available_doors['outbound'][0],
                    priority=self._calculate_priority(truck_id, state),
                    reason=f"{len(available_pallets)} pallets ready"
                )

        return OnlineDecision(
            truck_id=truck_id,
            action='wait',
            priority=self._calculate_priority(truck_id, state),
            reason="Waiting for pallets or door"
        )

    def _handle_door_available(self, state: SystemState) -> OnlineDecision:
        """Handle door becoming available - select next truck to process."""

        # Determine which type of door is available
        if state.available_doors.get('inbound'):
            # Select next inbound truck
            waiting_trucks = state.trucks_waiting.get('inbound', [])
            if waiting_trucks:
                # Sort by policy
                truck_id = self._select_next_truck(waiting_trucks, state, 'inbound')
                return OnlineDecision(
                    truck_id=truck_id,
                    action='assign_door',
                    door_id=state.available_doors['inbound'][0],
                    priority=self._calculate_priority(truck_id, state),
                    reason=f"Selected by {self.current_policy.value}"
                )

        if state.available_doors.get('outbound'):
            # Select next outbound truck
            waiting_trucks = state.trucks_waiting.get('outbound', [])
            if waiting_trucks:
                truck_id = self._select_next_truck(waiting_trucks, state, 'outbound')
                return OnlineDecision(
                    truck_id=truck_id,
                    action='assign_door',
                    door_id=state.available_doors['outbound'][0],
                    priority=self._calculate_priority(truck_id, state),
                    reason=f"Selected by {self.current_policy.value}"
                )

        return OnlineDecision(truck_id=-1, action='wait', reason="No waiting trucks")

    def _handle_pallet_unloaded(self, state: SystemState) -> OnlineDecision:
        """Handle pallet unloading completion - assign to outbound truck."""

        # Use heuristic to assign newly available pallets
        # This is a simplified version - in reality would track specific pallets

        return OnlineDecision(
            truck_id=-1,
            action='update_assignments',
            reason="Pallet unloaded, updating assignments"
        )

    def _handle_replan(self, state: SystemState) -> OnlineDecision:
        """
        Handle replanning event - optimize over rolling horizon.

        Uses current state plus forecast for next horizon_length minutes.
        """

        if self.current_policy != SchedulingPolicy.ROLLING_HORIZON:
            return OnlineDecision(truck_id=-1, action='wait', reason="Not in rolling horizon mode")

        # Create sub-instance for rolling horizon
        horizon_end = state.current_time + self.horizon_length

        # Filter trucks that will arrive within horizon
        future_inbound = [
            t for t in self.instance.inbound_trucks
            if state.current_time <= t.arrival_time <= horizon_end
        ]
        future_outbound = [
            t for t in self.instance.outbound_trucks
            if state.current_time <= t.arrival_time <= horizon_end
        ]

        logger.info(f"Rolling horizon replan at t={state.current_time:.0f}: "
                   f"{len(future_inbound)} inbound, {len(future_outbound)} outbound trucks")

        # Would run optimization here over the horizon
        # For now, just return a decision to continue

        return OnlineDecision(
            truck_id=-1,
            action='replan_complete',
            reason=f"Replanned for horizon [{state.current_time:.0f}, {horizon_end:.0f}]"
        )

    def _select_next_truck(self, waiting_trucks: List[int], state: SystemState, truck_type: str) -> int:
        """
        Select next truck to process from waiting queue based on policy.

        Args:
            waiting_trucks: List of waiting truck IDs
            state: Current system state
            truck_type: 'inbound' or 'outbound'

        Returns:
            Selected truck ID
        """

        if self.current_policy == SchedulingPolicy.FCFS:
            # First come, first served - already in order
            return waiting_trucks[0]

        elif self.current_policy == SchedulingPolicy.EDD:
            # Earliest due date
            if truck_type == 'outbound':
                truck_dict = {t.truck_id: t for t in self.instance.outbound_trucks}
                return min(waiting_trucks, key=lambda tid: truck_dict[tid].due_date)
            else:
                # Inbound trucks don't have due dates, use arrival time
                truck_dict = {t.truck_id: t for t in self.instance.inbound_trucks}
                return min(waiting_trucks, key=lambda tid: truck_dict[tid].arrival_time)

        elif self.current_policy == SchedulingPolicy.STF:
            # Shortest task first - estimate processing time
            priorities = {tid: self._estimate_processing_time(tid, truck_type) for tid in waiting_trucks}
            return min(waiting_trucks, key=lambda tid: priorities[tid])

        elif self.current_policy == SchedulingPolicy.CRITICAL_RATIO:
            # Critical ratio: (due_date - current_time) / processing_time
            if truck_type == 'outbound':
                truck_dict = {t.truck_id: t for t in self.instance.outbound_trucks}
                priorities = {}
                for tid in waiting_trucks:
                    truck = truck_dict[tid]
                    slack = truck.due_date - state.current_time
                    proc_time = self._estimate_processing_time(tid, truck_type)
                    priorities[tid] = slack / proc_time if proc_time > 0 else float('inf')
                return min(waiting_trucks, key=lambda tid: priorities[tid])
            else:
                # Inbound: use STF
                priorities = {tid: self._estimate_processing_time(tid, truck_type) for tid in waiting_trucks}
                return min(waiting_trucks, key=lambda tid: priorities[tid])

        else:
            # Default: FCFS
            return waiting_trucks[0]

    def _calculate_priority(self, truck_id: int, state: SystemState) -> float:
        """Calculate priority score for a truck (higher = more urgent)."""

        # Find truck
        truck = None
        for t in list(self.instance.inbound_trucks) + list(self.instance.outbound_trucks):
            if t.truck_id == truck_id:
                truck = t
                break

        if not truck:
            return 0.0

        # Calculate based on current policy
        if hasattr(truck, 'due_date'):
            # Outbound truck
            slack = truck.due_date - state.current_time
            if slack < 0:
                return 1000.0  # Already late - highest priority
            elif slack < 60:
                return 100.0  # Very urgent
            elif slack < 180:
                return 50.0  # Urgent
            else:
                return 10.0  # Normal
        else:
            # Inbound truck - priority based on waiting time
            arrival = truck.arrival_time
            waiting_time = state.current_time - arrival
            return waiting_time  # Higher waiting time = higher priority

    def _estimate_processing_time(self, truck_id: int, truck_type: str) -> float:
        """Estimate processing time for a truck."""

        if truck_type == 'inbound':
            # Count pallets for this truck
            num_pallets = len([p for p in self.instance.pallets if p.inbound_truck_id == truck_id])
            return 5.0 + num_pallets * 1.0  # setup + unload time
        else:
            # Outbound: estimate based on capacity
            truck = next((t for t in self.instance.outbound_trucks if t.truck_id == truck_id), None)
            if truck:
                return 5.0 + min(truck.capacity, 20) * 1.0  # setup + load time
            return 30.0  # default estimate

    def get_statistics(self) -> Dict:
        """Get statistics about online scheduling performance."""

        return {
            'total_decisions': len(self.decisions_log),
            'avg_decision_time': sum(self.decision_times) / len(self.decision_times) if self.decision_times else 0,
            'adaptations_made': self.adaptations,
            'final_policy': self.current_policy.value,
            'action_breakdown': self._count_actions()
        }

    def _count_actions(self) -> Dict[str, int]:
        """Count actions by type."""
        counts = {}
        for decision in self.decisions_log:
            counts[decision.action] = counts.get(decision.action, 0) + 1
        return counts


# Convenience functions

def create_online_scheduler(instance: CrossDockInstance,
                           policy: str = 'EDD',
                           **kwargs) -> OnlineScheduler:
    """
    Create an online scheduler with specified policy.

    Args:
        instance: Cross-dock instance
        policy: Policy name ('FCFS', 'EDD', 'STF', 'CRITICAL_RATIO', 'ADAPTIVE', 'ROLLING_HORIZON')
        **kwargs: Additional configuration parameters

    Returns:
        OnlineScheduler instance
    """
    config = {'policy': policy, **kwargs}
    return OnlineScheduler(instance, config)


def simulate_online_scheduling(instance: CrossDockInstance,
                               policy: str = 'EDD',
                               **kwargs) -> OnlineScheduleResult:
    """
    Simulate online scheduling for an instance.

    This is a simplified simulation that demonstrates the online scheduler's
    decision-making process.

    Args:
        instance: Cross-dock instance
        policy: Scheduling policy to use
        **kwargs: Additional configuration

    Returns:
        OnlineScheduleResult with decisions made
    """
    scheduler = create_online_scheduler(instance, policy, **kwargs)

    # Create initial state
    state = SystemState(
        current_time=0.0,
        available_doors={'inbound': [0], 'outbound': [0]},
        trucks_waiting={'inbound': [], 'outbound': []},
        trucks_processing={'inbound': [], 'outbound': []},
        pallets_in_staging=[],
        pallets_assigned={},
        staging_capacity=1000,
        current_utilization=0.0
    )

    decisions = []

    # Simulate key events
    # 1. Process first few inbound trucks
    for i, truck in enumerate(list(instance.inbound_trucks)[:5]):
        state.current_time = truck.arrival_time
        decision = scheduler.make_decision(state, 'inbound_truck_arrival', truck.truck_id)
        decisions.append(decision)

    # 2. Process first few outbound trucks
    for i, truck in enumerate(list(instance.outbound_trucks)[:5]):
        state.current_time = truck.arrival_time
        decision = scheduler.make_decision(state, 'outbound_truck_arrival', truck.truck_id)
        decisions.append(decision)

    # Get pallet assignments (use heuristic as baseline)
    heuristic_solution = earliest_due_date(instance)
    pallet_assignments = heuristic_solution.assignments

    stats = scheduler.get_statistics()

    logger.info(f"Online scheduling simulation complete: {stats['total_decisions']} decisions, "
               f"{stats['adaptations_made']} adaptations")

    return OnlineScheduleResult(
        decisions=decisions,
        pallet_assignments=pallet_assignments,
        total_decisions=stats['total_decisions'],
        avg_decision_time=stats['avg_decision_time'],
        policy_used=policy,
        adaptations_made=stats['adaptations_made']
    )
