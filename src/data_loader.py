"""
Data Loading Module for Cross-Docking Terminal Optimization

This module provides utilities for loading and validating the cross-docking
simulation dataset. It handles Excel file reading and creates structured
data objects for use in optimization and simulation models.

Author: Cross-Docking Optimization Project
Based on: Torbali (2023) - Real-Time Truck Scheduling in Cross-Docking
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InboundTruck:
    """Represents an inbound truck delivering pallets to the cross-dock."""
    truck_id: int
    arrival_time: float  # minutes from start of simulation

    def __repr__(self) -> str:
        return f"InboundTruck(id={self.truck_id}, arrival={self.arrival_time:.1f}min)"


@dataclass
class OutboundTruck:
    """Represents an outbound truck transporting pallets to destinations."""
    truck_id: int
    arrival_time: float  # minutes from start
    due_date: float  # minutes from start (deadline)
    destination: int  # 1, 2, or 3
    capacity: int = 26  # pallets per truck (from dataset spec)
    assigned_pallets: List[int] = field(default_factory=list)  # pallet IDs

    @property
    def fill_rate(self) -> float:
        """Calculate current fill rate (0 to 1)."""
        return len(self.assigned_pallets) / self.capacity if self.capacity > 0 else 0

    @property
    def is_full(self) -> bool:
        """Check if truck is at capacity."""
        return len(self.assigned_pallets) >= self.capacity

    @property
    def available_capacity(self) -> int:
        """Remaining capacity in pallets."""
        return max(0, self.capacity - len(self.assigned_pallets))

    def __repr__(self) -> str:
        return (f"OutboundTruck(id={self.truck_id}, dest={self.destination}, "
                f"arrival={self.arrival_time:.1f}min, due={self.due_date:.1f}min, "
                f"fill={self.fill_rate:.1%})")


@dataclass
class Pallet:
    """Represents a pallet to be transferred through the cross-dock."""
    pallet_id: int
    due_date: float  # minutes from start (deadline for delivery)
    destination: int  # 1, 2, or 3
    pallet_type: str  # 'A', 'B', or 'C'
    inbound_truck_id: int  # which truck delivered this pallet
    assigned_outbound_truck: Optional[int] = None  # assigned outbound truck ID

    @property
    def is_assigned(self) -> bool:
        """Check if pallet is assigned to an outbound truck."""
        return self.assigned_outbound_truck is not None

    def __repr__(self) -> str:
        return (f"Pallet(id={self.pallet_id}, type={self.pallet_type}, "
                f"dest={self.destination}, due={self.due_date:.1f}min)")


@dataclass
class CrossDockInstance:
    """
    Complete cross-docking instance with all trucks and pallets.

    Attributes:
        instance_name: Name of the instance (e.g., 'HH_168h/instance1')
        inbound_trucks: List of inbound truck objects
        outbound_trucks: List of outbound truck objects
        pallets: List of pallet objects
        metadata: Additional information about the instance
    """
    instance_name: str
    inbound_trucks: List[InboundTruck]
    outbound_trucks: List[OutboundTruck]
    pallets: List[Pallet]
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Calculate instance statistics after initialization."""
        self.metadata.update({
            'num_inbound_trucks': len(self.inbound_trucks),
            'num_outbound_trucks': len(self.outbound_trucks),
            'num_pallets': len(self.pallets),
            'load_timestamp': datetime.now().isoformat()
        })

    @property
    def stats(self) -> Dict:
        """Get comprehensive statistics about the instance."""
        pallet_df = pd.DataFrame([
            {
                'destination': p.destination,
                'type': p.pallet_type,
                'due_date': p.due_date,
                'inbound_truck': p.inbound_truck_id
            }
            for p in self.pallets
        ])

        outbound_df = pd.DataFrame([
            {
                'destination': t.destination,
                'arrival': t.arrival_time,
                'due_date': t.due_date
            }
            for t in self.outbound_trucks
        ])

        return {
            'instance_name': self.instance_name,
            'num_inbound_trucks': len(self.inbound_trucks),
            'num_outbound_trucks': len(self.outbound_trucks),
            'num_pallets': len(self.pallets),
            'pallets_by_destination': pallet_df['destination'].value_counts().to_dict(),
            'pallets_by_type': pallet_df['type'].value_counts().to_dict(),
            'trucks_by_destination': outbound_df['destination'].value_counts().to_dict(),
            'avg_pallet_due_date': pallet_df['due_date'].mean(),
            'avg_outbound_due_date': outbound_df['due_date'].mean(),
            'simulation_horizon': max([t.arrival_time for t in self.inbound_trucks] +
                                     [t.arrival_time for t in self.outbound_trucks]),
        }

    def get_pallets_by_destination(self, destination: int) -> List[Pallet]:
        """Get all pallets for a specific destination."""
        return [p for p in self.pallets if p.destination == destination]

    def get_pallets_by_type(self, pallet_type: str) -> List[Pallet]:
        """Get all pallets of a specific type."""
        return [p for p in self.pallets if p.pallet_type == pallet_type]

    def get_outbound_trucks_by_destination(self, destination: int) -> List[OutboundTruck]:
        """Get all outbound trucks going to a specific destination."""
        return [t for t in self.outbound_trucks if t.destination == destination]

    def __repr__(self) -> str:
        return (f"CrossDockInstance('{self.instance_name}': "
                f"{len(self.inbound_trucks)} inbound, "
                f"{len(self.outbound_trucks)} outbound, "
                f"{len(self.pallets)} pallets)")


class DataLoader:
    """
    Main data loading class for cross-docking instances.

    Handles reading Excel files, validation, and batch processing.
    """

    def __init__(self, dataset_root: str = "data"):
        """
        Initialize the data loader.

        Args:
            dataset_root: Root directory containing the dataset folders
        """
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        self.scenarios = ['HH_168h', 'MH_168h', 'MM_168h', 'LH_168h', 'LM_168h', 'LL_168h']
        logger.info(f"DataLoader initialized with root: {self.dataset_root}")

    def load_instance(self, scenario: str, instance_num: int) -> CrossDockInstance:
        """
        Load a single instance from the dataset.

        Args:
            scenario: Scenario name (e.g., 'HH_168h', 'MM_168h')
            instance_num: Instance number (1-10)

        Returns:
            CrossDockInstance object with all data

        Raises:
            FileNotFoundError: If the instance file doesn't exist
            ValueError: If the data is invalid
        """
        if scenario not in self.scenarios:
            raise ValueError(f"Invalid scenario '{scenario}'. Must be one of {self.scenarios}")

        if not 1 <= instance_num <= 10:
            raise ValueError(f"Invalid instance number {instance_num}. Must be between 1 and 10")

        # Construct file path
        file_path = self.dataset_root / scenario / f"instance{instance_num}.xlsx"

        if not file_path.exists():
            raise FileNotFoundError(f"Instance file not found: {file_path}")

        logger.info(f"Loading instance: {scenario}/instance{instance_num}")

        # Read all three sheets
        try:
            inbound_df = pd.read_excel(file_path, sheet_name='inboundTrucks')
            outbound_df = pd.read_excel(file_path, sheet_name='outboundTrucks')
            pallets_df = pd.read_excel(file_path, sheet_name='pallets')

            # Strip whitespace from column names
            inbound_df.columns = inbound_df.columns.str.strip()
            outbound_df.columns = outbound_df.columns.str.strip()
            pallets_df.columns = pallets_df.columns.str.strip()

            # Drop rows with missing critical values and log warnings
            initial_counts = {
                'inbound': len(inbound_df),
                'outbound': len(outbound_df),
                'pallets': len(pallets_df)
            }

            inbound_df = inbound_df.dropna()
            outbound_df = outbound_df.dropna()
            pallets_df = pallets_df.dropna()

            # Log if any rows were dropped
            if len(inbound_df) < initial_counts['inbound']:
                logger.warning(f"Dropped {initial_counts['inbound'] - len(inbound_df)} "
                             f"inbound trucks with missing values")
            if len(outbound_df) < initial_counts['outbound']:
                logger.warning(f"Dropped {initial_counts['outbound'] - len(outbound_df)} "
                             f"outbound trucks with missing values")
            if len(pallets_df) < initial_counts['pallets']:
                logger.warning(f"Dropped {initial_counts['pallets'] - len(pallets_df)} "
                             f"pallets with missing values")

        except Exception as e:
            raise ValueError(f"Error reading Excel file {file_path}: {str(e)}")

        # Validate data
        self._validate_data(inbound_df, outbound_df, pallets_df)

        # Create inbound truck objects
        inbound_trucks = [
            InboundTruck(
                truck_id=int(row['Truck ID']),
                arrival_time=float(row['Truck arrival time (min)'])
            )
            for _, row in inbound_df.iterrows()
        ]

        # Create outbound truck objects
        outbound_trucks = [
            OutboundTruck(
                truck_id=int(row['Truck ID']),
                arrival_time=float(row['Arrival time (min)']),
                due_date=float(row['Due date (min)']),
                destination=int(row['Destination'])
            )
            for _, row in outbound_df.iterrows()
        ]

        # Create pallet objects
        pallets = [
            Pallet(
                pallet_id=int(row['Pallet ID']),
                due_date=float(row['Due date (min)']),
                destination=int(row['Destination']),
                pallet_type=str(row['Type']),
                inbound_truck_id=int(row['TruckId'])
            )
            for _, row in pallets_df.iterrows()
        ]

        # Create instance object
        instance = CrossDockInstance(
            instance_name=f"{scenario}/instance{instance_num}",
            inbound_trucks=inbound_trucks,
            outbound_trucks=outbound_trucks,
            pallets=pallets,
            metadata={
                'scenario': scenario,
                'instance_number': instance_num,
                'file_path': str(file_path)
            }
        )

        logger.info(f"Successfully loaded: {instance}")
        return instance

    def load_all_instances(self, scenario: str) -> List[CrossDockInstance]:
        """
        Load all 10 instances for a given scenario.

        Args:
            scenario: Scenario name (e.g., 'MM_168h')

        Returns:
            List of CrossDockInstance objects
        """
        instances = []
        for i in range(1, 11):
            try:
                instance = self.load_instance(scenario, i)
                instances.append(instance)
            except FileNotFoundError as e:
                logger.warning(f"Skipping instance {i}: {str(e)}")

        logger.info(f"Loaded {len(instances)} instances for scenario {scenario}")
        return instances

    def load_all_scenarios(self) -> Dict[str, List[CrossDockInstance]]:
        """
        Load all instances from all scenarios.

        Returns:
            Dictionary mapping scenario names to lists of instances
        """
        all_data = {}
        for scenario in self.scenarios:
            all_data[scenario] = self.load_all_instances(scenario)

        total_instances = sum(len(instances) for instances in all_data.values())
        logger.info(f"Loaded {total_instances} total instances across {len(all_data)} scenarios")
        return all_data

    def _validate_data(self, inbound_df: pd.DataFrame,
                      outbound_df: pd.DataFrame,
                      pallets_df: pd.DataFrame) -> None:
        """
        Validate the loaded data for consistency and correctness.

        Args:
            inbound_df: Inbound trucks dataframe
            outbound_df: Outbound trucks dataframe
            pallets_df: Pallets dataframe

        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        required_inbound_cols = {'Truck ID', 'Truck arrival time (min)'}
        required_outbound_cols = {'Truck ID', 'Arrival time (min)', 'Due date (min)', 'Destination'}
        required_pallet_cols = {'Pallet ID', 'Due date (min)', 'Destination', 'Type', 'TruckId'}

        if not required_inbound_cols.issubset(inbound_df.columns):
            raise ValueError(f"Inbound trucks missing columns: {required_inbound_cols - set(inbound_df.columns)}")

        if not required_outbound_cols.issubset(outbound_df.columns):
            raise ValueError(f"Outbound trucks missing columns: {required_outbound_cols - set(outbound_df.columns)}")

        if not required_pallet_cols.issubset(pallets_df.columns):
            raise ValueError(f"Pallets missing columns: {required_pallet_cols - set(pallets_df.columns)}")

        # Note: Missing values are already handled by dropna() before validation
        # Validate destinations (should be 1, 2, or 3)
        valid_destinations = {1, 2, 3}
        if not set(outbound_df['Destination'].unique()).issubset(valid_destinations):
            raise ValueError(f"Invalid destinations in outbound trucks: {outbound_df['Destination'].unique()}")

        if not set(pallets_df['Destination'].unique()).issubset(valid_destinations):
            raise ValueError(f"Invalid destinations in pallets: {pallets_df['Destination'].unique()}")

        # Validate pallet types (should be 'A', 'B', or 'C')
        valid_types = {'A', 'B', 'C'}
        if not set(pallets_df['Type'].unique()).issubset(valid_types):
            raise ValueError(f"Invalid pallet types: {pallets_df['Type'].unique()}")

        # Check that pallet inbound truck IDs exist
        inbound_truck_ids = set(inbound_df['Truck ID'])
        pallet_truck_ids = set(pallets_df['TruckId'])
        if not pallet_truck_ids.issubset(inbound_truck_ids):
            missing = pallet_truck_ids - inbound_truck_ids
            raise ValueError(f"Pallets reference non-existent inbound truck IDs: {missing}")

        # Check for negative or invalid times
        if (inbound_df['Truck arrival time (min)'] < 0).any():
            raise ValueError("Negative arrival times in inbound trucks")

        if (outbound_df['Arrival time (min)'] < 0).any():
            raise ValueError("Negative arrival times in outbound trucks")

        if (outbound_df['Due date (min)'] < 0).any():
            raise ValueError("Negative due dates in outbound trucks")

        if (pallets_df['Due date (min)'] < 0).any():
            raise ValueError("Negative due dates in pallets")

        logger.debug("Data validation passed")

    def get_instance_summary(self, scenario: str = None) -> pd.DataFrame:
        """
        Get a summary DataFrame of all instances.

        Args:
            scenario: Optional scenario name to filter by

        Returns:
            DataFrame with summary statistics for each instance
        """
        scenarios_to_load = [scenario] if scenario else self.scenarios

        summaries = []
        for scen in scenarios_to_load:
            for i in range(1, 11):
                try:
                    instance = self.load_instance(scen, i)
                    stats = instance.stats
                    summaries.append(stats)
                except FileNotFoundError:
                    continue

        return pd.DataFrame(summaries)


def load_config(config_path: str = "config/parameters.yaml") -> Dict:
    """
    Load configuration parameters from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary of configuration parameters
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


# Example usage and testing
if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()

    # Load a single instance
    print("\n Loading Single Instance ")
    instance = loader.load_instance('MM_168h', 1)
    print(instance)
    print("\nInstance Stats:")
    for key, value in instance.stats.items():
        print(f"  {key}: {value}")

    # Load all instances from one scenario
    print("\n Loading All MM_168h Instances ")
    mm_instances = loader.load_all_instances('MM_168h')
    print(f"Loaded {len(mm_instances)} instances")

    # Get summary
    print("\n Instance Summary ")
    summary = loader.get_instance_summary('MM_168h')
    print(summary.head())

    # Load configuration
    print("\n Loading Configuration ")
    config = load_config()
    print(f"Facility has {config['facility']['num_forklifts']} forklifts")
    print(f"Truck capacity: {config['trucks']['outbound_truck_capacity']} pallets")
