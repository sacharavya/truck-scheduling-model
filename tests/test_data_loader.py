"""
Unit tests for data_loader module.

Tests data loading, validation, and preprocessing functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import (
    DataLoader,
    CrossDockInstance,
    InboundTruck,
    OutboundTruck,
    Pallet
)


class TestDataLoader:
    """Test suite for DataLoader class."""

    @pytest.fixture
    def loader(self):
        """Create DataLoader instance for testing."""
        return DataLoader()

    def test_initialization(self, loader):
        """Test DataLoader initialization."""
        assert loader.dataset_root.exists()
        assert len(loader.scenarios) == 6
        assert 'LL_168h' in loader.scenarios
        assert 'HH_168h' in loader.scenarios

    def test_load_single_instance(self, loader):
        """Test loading a single instance."""
        instance = loader.load_instance('LL_168h', 1)

        assert isinstance(instance, CrossDockInstance)
        assert instance.instance_name == 'LL_168h/instance1'
        assert len(instance.inbound_trucks) > 0
        assert len(instance.outbound_trucks) > 0
        assert len(instance.pallets) > 0

    def test_load_all_instances_in_scenario(self, loader):
        """Test loading all instances in a scenario."""
        instances = loader.load_all_instances('LL_168h')

        assert len(instances) == 10
        assert all(isinstance(inst, CrossDockInstance) for inst in instances)
        assert instances[0].instance_name == 'LL_168h/instance1'
        assert instances[9].instance_name == 'LL_168h/instance10'

    def test_invalid_scenario(self, loader):
        """Test loading with invalid scenario name."""
        with pytest.raises(FileNotFoundError):
            loader.load_instance('INVALID_SCENARIO', 1)

    def test_invalid_instance_number(self, loader):
        """Test loading with invalid instance number."""
        with pytest.raises(FileNotFoundError):
            loader.load_instance('LL_168h', 999)

    def test_inbound_truck_data(self, loader):
        """Test inbound truck data structure."""
        instance = loader.load_instance('LL_168h', 1)
        truck = instance.inbound_trucks[0]

        assert isinstance(truck, InboundTruck)
        assert hasattr(truck, 'truck_id')
        assert hasattr(truck, 'arrival_time')
        assert hasattr(truck, 'pallet_ids')
        assert truck.arrival_time >= 0
        assert len(truck.pallet_ids) > 0

    def test_outbound_truck_data(self, loader):
        """Test outbound truck data structure."""
        instance = loader.load_instance('LL_168h', 1)
        truck = instance.outbound_trucks[0]

        assert isinstance(truck, OutboundTruck)
        assert hasattr(truck, 'truck_id')
        assert hasattr(truck, 'arrival_time')
        assert hasattr(truck, 'due_date')
        assert hasattr(truck, 'destination')
        assert hasattr(truck, 'capacity')
        assert truck.arrival_time >= 0
        assert truck.due_date >= truck.arrival_time
        assert truck.capacity > 0

    def test_pallet_data(self, loader):
        """Test pallet data structure."""
        instance = loader.load_instance('LL_168h', 1)
        pallet = instance.pallets[0]

        assert isinstance(pallet, Pallet)
        assert hasattr(pallet, 'pallet_id')
        assert hasattr(pallet, 'destination')
        assert hasattr(pallet, 'pallet_type')
        assert hasattr(pallet, 'due_date')
        assert pallet.pallet_id >= 0
        assert pallet.destination in [1, 2, 3]
        assert pallet.due_date >= 0

    def test_data_consistency(self, loader):
        """Test consistency between trucks and pallets."""
        instance = loader.load_instance('LL_168h', 1)

        # Check that all pallets in inbound trucks exist
        all_inbound_pallet_ids = set()
        for truck in instance.inbound_trucks:
            all_inbound_pallet_ids.update(truck.pallet_ids)

        all_pallet_ids = {p.pallet_id for p in instance.pallets}
        assert all_inbound_pallet_ids.issubset(all_pallet_ids)

        # Check that destinations in pallets match outbound truck destinations
        pallet_dests = {p.destination for p in instance.pallets}
        truck_dests = {t.destination for t in instance.outbound_trucks}
        assert pallet_dests.issubset(truck_dests)

    def test_metadata(self, loader):
        """Test instance metadata."""
        instance = loader.load_instance('MM_168h', 1)

        assert 'num_inbound_trucks' in instance.metadata
        assert 'num_outbound_trucks' in instance.metadata
        assert 'num_pallets' in instance.metadata
        assert instance.metadata['num_pallets'] == len(instance.pallets)

    def test_scenario_list(self, loader):
        """Test available scenarios listing."""
        scenarios = loader.list_available_scenarios()

        assert len(scenarios) == 6
        assert all(s.endswith('_168h') for s in scenarios)

    def test_multiple_scenarios(self, loader):
        """Test loading instances from multiple scenarios."""
        ll_instance = loader.load_instance('LL_168h', 1)
        hh_instance = loader.load_instance('HH_168h', 1)

        # HH should have more pallets than LL (higher traffic)
        assert len(hh_instance.pallets) > len(ll_instance.pallets)
        assert len(hh_instance.outbound_trucks) > len(ll_instance.outbound_trucks)


class TestCrossDockInstance:
    """Test suite for CrossDockInstance class."""

    @pytest.fixture
    def instance(self):
        """Load a sample instance for testing."""
        loader = DataLoader()
        return loader.load_instance('LL_168h', 1)

    def test_instance_name(self, instance):
        """Test instance name format."""
        assert instance.instance_name == 'LL_168h/instance1'
        assert '/' in instance.instance_name

    def test_truck_count(self, instance):
        """Test truck counts."""
        assert len(instance.inbound_trucks) > 0
        assert len(instance.outbound_trucks) > 0
        # LL scenario should have roughly equal inbound/outbound
        assert abs(len(instance.inbound_trucks) - len(instance.outbound_trucks)) < 10

    def test_pallet_count(self, instance):
        """Test pallet count consistency."""
        # Count pallets from inbound trucks
        inbound_pallet_count = sum(len(t.pallet_ids) for t in instance.inbound_trucks)

        # Should match total pallets
        assert inbound_pallet_count == len(instance.pallets)

    def test_destination_distribution(self, instance):
        """Test pallet distribution across destinations."""
        dest_counts = {}
        for pallet in instance.pallets:
            dest_counts[pallet.destination] = dest_counts.get(pallet.destination, 0) + 1

        # Should have pallets for all 3 destinations
        assert set(dest_counts.keys()) == {1, 2, 3}

        # Distribution should be relatively balanced
        total = sum(dest_counts.values())
        for count in dest_counts.values():
            proportion = count / total
            assert 0.2 < proportion < 0.5  # Each destination gets 20-50%


class TestDataTypes:
    """Test suite for data type classes."""

    def test_inbound_truck_creation(self):
        """Test InboundTruck creation."""
        truck = InboundTruck(
            truck_id=1,
            arrival_time=100.0,
            pallet_ids=[1, 2, 3]
        )

        assert truck.truck_id == 1
        assert truck.arrival_time == 100.0
        assert truck.pallet_ids == [1, 2, 3]

    def test_outbound_truck_creation(self):
        """Test OutboundTruck creation."""
        truck = OutboundTruck(
            truck_id=101,
            arrival_time=200.0,
            due_date=400.0,
            destination=2,
            capacity=26
        )

        assert truck.truck_id == 101
        assert truck.arrival_time == 200.0
        assert truck.due_date == 400.0
        assert truck.destination == 2
        assert truck.capacity == 26

    def test_pallet_creation(self):
        """Test Pallet creation."""
        pallet = Pallet(
            pallet_id=1001,
            destination=1,
            pallet_type='A',
            due_date=500.0
        )

        assert pallet.pallet_id == 1001
        assert pallet.destination == 1
        assert pallet.pallet_type == 'A'
        assert pallet.due_date == 500.0


class TestDataValidation:
    """Test suite for data validation."""

    @pytest.fixture
    def loader(self):
        """Create DataLoader instance."""
        return DataLoader()

    def test_no_missing_values(self, loader):
        """Test that loaded data has no missing values."""
        instance = loader.load_instance('LL_168h', 1)

        # Check inbound trucks
        for truck in instance.inbound_trucks:
            assert truck.truck_id is not None
            assert truck.arrival_time is not None
            assert truck.pallet_ids is not None

        # Check outbound trucks
        for truck in instance.outbound_trucks:
            assert truck.truck_id is not None
            assert truck.arrival_time is not None
            assert truck.due_date is not None
            assert truck.destination is not None
            assert truck.capacity is not None

        # Check pallets
        for pallet in instance.pallets:
            assert pallet.pallet_id is not None
            assert pallet.destination is not None
            assert pallet.due_date is not None

    def test_valid_time_values(self, loader):
        """Test that time values are valid."""
        instance = loader.load_instance('LL_168h', 1)

        # All arrival times should be non-negative
        for truck in instance.inbound_trucks:
            assert truck.arrival_time >= 0

        for truck in instance.outbound_trucks:
            assert truck.arrival_time >= 0
            assert truck.due_date >= truck.arrival_time

        # Pallet due dates should be non-negative
        for pallet in instance.pallets:
            assert pallet.due_date >= 0

    def test_valid_capacity_values(self, loader):
        """Test that capacity values are valid."""
        instance = loader.load_instance('LL_168h', 1)

        for truck in instance.outbound_trucks:
            assert truck.capacity > 0
            assert truck.capacity <= 26  # Standard truck capacity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
