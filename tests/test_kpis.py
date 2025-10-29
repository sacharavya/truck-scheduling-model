"""
Unit tests for KPI calculation and analysis modules.

Tests KPI calculator, metrics, and reporting functionality.
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.models.heuristics import earliest_due_date, first_fit
from src.analysis.kpis import KPICalculator, KPIReport


class TestKPICalculator:
    """Test suite for KPICalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create KPI calculator for testing."""
        return KPICalculator()

    @pytest.fixture
    def instance_and_solution(self):
        """Load instance and generate solution."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)
        solution = earliest_due_date(instance)
        return instance, solution

    def test_calculator_initialization(self, calculator):
        """Test KPI calculator initialization."""
        assert calculator is not None

    def test_calculate_kpis_returns_report(self, calculator, instance_and_solution):
        """Test that calculate_kpis returns KPIReport."""
        instance, solution = instance_and_solution

        report = calculator.calculate_kpis(solution, instance, "EDD")

        assert isinstance(report, KPIReport)
        assert report.algorithm_name == "EDD"

    def test_basic_metrics_calculated(self, calculator, instance_and_solution):
        """Test that basic metrics are calculated."""
        instance, solution = instance_and_solution

        report = calculator.calculate_kpis(solution, instance, "EDD")

        # Check that all basic metrics exist
        assert hasattr(report, 'avg_fill_rate')
        assert hasattr(report, 'service_level')
        assert hasattr(report, 'num_late_pallets')
        assert hasattr(report, 'solve_time')

    def test_fill_rate_range(self, calculator, instance_and_solution):
        """Test that fill rate is in valid range."""
        instance, solution = instance_and_solution

        report = calculator.calculate_kpis(solution, instance, "EDD")

        assert 0 <= report.avg_fill_rate <= 1.0
        assert 0 <= report.min_fill_rate <= 1.0
        assert 0 <= report.max_fill_rate <= 1.0
        assert report.min_fill_rate <= report.avg_fill_rate <= report.max_fill_rate

    def test_service_level_range(self, calculator, instance_and_solution):
        """Test that service level is in valid range."""
        instance, solution = instance_and_solution

        report = calculator.calculate_kpis(solution, instance, "EDD")

        assert 0 <= report.service_level <= 1.0

    def test_late_pallets_count(self, calculator, instance_and_solution):
        """Test late pallets counting."""
        instance, solution = instance_and_solution

        report = calculator.calculate_kpis(solution, instance, "EDD")

        assert report.num_late_pallets >= 0
        assert report.num_late_pallets <= len(instance.pallets)
        assert report.pct_late_pallets >= 0
        assert report.pct_late_pallets <= 100

    def test_unassigned_pallets_count(self, calculator, instance_and_solution):
        """Test unassigned pallets counting."""
        instance, solution = instance_and_solution

        report = calculator.calculate_kpis(solution, instance, "EDD")

        assert report.unassigned_pallets >= 0
        assert report.unassigned_pallets <= len(instance.pallets)

    def test_timing_metrics(self, calculator, instance_and_solution):
        """Test timing-related metrics."""
        instance, solution = instance_and_solution

        report = calculator.calculate_kpis(solution, instance, "EDD")

        assert report.solve_time > 0
        assert report.makespan >= 0
        if report.num_late_pallets > 0:
            assert report.avg_tardiness >= 0
            assert report.max_tardiness >= 0


class TestKPIReport:
    """Test suite for KPIReport class."""

    @pytest.fixture
    def report(self):
        """Generate a sample KPI report."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)
        solution = earliest_due_date(instance)

        calculator = KPICalculator()
        return calculator.calculate_kpis(solution, instance, "EDD")

    def test_report_has_all_fields(self, report):
        """Test that report has all expected fields."""
        required_fields = [
            'algorithm_name',
            'avg_fill_rate',
            'service_level',
            'num_late_pallets',
            'solve_time',
            'makespan',
            'unassigned_pallets'
        ]

        for field in required_fields:
            assert hasattr(report, field), f"Missing field: {field}"

    def test_report_to_dict(self, report):
        """Test conversion to dictionary."""
        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert 'algorithm_name' in report_dict
        assert 'service_level' in report_dict
        assert 'solve_time' in report_dict

    def test_report_print(self, report, capsys):
        """Test print_report method."""
        report.print_report()

        captured = capsys.readouterr()
        assert "KPI REPORT" in captured.out
        assert report.algorithm_name in captured.out

    def test_multiple_algorithms(self):
        """Test reports from different algorithms."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        calculator = KPICalculator()

        edd_solution = earliest_due_date(instance)
        ff_solution = first_fit(instance)

        edd_report = calculator.calculate_kpis(edd_solution, instance, "EDD")
        ff_report = calculator.calculate_kpis(ff_solution, instance, "First-Fit")

        assert edd_report.algorithm_name == "EDD"
        assert ff_report.algorithm_name == "First-Fit"

        # EDD should have better or equal service level
        assert edd_report.service_level >= ff_report.service_level


class TestMetricsConsistency:
    """Test suite for metrics consistency."""

    @pytest.fixture
    def instance_solution_report(self):
        """Generate instance, solution, and report."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)
        solution = earliest_due_date(instance)

        calculator = KPICalculator()
        report = calculator.calculate_kpis(solution, instance, "EDD")

        return instance, solution, report

    def test_pallet_counts_consistent(self, instance_solution_report):
        """Test that pallet counts are consistent."""
        instance, solution, report = instance_solution_report

        total_pallets = len(instance.pallets)
        assigned_pallets = len([p for p in solution.pallet_to_truck.values() if p != 'UNASSIGNED'])
        unassigned_pallets = report.unassigned_pallets

        # Assigned + unassigned should equal total
        assert assigned_pallets + unassigned_pallets == total_pallets

    def test_service_level_calculation(self, instance_solution_report):
        """Test service level calculation consistency."""
        instance, solution, report = instance_solution_report

        total_pallets = len(instance.pallets)
        late_pallets = report.num_late_pallets

        # Calculate service level manually
        expected_service_level = (total_pallets - late_pallets) / total_pallets

        # Should match report (within floating point tolerance)
        assert abs(report.service_level - expected_service_level) < 0.0001

    def test_percentage_calculations(self, instance_solution_report):
        """Test percentage calculations are consistent."""
        instance, solution, report = instance_solution_report

        # Late pallets percentage
        total = len(instance.pallets)
        expected_late_pct = (report.num_late_pallets / total) * 100

        assert abs(report.pct_late_pallets - expected_late_pct) < 0.01


class TestEDDPerformanceMetrics:
    """Test suite specifically for EDD performance metrics."""

    @pytest.fixture
    def edd_reports(self):
        """Generate EDD reports for multiple instances."""
        loader = DataLoader()
        calculator = KPICalculator()

        reports = []
        for i in range(1, 4):  # Test on first 3 instances
            instance = loader.load_instance('LL_168h', i)
            solution = earliest_due_date(instance)
            report = calculator.calculate_kpis(solution, instance, "EDD")
            reports.append(report)

        return reports

    def test_edd_high_service_level(self, edd_reports):
        """Test that EDD achieves high service level."""
        for report in edd_reports:
            # EDD should achieve at least 99% service level
            assert report.service_level >= 0.99

    def test_edd_low_late_pallets(self, edd_reports):
        """Test that EDD has low number of late pallets."""
        for report in edd_reports:
            # EDD should have less than 1% late pallets
            assert report.pct_late_pallets < 1.0

    def test_edd_fast_execution(self, edd_reports):
        """Test that EDD executes quickly."""
        for report in edd_reports:
            # EDD should complete in under 100ms for small instances
            assert report.solve_time < 0.1

    def test_edd_consistency(self, edd_reports):
        """Test that EDD performance is consistent."""
        service_levels = [r.service_level for r in edd_reports]

        # All should be above 99%
        assert all(sl >= 0.99 for sl in service_levels)

        # Variance should be low
        avg_sl = sum(service_levels) / len(service_levels)
        variance = sum((sl - avg_sl) ** 2 for sl in service_levels) / len(service_levels)

        assert variance < 0.0001  # Very low variance


class TestComparisonMetrics:
    """Test suite for algorithm comparison metrics."""

    @pytest.fixture
    def comparison_reports(self):
        """Generate reports from multiple algorithms."""
        loader = DataLoader()
        instance = loader.load_instance('LL_168h', 1)

        calculator = KPICalculator()

        algorithms = {
            'EDD': earliest_due_date,
            'First-Fit': first_fit
        }

        reports = {}
        for name, algo_func in algorithms.items():
            solution = algo_func(instance)
            reports[name] = calculator.calculate_kpis(solution, instance, name)

        return reports

    def test_edd_outperforms_first_fit(self, comparison_reports):
        """Test that EDD outperforms First-Fit."""
        edd_report = comparison_reports['EDD']
        ff_report = comparison_reports['First-Fit']

        # EDD should have better or equal service level
        assert edd_report.service_level >= ff_report.service_level

        # EDD should have fewer or equal late pallets
        assert edd_report.num_late_pallets <= ff_report.num_late_pallets

    def test_all_algorithms_complete_quickly(self, comparison_reports):
        """Test that all algorithms complete quickly."""
        for name, report in comparison_reports.items():
            # All heuristics should be fast
            assert report.solve_time < 1.0, f"{name} too slow: {report.solve_time}s"


class TestEdgeCases:
    """Test suite for edge cases."""

    @pytest.fixture
    def calculator(self):
        """Create KPI calculator."""
        return KPICalculator()

    def test_different_instance_sizes(self, calculator):
        """Test KPI calculation on different instance sizes."""
        loader = DataLoader()

        scenarios = ['LL_168h', 'MM_168h', 'HH_168h']

        for scenario in scenarios:
            instance = loader.load_instance(scenario, 1)
            solution = earliest_due_date(instance)
            report = calculator.calculate_kpis(solution, instance, "EDD")

            # All reports should be valid
            assert report.service_level >= 0
            assert report.solve_time > 0

    def test_multiple_instances_same_scenario(self, calculator):
        """Test consistency across multiple instances."""
        loader = DataLoader()

        reports = []
        for i in range(1, 4):
            instance = loader.load_instance('LL_168h', i)
            solution = earliest_due_date(instance)
            report = calculator.calculate_kpis(solution, instance, "EDD")
            reports.append(report)

        # All should have high service level
        assert all(r.service_level >= 0.95 for r in reports)

        # All should complete quickly
        assert all(r.solve_time < 0.1 for r in reports)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
