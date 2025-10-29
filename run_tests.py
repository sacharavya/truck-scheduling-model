#!/usr/bin/env python3
"""
Test Runner for Cross-Docking Optimization Project

Provides convenient test execution with various options.
"""

import pytest
import sys
from pathlib import Path


def run_all_tests():
    """Run all tests with verbose output."""
    print("="*80)
    print("RUNNING ALL TESTS")
    print("="*80)
    return pytest.main(["-v", "tests/"])


def run_unit_tests():
    """Run only unit tests."""
    print("="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)
    return pytest.main(["-v", "-m", "unit", "tests/"])


def run_integration_tests():
    """Run only integration tests."""
    print("="*80)
    print("RUNNING INTEGRATION TESTS")
    print("="*80)
    return pytest.main(["-v", "tests/test_integration.py"])


def run_quick_tests():
    """Run quick tests only."""
    print("="*80)
    print("RUNNING QUICK TESTS")
    print("="*80)
    return pytest.main(["-v", "-m", "quick", "tests/"])


def run_specific_module(module_name):
    """Run tests for a specific module."""
    print("="*80)
    print(f"RUNNING TESTS FOR: {module_name}")
    print("="*80)

    test_file = f"tests/test_{module_name}.py"
    if not Path(test_file).exists():
        print(f"Error: Test file not found: {test_file}")
        return 1

    return pytest.main(["-v", test_file])


def run_with_coverage():
    """Run tests with coverage reporting."""
    print("="*80)
    print("RUNNING TESTS WITH COVERAGE")
    print("="*80)
    return pytest.main([
        "-v",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term",
        "tests/"
    ])


def print_usage():
    """Print usage information."""
    print("""
Cross-Docking Optimization Test Runner
======================================

Usage: python run_tests.py [command]

Commands:
    all             Run all tests (default)
    unit            Run unit tests only
    integration     Run integration tests only
    quick           Run quick tests only
    coverage        Run all tests with coverage report
    <module>        Run tests for specific module (e.g., 'data_loader', 'heuristics')

Examples:
    python run_tests.py
    python run_tests.py all
    python run_tests.py integration
    python run_tests.py data_loader
    python run_tests.py coverage

Test Modules:
    - data_loader       Tests for data loading and validation
    - heuristics        Tests for heuristic algorithms
    - kpis              Tests for KPI calculation
    - integration       End-to-end integration tests
""")


def main():
    """Main entry point."""
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        return run_all_tests()

    command = sys.argv[1]

    if command == "help" or command == "-h" or command == "--help":
        print_usage()
        return 0
    elif command == "unit":
        return run_unit_tests()
    elif command == "integration":
        return run_integration_tests()
    elif command == "quick":
        return run_quick_tests()
    elif command == "coverage":
        return run_with_coverage()
    else:
        # Assume it's a module name
        return run_specific_module(command)


if __name__ == "__main__":
    sys.exit(main())
