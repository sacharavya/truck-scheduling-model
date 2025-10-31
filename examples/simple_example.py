#!/usr/bin/env python3
"""
Simple Example: Cross-Docking Optimization

This script demonstrates how to:
1. Load a cross-docking instance from the dataset
2. Run the EDD (Earliest Due Date) optimization algorithm
3. Calculate performance metrics (KPIs)
4. Display and save results

Usage:
    python examples/simple_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import DataLoader
from src.models.heuristics import earliest_due_date
from src.analysis.kpis import KPICalculator
from src.analysis.visualization import SolutionVisualizer


def main():
    """Run a simple optimization example."""

    print("="*80)
    print("CROSS-DOCKING OPTIMIZATION - SIMPLE EXAMPLE")
    print("="*80)

    # Step 1: Load a dataset instance
    print("\n[Step 1] Loading dataset instance...")
    print("  Scenario: LL_168h (Low traffic)")
    print("  Instance: 1")

    loader = DataLoader(dataset_root=str(project_root / 'data'))
    instance = loader.load_instance('MM_168h', 1)

    # Get unique destinations from pallets
    destinations = set(p.destination for p in instance.pallets)

    print(f"\n  ✅ Instance loaded successfully!")
    print(f"     - Inbound trucks: {len(instance.inbound_trucks)}")
    print(f"     - Outbound trucks: {len(instance.outbound_trucks)}")
    print(f"     - Total pallets: {len(instance.pallets)}")
    print(f"     - Destinations: {len(destinations)}")

    # Step 2: Run optimization
    print("\n[Step 2] Running EDD (Earliest Due Date) optimization...")
    print("  Algorithm: EDD Heuristic")
    print("  Expected solve time: <10ms")

    solution = earliest_due_date(instance)

    print(f"\n  ✅ Optimization complete!")
    print(f"     - Pallet assignments: {len(solution.assignments)}")
    print(f"     - Outbound trucks used: {len(solution.truck_loads)}")

    # Step 3: Calculate KPIs
    print("\n[Step 3] Calculating performance metrics (KPIs)...")

    calculator = KPICalculator()
    kpis = calculator.calculate_kpis(solution, instance)

    print("\n  ✅ KPIs calculated!")

    # Step 4: Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n📊 Performance Metrics:")
    print(f"   Service Level:       {kpis.service_level:>8.2%}  (Target: >95%)")
    print(f"   Fill Rate (avg):     {kpis.avg_fill_rate:>8.2%}  (Target: >80%)")
    print(f"   Solve Time:          {kpis.solve_time:>8.4f} sec")

    print(f"\n📦 Pallet Statistics:")
    print(f"   Total pallets:       {kpis.total_pallets:>8}")
    print(f"   Assigned pallets:    {kpis.assigned_pallets:>8}")
    print(f"   Late pallets:        {kpis.num_late_pallets:>8}")
    print(f"   Unassigned pallets:  {kpis.unassigned_pallets:>8}")

    print(f"\n🚚 Truck Statistics:")
    print(f"   Trucks utilized:     {kpis.trucks_utilized:>8}")
    print(f"   Total capacity:      {kpis.total_truck_capacity:>8}")
    print(f"   Utilized capacity:   {kpis.utilized_capacity:>8}")

    print(f"\n💰 Estimated Costs:")
    cost_per_late = 100.0  # $100 per late pallet
    cost_per_unassigned = 200.0  # $200 per unassigned pallet
    total_cost = (kpis.num_late_pallets * cost_per_late +
                  kpis.unassigned_pallets * cost_per_unassigned)
    print(f"   Late pallet costs:   ${kpis.num_late_pallets * cost_per_late:>8,.2f}")
    print(f"   Unassigned costs:    ${kpis.unassigned_pallets * cost_per_unassigned:>8,.2f}")
    print(f"   Total cost:          ${total_cost:>8,.2f}")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    if kpis.service_level >= 0.99:
        status = "✅ EXCELLENT"
    elif kpis.service_level >= 0.95:
        status = "✅ GOOD"
    elif kpis.service_level >= 0.90:
        status = "⚠️  ACCEPTABLE"
    else:
        status = "❌ POOR"

    print(f"\nService Level: {status}")
    print(f"  - {kpis.service_level:.2%} of pallets delivered on time")
    print(f"  - Industry benchmark: 95%+")

    if kpis.avg_fill_rate >= 0.85:
        fill_status = "✅ EXCELLENT"
    elif kpis.avg_fill_rate >= 0.75:
        fill_status = "✅ GOOD"
    else:
        fill_status = "⚠️  LOW"

    print(f"\nTruck Utilization: {fill_status}")
    print(f"  - {kpis.avg_fill_rate:.2%} average fill rate")
    print(f"  - Each truck capacity: 26 pallets")

    print(f"\nOperational Efficiency:")
    if kpis.unassigned_pallets == 0:
        print(f"  ✅ All pallets assigned to outbound trucks")
    else:
        print(f"  ⚠️  {kpis.unassigned_pallets} pallets remain unassigned")

    # Step 5: Save results (optional)
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_dir = project_root / 'results' / 'examples'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save solution details
    output_file = output_dir / 'simple_example_results.txt'
    with open(output_file, 'w') as f:
        f.write("CROSS-DOCKING OPTIMIZATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Instance: LL_168h/instance1\n")
        f.write(f"Algorithm: EDD (Earliest Due Date)\n\n")
        f.write(f"Service Level: {kpis.service_level:.2%}\n")
        f.write(f"Fill Rate: {kpis.avg_fill_rate:.2%}\n")
        f.write(f"Late Pallets: {kpis.num_late_pallets}\n")
        f.write(f"Total Cost: ${total_cost:,.2f}\n")

    print(f"\n✅ Results saved to: {output_file}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Try different scenarios:")
    print("   - 'HH_168h' for high traffic")
    print("   - 'MM_168h' for medium traffic")
    print("\n2. Compare algorithms:")
    print("   - Run examples/compare_algorithms.py")
    print("\n3. Explore in Jupyter notebooks:")
    print("   - notebooks/01_data_exploration.ipynb")
    print("   - notebooks/02_optimization_experiments.ipynb")
    print("\n4. Run full benchmarks:")
    print("   - notebooks/03_results_dashboard.ipynb")

    print("\n" + "="*80)
    print("✅ EXAMPLE COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
