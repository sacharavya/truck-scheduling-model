#!/usr/bin/env python3
"""
Run Cross-Docking Simulation

This script runs a discrete-event simulation of cross-docking operations,
validating optimization solutions under realistic conditions.

Features:
- Load any instance from the dataset
- Solve with optimization algorithm
- Simulate realistic terminal operations
- Track resource utilization, queues, and performance
- Export detailed event logs

Usage:
    python3 run_simulation.py                    # Interactive mode
    python3 run_simulation.py --scenario LL_168h --instance 1  # Direct mode
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data_loader import DataLoader
from src.models.heuristics import earliest_due_date, first_fit, best_fit
from src.simulation.cross_dock_sim import CrossDockSimulation, SimulationConfig
from src.simulation.optimizer_integration import OptimizationValidator
import pandas as pd


def main():
    """Run the simulation."""

    parser = argparse.ArgumentParser(description='Run cross-docking simulation')
    parser.add_argument('--scenario', type=str, help='Scenario name (e.g., LL_168h)')
    parser.add_argument('--instance', type=int, help='Instance number (1-10)')
    parser.add_argument('--algorithm', type=str, default='EDD',
                       choices=['EDD', 'First-Fit', 'Best-Fit'],
                       help='Optimization algorithm')
    parser.add_argument('--forklifts', type=int, default=15,
                       help='Number of forklifts')
    parser.add_argument('--export-events', action='store_true',
                       help='Export event log to CSV')
    args = parser.parse_args()

    print("="*80)
    print("CROSS-DOCKING TERMINAL SIMULATION")
    print("="*80)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Interactive mode if no arguments
    if not args.scenario:
        print("Available scenarios:")
        scenarios = ['HH_168h', 'MH_168h', 'MM_168h', 'LH_168h', 'LM_168h', 'LL_168h']
        for i, scenario in enumerate(scenarios, 1):
            print(f"  {i}. {scenario}")

        choice = input("\nSelect scenario (1-6) [default: 6 (LL_168h)]: ").strip()
        scenario_idx = int(choice) - 1 if choice else 5
        args.scenario = scenarios[scenario_idx]

    if not args.instance:
        instance_choice = input("Select instance (1-10) [default: 1]: ").strip()
        args.instance = int(instance_choice) if instance_choice else 1

    if not args.algorithm or sys.argv == [sys.argv[0]]:
        print("\nAvailable algorithms:")
        print("  1. EDD (Earliest Due Date) - Best performance")
        print("  2. First-Fit - Simple greedy")
        print("  3. Best-Fit - Better packing")
        algo_choice = input("Select algorithm (1-3) [default: 1 (EDD)]: ").strip()
        algo_map = {' 1': 'EDD', '2': 'First-Fit', '3': 'Best-Fit'}
        args.algorithm = algo_map.get(algo_choice, 'EDD')

    # Load instance
    print("\n" + "="*80)
    print("LOADING INSTANCE")
    print("="*80)
    print(f"\nScenario: {args.scenario}")
    print(f"Instance: {args.instance}")

    loader = DataLoader(dataset_root=str(project_root / 'data'))
    instance = loader.load_instance(args.scenario, args.instance)

    print(f"\n✅ Instance loaded successfully!")
    print(f"   Inbound trucks: {len(instance.inbound_trucks)}")
    print(f"   Outbound trucks: {len(instance.outbound_trucks)}")
    print(f"   Total pallets: {len(instance.pallets)}")

    # Solve with optimization algorithm
    print("\n" + "="*80)
    print("OPTIMIZATION")
    print("="*80)
    print(f"\nAlgorithm: {args.algorithm}")

    algorithms = {
        'EDD': earliest_due_date,
        'First-Fit': first_fit,
        'Best-Fit': best_fit
    }

    solution = algorithms[args.algorithm](instance)

    print(f"\n✅ Optimization complete!")
    print(f"   Solution status: {solution.status}")
    print(f"   Fill rate: {solution.avg_fill_rate:.2%}")
    print(f"   Late pallets (optimization): {solution.num_late_pallets}")
    print(f"   Solve time: {solution.solve_time:.4f}s")

    # Configure simulation
    print("\n" + "="*80)
    print("SIMULATION CONFIGURATION")
    print("="*80)

    config = SimulationConfig(
        num_forklifts=args.forklifts,
        num_inbound_doors=1,
        num_outbound_doors=1,
        pallet_unload_time=2.0,
        pallet_load_time=2.0,
        truck_positioning_time=5.0,
        pallet_transfer_time=1.5,
        random_seed=42
    )

    print(f"\nResources:")
    print(f"   Forklifts: {config.num_forklifts}")
    print(f"   Inbound doors: {config.num_inbound_doors}")
    print(f"   Outbound doors: {config.num_outbound_doors}")

    print(f"\nProcessing times:")
    print(f"   Pallet unload: {config.pallet_unload_time} min")
    print(f"   Pallet load: {config.pallet_load_time} min")
    print(f"   Truck positioning: {config.truck_positioning_time} min")
    print(f"   Pallet transfer: {config.pallet_transfer_time} min")

    # Run simulation
    print("\n" + "="*80)
    print("RUNNING SIMULATION")
    print("="*80)
    print(f"\nEstimated duration: ~10-30 seconds for this instance")
    print("Please wait...\n")

    start_time = datetime.now()

    sim = CrossDockSimulation(instance, solution, config)
    results = sim.run()

    end_time = datetime.now()
    sim_duration = (end_time - start_time).total_seconds()

    print(f"✅ Simulation complete! (took {sim_duration:.2f} seconds)")

    # Print results
    print("\n" + "="*80)
    print("SIMULATION RESULTS")
    print("="*80)

    total_pallets = results.total_pallets_processed
    service_level = results.pallets_delivered_on_time / total_pallets if total_pallets > 0 else 0

    print(f"\nInstance: {instance.instance_name}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Makespan: {results.makespan:.2f} minutes ({results.makespan/60:.1f} hours)")

    print(f"\n📦 PALLET FLOW")
    print(f"   Total Processed: {total_pallets}")
    print(f"   On-Time: {results.pallets_delivered_on_time}")
    print(f"   Late: {results.pallets_delivered_late}")
    print(f"   Service Level: {service_level:.2%}")
    print(f"   Total Tardiness: {results.total_tardiness:.2f} min")

    print(f"\n🚜 RESOURCE UTILIZATION")
    print(f"   Avg Forklift Utilization: {results.avg_forklift_utilization:.2%}")
    print(f"   Avg Inbound Door Utilization: {results.avg_inbound_door_utilization:.2%}")
    print(f"   Avg Outbound Door Utilization: {results.avg_outbound_door_utilization:.2%}")

    print(f"\n📊 QUEUE STATISTICS")
    print(f"   Avg Inbound Queue: {results.avg_inbound_queue_length:.2f} trucks")
    print(f"   Max Inbound Queue: {results.max_inbound_queue_length} trucks")
    print(f"   Avg Outbound Queue: {results.avg_outbound_queue_length:.2f} trucks")
    print(f"   Max Outbound Queue: {results.max_outbound_queue_length} trucks")

    print(f"\n📦 STAGING INVENTORY")
    print(f"   Avg Inventory: {results.avg_staging_inventory:.2f} pallets")
    print(f"   Max Inventory: {results.max_staging_inventory} pallets")

    print(f"\n⏱️  TIMING")
    print(f"   Avg Pallet Flow Time: {results.avg_pallet_flow_time:.2f} min")
    print(f"   Total Makespan: {results.makespan:.2f} min ({results.makespan/1440:.1f} days)")

    # Comparison with optimization
    print("\n" + "="*80)
    print("OPTIMIZATION vs SIMULATION COMPARISON")
    print("="*80)

    opt_service = (len(instance.pallets) - solution.num_late_pallets) / len(instance.pallets)
    sim_service = service_level

    print(f"\nService Level:")
    print(f"   Optimization prediction: {opt_service:.2%}")
    print(f"   Simulation actual: {sim_service:.2%}")
    print(f"   Difference: {sim_service - opt_service:+.2%}")

    print(f"\nLate Pallets:")
    print(f"   Optimization prediction: {solution.num_late_pallets}")
    print(f"   Simulation actual: {results.pallets_delivered_late}")
    print(f"   Difference: {results.pallets_delivered_late - solution.num_late_pallets:+d}")

    # Export events if requested
    if args.export_events:
        print("\n" + "="*80)
        print("EXPORTING EVENT LOG")
        print("="*80)

        output_dir = project_root / 'results' / 'simulation'
        output_dir.mkdir(parents=True, exist_ok=True)

        events_file = output_dir / f'events_{args.scenario}_inst{args.instance}_{args.algorithm}.csv'
        events_df = pd.DataFrame(results.events)
        events_df.to_csv(events_file, index=False)

        print(f"\n✅ Event log exported:")
        print(f"   File: {events_file}")
        print(f"   Total events: {len(events_df)}")

        # Show sample events
        print(f"\nFirst 10 events:")
        print(events_df.head(10).to_string())

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if service_level >= 0.99:
        status = "✅ EXCELLENT"
    elif service_level >= 0.95:
        status = "✅ GOOD"
    elif service_level >= 0.90:
        status = "⚠️  ACCEPTABLE"
    else:
        status = "❌ POOR"

    print(f"\nPerformance: {status}")
    print(f"   Service Level: {service_level:.2%}")
    print(f"   Late Pallets: {results.pallets_delivered_late}")
    print(f"   Avg Flow Time: {results.avg_pallet_flow_time:.2f} min")

    util_status = "✅ GOOD" if results.avg_forklift_utilization < 0.8 else "⚠️  HIGH"
    print(f"\nResource Utilization: {util_status}")
    print(f"   Forklift utilization: {results.avg_forklift_utilization:.2%}")
    if results.avg_forklift_utilization > 0.8:
        print(f"   ⚠️  Consider adding more forklifts (currently {args.forklifts})")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Try different scenarios:")
    print("   python3 run_simulation.py --scenario HH_168h --instance 1")
    print("\n2. Compare algorithms:")
    print("   python3 run_simulation.py --algorithm First-Fit")
    print("\n3. Adjust resources:")
    print("   python3 run_simulation.py --forklifts 20")
    print("\n4. Export event logs:")
    print("   python3 run_simulation.py --export-events")

    print("\n" + "="*80)
    print("✅ SIMULATION COMPLETE!")
    print("="*80)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
