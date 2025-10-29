"""
Visualization Suite for Cross-Docking Analysis

This module provides comprehensive visualization functions for analyzing
optimization solutions, comparing algorithms, and presenting results.

Visualizations include:
- Gantt charts for truck schedules
- Fill rate distributions
- Tardiness analysis
- Solution comparisons
- Interactive dashboards

Author: Cross-Docking Optimization Project
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import CrossDockInstance
from models.pallet_assignment import AssignmentSolution
from analysis.kpis import KPIReport, KPICalculator

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')


class SolutionVisualizer:
    """Visualizer for optimization solutions."""

    def __init__(self, figsize=(12, 6), dpi=300):
        """Initialize visualizer with default settings."""
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette('Set2', 8)

    def plot_fill_rate_distribution(
        self,
        solution: AssignmentSolution,
        instance: CrossDockInstance,
        save_path: Optional[str] = None
    ):
        """Plot distribution of truck fill rates."""
        truck_dict = {t.truck_id: t for t in instance.outbound_trucks}

        fill_rates = []
        destinations = []

        for truck_id, pallet_ids in solution.truck_loads.items():
            fill_rate = len(pallet_ids) / 26  # Capacity = 26
            fill_rates.append(fill_rate * 100)  # Convert to percentage
            destinations.append(truck_dict[truck_id].destination)

        # Create DataFrame
        df = pd.DataFrame({
            'fill_rate': fill_rates,
            'destination': destinations
        })

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overall histogram
        axes[0].hist(fill_rates, bins=20, alpha=0.7, color=self.colors[0], edgecolor='black')
        axes[0].axvline(np.mean(fill_rates), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(fill_rates):.1f}%')
        axes[0].set_xlabel('Fill Rate (%)')
        axes[0].set_ylabel('Number of Trucks')
        axes[0].set_title('Truck Fill Rate Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # By destination
        for dest in [1, 2, 3]:
            dest_fill_rates = df[df['destination'] == dest]['fill_rate']
            axes[1].hist(dest_fill_rates, bins=15, alpha=0.5,
                        label=f'Dest {dest}', edgecolor='black')

        axes[1].set_xlabel('Fill Rate (%)')
        axes[1].set_ylabel('Number of Trucks')
        axes[1].set_title('Fill Rate Distribution by Destination')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def plot_tardiness_analysis(
        self,
        solution: AssignmentSolution,
        instance: CrossDockInstance,
        save_path: Optional[str] = None
    ):
        """Plot tardiness analysis."""
        truck_dict = {t.truck_id: t for t in instance.outbound_trucks}
        pallet_dict = {p.pallet_id: p for p in instance.pallets}

        tardiness_data = []

        for truck_id, pallet_ids in solution.truck_loads.items():
            truck = truck_dict[truck_id]
            for pid in pallet_ids:
                pallet = pallet_dict[pid]
                tardiness = max(0, truck.due_date - pallet.due_date)
                is_late = pallet.due_date < truck.due_date

                tardiness_data.append({
                    'pallet_id': pid,
                    'destination': pallet.destination,
                    'tardiness': tardiness,
                    'is_late': is_late
                })

        df = pd.DataFrame(tardiness_data)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Overall tardiness distribution
        late_df = df[df['is_late']]
        if len(late_df) > 0:
            axes[0, 0].hist(late_df['tardiness'], bins=30, alpha=0.7,
                           color=self.colors[1], edgecolor='black')
            axes[0, 0].axvline(late_df['tardiness'].mean(), color='red',
                              linestyle='--', linewidth=2,
                              label=f"Mean: {late_df['tardiness'].mean():.1f} min")
            axes[0, 0].set_xlabel('Tardiness (minutes)')
            axes[0, 0].set_ylabel('Number of Pallets')
            axes[0, 0].set_title(f'Tardiness Distribution ({len(late_df)} late pallets)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No late pallets!',
                           ha='center', va='center', fontsize=14)
            axes[0, 0].set_title('Tardiness Distribution')

        # 2. On-time vs Late by destination
        dest_summary = df.groupby('destination')['is_late'].agg(['sum', 'count'])
        dest_summary['on_time'] = dest_summary['count'] - dest_summary['sum']

        x = np.arange(len(dest_summary))
        width = 0.35

        axes[0, 1].bar(x - width/2, dest_summary['on_time'], width,
                      label='On-Time', color=self.colors[2])
        axes[0, 1].bar(x + width/2, dest_summary['sum'], width,
                      label='Late', color=self.colors[3])
        axes[0, 1].set_xlabel('Destination')
        axes[0, 1].set_ylabel('Number of Pallets')
        axes[0, 1].set_title('On-Time vs Late Pallets by Destination')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'Dest {i}' for i in dest_summary.index])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. Service level by destination
        dest_summary['service_level'] = (dest_summary['on_time'] / dest_summary['count']) * 100

        axes[1, 0].bar(dest_summary.index, dest_summary['service_level'],
                      color=self.colors[4], edgecolor='black')
        axes[1, 0].axhline(95, color='red', linestyle='--', linewidth=2,
                          label='95% Target')
        axes[1, 0].set_xlabel('Destination')
        axes[1, 0].set_ylabel('Service Level (%)')
        axes[1, 0].set_title('Service Level by Destination')
        axes[1, 0].set_ylim([0, 105])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. Cumulative tardiness
        if len(late_df) > 0:
            sorted_tardiness = np.sort(late_df['tardiness'].values)
            cumulative = np.cumsum(sorted_tardiness)

            axes[1, 1].plot(range(1, len(sorted_tardiness) + 1), cumulative,
                           linewidth=2, color=self.colors[5])
            axes[1, 1].fill_between(range(1, len(sorted_tardiness) + 1), cumulative,
                                   alpha=0.3, color=self.colors[5])
            axes[1, 1].set_xlabel('Number of Late Pallets')
            axes[1, 1].set_ylabel('Cumulative Tardiness (minutes)')
            axes[1, 1].set_title('Cumulative Tardiness')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No tardiness!',
                           ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('Cumulative Tardiness')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def plot_solution_comparison(
        self,
        reports: List[KPIReport],
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Compare multiple solutions across key metrics."""
        if metrics is None:
            metrics = ['avg_fill_rate', 'service_level', 'pct_unassigned', 'solve_time']

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        solution_names = [r.solution_name for r in reports]
        x = np.arange(len(solution_names))

        metric_labels = {
            'avg_fill_rate': ('Average Fill Rate (%)', 100),
            'service_level': ('Service Level (%)', 100),
            'pct_unassigned': ('Unassigned (%)', 100),
            'solve_time': ('Solve Time (s)', 1),
            'num_late_pallets': ('Late Pallets', 1),
            'total_tardiness': ('Total Tardiness (min)', 1)
        }

        for idx, metric in enumerate(metrics):
            values = [getattr(r, metric) for r in reports]
            label, multiplier = metric_labels.get(metric, (metric, 1))

            values = [v * multiplier for v in values]

            axes[idx].bar(x, values, color=self.colors[idx % len(self.colors)],
                         edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel('Solution')
            axes[idx].set_ylabel(label)
            axes[idx].set_title(label)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(solution_names, rotation=45, ha='right')
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def plot_gantt_chart(
        self,
        solution: AssignmentSolution,
        instance: CrossDockInstance,
        max_trucks: int = 50,
        save_path: Optional[str] = None
    ):
        """Create Gantt chart for truck schedules (limited to first max_trucks)."""
        truck_dict = {t.truck_id: t for t in instance.outbound_trucks}

        # Prepare data
        truck_data = []
        for idx, (truck_id, pallet_ids) in enumerate(list(solution.truck_loads.items())[:max_trucks]):
            truck = truck_dict[truck_id]
            truck_data.append({
                'truck_id': truck_id,
                'arrival': truck.arrival_time,
                'due_date': truck.due_date,
                'destination': truck.destination,
                'num_pallets': len(pallet_ids),
                'idx': idx
            })

        df = pd.DataFrame(truck_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, max_trucks * 0.15)))

        colors_by_dest = {1: self.colors[0], 2: self.colors[1], 3: self.colors[2]}

        for _, row in df.iterrows():
            # Arrival to due date bar
            duration = row['due_date'] - row['arrival']
            ax.barh(row['idx'], duration, left=row['arrival'],
                   height=0.6, color=colors_by_dest[row['destination']],
                   alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add pallet count text
            ax.text(row['arrival'] + duration/2, row['idx'],
                   f"{row['num_pallets']}", ha='center', va='center',
                   fontsize=7, fontweight='bold')

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Truck Index')
        ax.set_title(f'Truck Schedule Gantt Chart (First {len(df)} trucks)')
        ax.set_yticks(df['idx'])
        ax.set_yticklabels([f"T{tid}" for tid in df['truck_id']], fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

        # Legend
        legend_elements = [
            mpatches.Patch(color=colors_by_dest[1], label='Destination 1'),
            mpatches.Patch(color=colors_by_dest[2], label='Destination 2'),
            mpatches.Patch(color=colors_by_dest[3], label='Destination 3')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def create_summary_dashboard(
        self,
        report: KPIReport,
        save_path: Optional[str] = None
    ):
        """Create a comprehensive dashboard with key metrics."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(f'Solution Dashboard: {report.solution_name}\n{report.instance_name}',
                    fontsize=16, fontweight='bold')

        # 1. Fill Rate Gauge (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_gauge(ax1, report.avg_fill_rate, 'Fill Rate', '%')

        # 2. Service Level Gauge (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_gauge(ax2, report.service_level, 'Service Level', '%')

        # 3. Unassigned Gauge (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._create_gauge(ax3, 1 - report.pct_unassigned, 'Assignment Rate', '%')

        # 4. Key metrics table (middle-left)
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('tight')
        ax4.axis('off')

        table_data = [
            ['Metric', 'Value'],
            ['Trucks Utilized', f"{report.trucks_utilized}"],
            ['Total Pallets', f"{report.total_pallets}"],
            ['Assigned Pallets', f"{report.assigned_pallets}"],
            ['Unassigned Pallets', f"{report.unassigned_pallets}"],
            ['Late Pallets', f"{report.num_late_pallets} ({report.pct_late_pallets:.2%})"],
            ['Total Tardiness', f"{report.total_tardiness:.2f} min"],
            ['Avg Tardiness', f"{report.avg_tardiness:.2f} min"],
            ['Solve Time', f"{report.solve_time:.3f} s"],
            ['Status', report.status]
        ]

        table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 5. Fill rate by destination (bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        if report.fill_rate_by_dest:
            dests = list(report.fill_rate_by_dest.keys())
            rates = [report.fill_rate_by_dest[d] * 100 for d in dests]
            ax5.bar(dests, rates, color=self.colors[:len(dests)], edgecolor='black')
            ax5.set_xlabel('Destination')
            ax5.set_ylabel('Fill Rate (%)')
            ax5.set_title('Fill Rate by Destination')
            ax5.set_ylim([0, 105])
            ax5.grid(True, alpha=0.3, axis='y')

        # 6. Late pallets by destination (bottom-center)
        ax6 = fig.add_subplot(gs[2, 1])
        if report.late_pallets_by_dest:
            dests = list(report.late_pallets_by_dest.keys())
            late_counts = [report.late_pallets_by_dest[d] for d in dests]
            ax6.bar(dests, late_counts, color=self.colors[3:3+len(dests)], edgecolor='black')
            ax6.set_xlabel('Destination')
            ax6.set_ylabel('Late Pallets')
            ax6.set_title('Late Pallets by Destination')
            ax6.grid(True, alpha=0.3, axis='y')

        # 7. Summary pie chart (bottom-right)
        ax7 = fig.add_subplot(gs[2, 2])
        pie_data = [
            report.assigned_pallets - report.num_late_pallets,  # On-time
            report.num_late_pallets,  # Late
            report.unassigned_pallets  # Unassigned
        ]
        pie_labels = ['On-Time', 'Late', 'Unassigned']
        pie_colors = [self.colors[2], self.colors[3], self.colors[4]]

        ax7.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
               colors=pie_colors, startangle=90)
        ax7.set_title('Pallet Status Distribution')

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")

        plt.show()

    def _create_gauge(self, ax, value, title, unit=''):
        """Create a simple gauge chart."""
        value_pct = value * 100 if value <= 1 else value

        # Determine color based on value
        if value_pct >= 95:
            color = '#4CAF50'  # Green
        elif value_pct >= 80:
            color = '#FFC107'  # Yellow
        else:
            color = '#F44336'  # Red

        # Draw gauge
        ax.barh([0], [value_pct], height=0.3, color=color, edgecolor='black')
        ax.barh([0], [100], height=0.3, color='lightgray', alpha=0.3, zorder=0)

        ax.set_xlim([0, 100])
        ax.set_ylim([-0.5, 0.5])
        ax.set_yticks([])
        ax.set_xlabel(f'{unit}')
        ax.set_title(f'{title}\n{value_pct:.1f}{unit}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')


# Testing
if __name__ == "__main__":
    import os
    from data_loader import DataLoader
    from models.heuristics import earliest_due_date, first_fit
    from analysis.kpis import KPICalculator

    # Change to project root
    os.chdir(Path(__file__).parent.parent.parent)

    # Load instance
    loader = DataLoader()
    instance = loader.load_instance('LL_168h', 1)

    # Solve with EDD
    edd_solution = earliest_due_date(instance)
    ff_solution = first_fit(instance)

    # Calculate KPIs
    calculator = KPICalculator()
    edd_report = calculator.calculate_kpis(edd_solution, instance, "EDD")
    ff_report = calculator.calculate_kpis(ff_solution, instance, "First-Fit")

    # Create visualizer
    viz = SolutionVisualizer()

    print("Creating visualizations...\n")

    # 1. Fill rate distribution
    viz.plot_fill_rate_distribution(edd_solution, instance,
                                     'results/figures/fill_rate_dist_edd.png')

    # 2. Tardiness analysis
    viz.plot_tardiness_analysis(edd_solution, instance,
                                'results/figures/tardiness_analysis_edd.png')

    # 3. Solution comparison
    viz.plot_solution_comparison([edd_report, ff_report],
                                metrics=['avg_fill_rate', 'service_level', 'solve_time'],
                                save_path='results/figures/solution_comparison.png')

    # 4. Gantt chart
    viz.plot_gantt_chart(edd_solution, instance, max_trucks=30,
                        save_path='results/figures/gantt_chart_edd.png')

    # 5. Dashboard
    viz.create_summary_dashboard(edd_report,
                                 save_path='results/figures/dashboard_edd.png')

    print("\nAll visualizations created successfully!")
