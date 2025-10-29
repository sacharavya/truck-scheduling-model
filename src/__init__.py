"""
Cross-Docking Optimization Package

Main modules:
- data_loader: Data loading and validation
- models: Optimization models (MILP and heuristics)
- analysis: KPI calculation and visualization
- simulation: Discrete-event simulation framework
- benchmarking: Automated benchmarking pipeline
"""

__version__ = "1.0.0"

from pathlib import Path

self.dataset_root = Path(dataset_root)
if not self.dataset_root.exists():
    raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
