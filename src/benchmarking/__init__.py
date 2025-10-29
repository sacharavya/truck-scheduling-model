from pathlib import Path

self.dataset_root = Path(dataset_root)
if not self.dataset_root.exists():
    raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")