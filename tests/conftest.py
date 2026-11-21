import numpy as np
import pytest


@pytest.fixture
def tiny_medmnist(tmp_path):
    """A synthetic MedMNIST-style dataset (3 splits, 2 classes, 8x8 images)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "ds"
    root.mkdir()
    sizes = {"train": 60, "val": 20, "test": 30}
    for split, n in sizes.items():
        labels = (np.arange(n) % 2).reshape(-1, 1).astype(np.uint8)
        images = rng.integers(0, 255, size=(n, 8, 8), dtype=np.uint8)
        np.save(root / f"{split}_images.npy", images)
        np.save(root / f"{split}_labels.npy", labels)
    return root
