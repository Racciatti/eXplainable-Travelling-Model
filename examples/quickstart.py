"""End-to-end XTM demo using the programmatic API (no interactive prompts).

What it does:
  1. Downloads PneumoniaMNIST (28x28, ~4 MB) with the ``medmnist`` package.
  2. Stores it in the ``.npy`` layout XTM expects.
  3. Splits the training data into N disjoint datasubsets and builds one node per subset.
  4. Builds a tiny CNN, saves it as ``.keras`` and registers it.
  5. Lets the model travel through the nodes for a few cycles, evaluating after each node.
  6. Writes a Markdown report to ``<workdir>/reports/``.

Run from the repository root:

    python examples/quickstart.py --workdir xtm_demo --nodes 3 --cycles 2 --epochs 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def prepare_pneumoniamnist(datasets_dir: str, cache_dir: str) -> str:
    """Download PneumoniaMNIST and convert it to the ``<split>_images/labels.npy`` layout."""
    import medmnist

    out_dir = os.path.join(datasets_dir, "pneumoniamnist")
    if os.path.exists(os.path.join(out_dir, "test_images.npy")):
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        ds = medmnist.PneumoniaMNIST(split=split, download=True, root=cache_dir, size=28)
        np.save(os.path.join(out_dir, f"{split}_images.npy"), ds.imgs)
        np.save(os.path.join(out_dir, f"{split}_labels.npy"), ds.labels)
    return out_dir


def build_model(path: str, n_classes: int = 2) -> None:
    """Create and save a small CNN whose penultimate layer is a 32-unit dense layer."""
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Reshape((28, 28, 1)),
        keras.layers.Rescaling(1.0 / 255),
        keras.layers.Conv2D(8, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(16, 3, activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default="xtm_demo", help="directory for data/, reports/ and system_state/")
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--explain", action="store_true", help="also create UMAP feature-space plots (slower)")
    args = parser.parse_args()

    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)  # XTM stores data/, reports/ and system_state/ relative to the CWD

    from classes.data import DataController
    from classes.scenarios import ScenarioController
    from classes.training import TrainingController

    dc = DataController()
    dataset_dir = prepare_pneumoniamnist(dc.datasets_path, cache_dir=os.path.join(workdir, "data", "downloads"))
    dc.add_dataset("pneumonia", dataset_dir)

    dc.create_datasubsets_from_dataset("pneumonia", n_subsets=args.nodes, split_method="uniform",
                                       data_subsets_name="pneu")
    scenario_nodes = []
    for i in range(args.nodes):
        node_name = f"hospital_{i}"
        dc.add_node(node_name, f"Simulated institution {i}")
        dc.add_subset_to_node(datasubset_name=f"pneu_subset_{i}", node_name=node_name)
        scenario_nodes.append(node_name)

    sc = ScenarioController(dc)
    sc.create_scenario("demo", "Sequential traversal over all nodes")
    for node_name in scenario_nodes:
        sc.add_node_to_scenario("demo", node_name)
    sc.set_active_scenario("demo")

    os.makedirs(os.path.join("data", "models"), exist_ok=True)
    model_path = os.path.join("data", "models", "demo_cnn.keras")
    build_model(model_path)

    tc = TrainingController(sc)
    tc.add_model("demo_cnn", model_path, "Tiny CNN for the quickstart")
    tc.set_active_model("demo_cnn")
    tc.set_experiment_name("quickstart")
    tc.evaluator.set_active_metrics(["accuracy", "recall", "precision", "auc", "f1_score"])
    tc.configure_evaluation_timings(["post-node"])
    if args.explain:
        tc.configure_explanation({"method": "umap", "layer_index": -2, "n_neighbors": 15,
                                  "min_dist": 0.1, "n_epochs": 100, "data_fraction": 0.5})
        tc.configure_explanation_timings(["post-cycle"])

    for _ in range(args.cycles):
        if not tc.train_scenario_cycle(epochs=args.epochs, batch_size=32):
            break

    tc.generate_markdown_report("quickstart")
    print(f"\nDone. Open {os.path.join(workdir, 'reports', 'quickstart.md')}")


if __name__ == "__main__":
    main()
