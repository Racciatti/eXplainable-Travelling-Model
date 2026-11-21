import numpy as np
import pytest

from classes.data import DataController, Dataset


@pytest.fixture
def controller(tmp_path, tiny_medmnist):
    dc = DataController(base_path=str(tmp_path / "data") + "/")
    dc.add_dataset("tiny", str(tiny_medmnist))
    return dc


def test_controller_creates_storage_dirs(tmp_path):
    dc = DataController(base_path=str(tmp_path / "fresh") + "/")
    assert (tmp_path / "fresh" / "datasets").is_dir()
    assert (tmp_path / "fresh" / "datasubsets").is_dir()
    assert dc.datasets == {}


def test_dataset_reads_medmnist_metadata(tiny_medmnist):
    ds = Dataset("tiny", str(tiny_medmnist))
    assert ds.structure == "medmnist"
    assert ds.n_classes == 2
    # validation split is merged into training
    assert ds.n_examples_train == 80
    assert ds.n_examples_test == 30


def test_dataset_invalid_path_raises(tmp_path):
    with pytest.raises(ValueError):
        Dataset("x", str(tmp_path / "missing"))


def test_uniform_split_is_disjoint_and_complete(controller):
    controller.create_datasubsets_from_dataset("tiny", n_subsets=4, split_method="uniform", data_subsets_name="u")
    assert len(controller.datasubsets) == 4
    total = sum(s.n_examples_train for s in controller.datasubsets.values())
    assert total == 80
    for s in controller.datasubsets.values():
        # test arrays inside subsets are intentionally empty: no test leakage into training
        assert s.n_examples_test == 0
        assert s.n_examples_train == 20


def test_manual_split_respects_proportions(controller):
    controller.create_datasubsets_from_dataset(
        "tiny", n_subsets=2, split_method="manual", data_subsets_name="m", partitions=[0.75, 0.25])
    sizes = sorted(s.n_examples_train for s in controller.datasubsets.values())
    assert sizes == [20, 60]


def test_manual_split_rejects_bad_partitions(controller):
    with pytest.raises(ValueError):
        controller.create_datasubsets_from_dataset(
            "tiny", n_subsets=2, split_method="manual", data_subsets_name="bad", partitions=[0.9, 0.9])


def test_nodes_and_global_test_set_uses_original_test_split(controller):
    controller.create_datasubsets_from_dataset("tiny", n_subsets=2, data_subsets_name="u")
    for i in range(2):
        controller.add_node(f"n{i}", "node")
        controller.add_subset_to_node(f"u_subset_{i}", f"n{i}")
    x, y = controller.get_global_test_set(["n0", "n1"])
    # origin test set counted once, even though two nodes share the same origin dataset
    assert len(x) == len(y) == 30


def test_duplicate_node_is_rejected(controller):
    controller.add_node("a", "first")
    with pytest.raises(ValueError):
        controller.add_node("a", "again")


def test_harmful_datasubset_label_noise(controller):
    controller.create_datasubsets_from_dataset("tiny", n_subsets=2, data_subsets_name="u")
    harmful = controller.create_harmful_datasubset("u_subset_0", "label_noise", {"p": 1.0})
    clean_y = np.load(controller.datasubsets["u_subset_0"].path + "/train_labels.npy")
    noisy_y = np.load(harmful.path + "/train_labels.npy")
    assert clean_y.shape == noisy_y.shape
    assert np.all(clean_y != noisy_y)
