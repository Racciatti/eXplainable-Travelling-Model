import pytest

from classes.data import DataController
from classes.scenarios import ScenarioController


@pytest.fixture
def scenarios(tmp_path, tiny_medmnist):
    dc = DataController(base_path=str(tmp_path / "data") + "/")
    dc.add_dataset("tiny", str(tiny_medmnist))
    dc.create_datasubsets_from_dataset("tiny", n_subsets=3, data_subsets_name="u")
    for i in range(3):
        dc.add_node(f"n{i}", "node")
        dc.add_subset_to_node(f"u_subset_{i}", f"n{i}")
    sc = ScenarioController(dc, base_path=str(tmp_path / "data") + "/")
    sc.create_scenario("s", "demo")
    for i in range(3):
        sc.add_node_to_scenario("s", f"n{i}")
    return sc


def test_traversal_wraps_around_after_last_node(scenarios):
    scenarios.set_active_scenario("s")
    assert scenarios.get_current_node().name == "n0"
    assert scenarios.advance_to_next_node() is True
    assert scenarios.advance_to_next_node() is True
    assert scenarios.get_current_node().name == "n2"
    assert scenarios.advance_to_next_node() is False  # new cycle starts
    assert scenarios.get_current_node().name == "n0"


def test_set_node_order(scenarios):
    scenarios.set_node_order("s", ["n2", "n0", "n1"])
    scenarios.set_active_scenario("s")
    assert scenarios.get_current_node().name == "n2"


def test_set_node_order_requires_same_nodes(scenarios):
    with pytest.raises(ValueError):
        scenarios.set_node_order("s", ["n0", "n1"])


def test_unsupported_traversal_order_rejected(scenarios):
    with pytest.raises(ValueError):
        scenarios.create_scenario("bad", "x", traversal_order="zigzag")


def test_duplicate_scenario_rejected(scenarios):
    with pytest.raises(ValueError):
        scenarios.create_scenario("s", "again")


def test_overlapping_nodes_cannot_be_activated(scenarios):
    dc = scenarios._data_controller
    dc.add_node("dup", "shares a subset with n0")
    dc.add_subset_to_node("u_subset_0", "dup")
    scenarios.create_scenario("overlap", "x")
    scenarios.add_node_to_scenario("overlap", "n0")
    scenarios.add_node_to_scenario("overlap", "dup")
    with pytest.raises(Exception, match="repeated datasubsets"):
        scenarios.set_active_scenario("overlap")
