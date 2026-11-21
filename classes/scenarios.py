import os
import pickle
from classes.data import DataController, Node

class Scenario:
    """
    While the data module is concerned with the arrangement of data into logical units (Datasubsets) and 
    the creation of Nodes (which represent data silos), the materialization of those nodes in a concrete setting 
    is done in this module.
    
    Hence, a scenario is a given setting in which training can occur: It contains a set of data silos (Nodes), 
    the order in which they shall be traversed, and any other information that may pertain to that. 
    
    The construction of a scenario is necessary so that the training module can have a simplified interface with the
    data, i.e. instead of managing nodes and datasubsets, it can just make "get_next_node", "train_in_current_node",
    etc.
    """

    SUPPORTED_TRAVERSAL_ORDERS = ['sequential', 'random']

    def __init__(self, name: str, description: str, data_controller: DataController, traversal_order: str = 'sequential'):

        if traversal_order not in self.SUPPORTED_TRAVERSAL_ORDERS:
            raise ValueError(f'Traversal order "{traversal_order}" not supported. Choose from: {self.SUPPORTED_TRAVERSAL_ORDERS}')

        self.name = name
        self.description = description
        self.traversal_order = traversal_order

        # Nodes registered in this scenario, stored as an ordered list to define traversal sequence
        self.nodes: list[Node] = []

        # Pointer to the currently active node during a training run
        self._current_node_index: int = 0

        # Snapshot of node names sourced from the data controller at construction time
        self._node_names_at_build: list[str] = list(data_controller.nodes.keys())


    def add_node(self, node_name: str, available_nodes: dict[str, Node]):

        if node_name not in available_nodes.keys():
            raise ValueError(f'Node "{node_name}" does not exist in the data controller.')

        node = available_nodes[node_name]

        if node in self.nodes:
            raise ValueError(f'Node "{node_name}" is already part of this scenario.')

        self.nodes.append(node)

    def remove_node(self, node_name: str):

        # Efficient search 
        target = next((n for n in self.nodes if n.name == node_name), None)

        if target is None:
            raise ValueError(f'Node "{node_name}" is not part of this scenario.')

        self.nodes.remove(target)

        # Reset traversal pointer if it has gone out of bounds after removal
        if self._current_node_index >= len(self.nodes):
            self._current_node_index = 0

    def set_node_order(self, ordered_node_names: list[str]):
        """
        Redefines the traversal order by providing an explicit sequence of node names.
        All names must correspond to nodes already added to this scenario.
        """

        current_names = {n.name for n in self.nodes}

        if set(ordered_node_names) != current_names:
            raise ValueError('The provided node list must contain exactly the same nodes already in the scenario.')

        self.nodes = [next(n for n in self.nodes if n.name == name) for name in ordered_node_names]

        self._current_node_index = 0

    def reset(self):
        self._current_node_index = 0

    def get_current_node(self) -> Node:
        """
        Returns the node at the current position in the traversal sequence
        without advancing the pointer.
        """

        if not self.nodes:
            raise RuntimeError('Scenario has no nodes. Add nodes before starting traversal.')

        return self.nodes[self._current_node_index]

    def advance_to_next_node(self) -> bool:
        """
        Advances the traversal pointer to the next node.
        
        Returns 
            True if the pointer was advanced successfully, 
            False if the scenario has been fully traversed and a new cycle has started.
        """

        if self._current_node_index + 1 >= len(self.nodes):
            self._current_node_index = 0
            return False

        self._current_node_index += 1
        return True

    def is_complete(self) -> bool:
        """
        Returns True if all nodes in the scenario have been traversed
        """
        return self._current_node_index >= len(self.nodes) - 1

    def verify_integrity(self, available_nodes: dict[str, Node]):
        """
        Verifies that all nodes referenced by this scenario still exist in the
        data controller. Nodes that have been removed externally are dropped
        from the scenario with a warning.
        Also re-links the node references to the ones currently managed by the
        data controller.
        """

        invalid_nodes = []
        updated_nodes = []

        for node in self.nodes:
            if node.name not in available_nodes.keys():
                print(f'[WARNING] Scenario "{self.name}": node "{node.name}" no longer exists in the data controller and will be removed.')
                invalid_nodes.append(node)
            else:
                updated_nodes.append(available_nodes[node.name])

        self.nodes = updated_nodes

        if self._current_node_index >= len(self.nodes):
            self._current_node_index = 0
    
    def _assert_disjoint_datasets(self):
        """
        Ensure the data contained in this scenario's nodes does not overlap, i.e. that the sets of datasubsets are disjoint  
        """

        already_present_sets = set()

        for node in self.nodes:

            node_datasubsets_set = set(node.datasets.keys())

            if node_datasubsets_set & already_present_sets != set():
                raise Exception('The nodes in this scenario contain repeated datasubsets across them!')
            
            already_present_sets = already_present_sets | node_datasubsets_set
        
        return True
        



    def __str__(self) -> str:
        node_lines = '\n'.join(
            [f'        {">>>" if i == self._current_node_index else "   "} [{i}] {n.name}'
             for i, n in enumerate(self.nodes)]
        )
        return f"""
        Scenario: {self.name}
        Description: {self.description}
        Traversal order: {self.traversal_order}
        Current node index: {self._current_node_index}
        Nodes ({len(self.nodes)}):
{node_lines}
        """



class ScenarioController:
    """
    Manages the lifecycle of Scenario objects: creation, persistence, integrity
    verification, and selection of the active scenario used by the training module.

    Responsibilities
    ----------------
    - CRUD operations over the pool of known scenarios.
    - Saving and loading all scenario state to / from disk (scenarios are never
      responsible for their own persistence).
    - Integrity verification that delegates to each Scenario and prunes stale ones.
    - A single "active scenario" pointer so the training module always has a
      well-defined, unambiguous scenario to work with.
    - Interface methods whose signatures mirror the command-dispatch pattern in
      main.py, keeping the controller easy to wire up there.
    """

    def __init__(self, data_controller: DataController, base_path: str = './data/', scenario_dir_name: str = 'scenarios'):

        self.base_path = base_path
        self.scenarios_path = os.path.join(base_path, scenario_dir_name)

        self._data_controller = data_controller

        # Registry of all known scenarios keyed by name
        self.scenarios: dict[str, Scenario] = {}

        # Name of the currently selected scenario; None until explicitly set
        self._active_scenario_name: str | None = None

    def __verify_path_integrity(self):
        """Ensures the scenarios directory exists on disk."""

        if not os.path.isdir(self.base_path):
            os.mkdir(self.base_path)

        if not os.path.isdir(self.scenarios_path):
            os.mkdir(self.scenarios_path)

    def _get_scenario(self, scenario_name: str) -> Scenario:
        
        if scenario_name not in self.scenarios:
            raise ValueError(f'Scenario "{scenario_name}" does not exist.')

        return self.scenarios[scenario_name]

    def verify_integrity(self):
        """
        Runs integrity checks on paths and scenarios.
        Scenarios that have no nodes generate a warning but are kept.
        """

        self.__verify_path_integrity()

        for scenario_name, scenario in list(self.scenarios.items()):
            scenario.verify_integrity(available_nodes=self._data_controller.nodes)

            if not scenario.nodes:
                print(f'[WARNING] Scenario "{scenario_name}" has no valid nodes')

        # If the active scenario was left node-less or got removed externally, deactivate it
        if self._active_scenario_name and self._active_scenario_name not in self.scenarios:
            print(f'[WARNING] Active scenario "{self._active_scenario_name}" is no longer registered. Deactivating.')
            self._active_scenario_name = None

        print('Scenario integrity verified.')

    def save_all_data(self, file_path):

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: str = './scenario_state.pkl') -> 'ScenarioController':

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def create_scenario(self, name: str, description: str, traversal_order: str = 'sequential'):

        if name in self.scenarios:
            raise ValueError(f'A scenario named "{name}" already exists.')

        new_scenario = Scenario(
            name=name,
            description=description,
            data_controller=self._data_controller,
            traversal_order=traversal_order
        )

        self.scenarios[name] = new_scenario
        print(f'Scenario "{name}" created.')
    
    def edit_scenario(self, old_name: str, new_name: str, new_description: str):
        if old_name not in self.scenarios:
            raise NameError(f'Scenario "{old_name}" does not exist.')
        
        if new_name != old_name and new_name in self.scenarios:
            raise ValueError(f'A scenario named "{new_name}" already exists.')

        scenario = self.scenarios.pop(old_name)
        scenario.name = new_name
        if new_description is not None:
            scenario.description = new_description
            
        self.scenarios[new_name] = scenario
        
        if self._active_scenario_name == old_name:
            self._active_scenario_name = new_name
            
        print(f'Scenario "{old_name}" updated to "{new_name}".')
    
    def remove_scenario(self, scenario_name: str):

        self._get_scenario(scenario_name) 
        del self.scenarios[scenario_name]

        if self._active_scenario_name == scenario_name:
            self._active_scenario_name = None
            print(f'[INFO] Active scenario "{scenario_name}" was removed. No scenario is currently active.')

        print(f'Scenario "{scenario_name}" removed.')

    def add_node_to_scenario(self, scenario_name: str, node_name: str):
        
        scenario = self._get_scenario(scenario_name)
        scenario.add_node(node_name, available_nodes=self._data_controller.nodes)
        print(f'Node "{node_name}" added to scenario "{scenario_name}".')

    def remove_node_from_scenario(self, scenario_name: str, node_name: str):

        scenario = self._get_scenario(scenario_name)
        scenario.remove_node(node_name)
        print(f'Node "{node_name}" removed from scenario "{scenario_name}".')

    def set_node_order(self, scenario_name: str, ordered_node_names: list[str]):

        scenario = self._get_scenario(scenario_name)
        scenario.set_node_order(ordered_node_names)
        print(f'Node order updated for scenario "{scenario_name}".')

    def set_active_scenario(self, scenario_name: str):

        scenario = self._get_scenario(scenario_name)

        if not scenario.nodes:
            raise RuntimeError(f'Scenario "{scenario_name}" has no nodes and cannot be activated.')
        
        if scenario._assert_disjoint_datasets():
            print('[INFO] Disjoint datasets across nodes in this scenario were verified')

        self._active_scenario_name = scenario_name
        scenario.reset()
        print(f'Scenario "{scenario_name}" is now active.')

    def get_active_scenario(self) -> Scenario:

        if self._active_scenario_name is None:
            raise RuntimeError('No active scenario. Call set_active_scenario() first.')

        return self._get_scenario(self._active_scenario_name)

    def get_current_node(self):
        """
        Returns the node at the current position of the active scenario.
        """
        return self.get_active_scenario().get_current_node()

    def advance_to_next_node(self) -> bool:
        """
        Advances the active scenario's traversal pointer to the next node.

        Returns
            True if the pointer was advanced; 
            False if all nodes have been visited.
            
        """

        return self.get_active_scenario().advance_to_next_node()

    def is_scenario_complete(self) -> bool:

        return self.get_active_scenario().is_complete()

    def reset_active_scenario(self):

        self.get_active_scenario().reset()
        print(f'Scenario "{self._active_scenario_name}" reset to the first node.')

    def get_dataset(self, dataset_name:str):
        """
        Gets a dataset from the data controller and returns it. Used by the training controller. 
        """
        return self._data_controller.get_dataset(dataset_name)

    def __str__(self) -> str:
        scenario_lines = '\n'.join(
            [f'        {"[ACTIVE]" if name == self._active_scenario_name else "        "} {name}'
             for name in self.scenarios]
        ) or '        (none)'

        return f"""
        ScenarioController
        Base path:       {self.base_path}
        Scenarios path:  {self.scenarios_path}
        Active scenario: {self._active_scenario_name or '(none)'}
        ===== SCENARIOS =====
{scenario_lines}
        ===== DETAILED SCENARIOS=====
        {'\n\n'.join([self.scenarios[scenario_name].__str__() for scenario_name in self.scenarios])}
        """
