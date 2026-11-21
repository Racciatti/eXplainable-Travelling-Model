"""XTM command-line interface.

An interactive REPL that dispatches text commands to the three XTM controllers
(data, scenario, training). Controller state is pickled to ``./system_state/``
on exit and restored on the next start.
"""
from __future__ import annotations

from classes.data import DataController
from classes.scenarios import ScenarioController
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classes.training import TrainingController

import os
import pickle
import csv


# DATA CALLS HANDLERS
def handle_add_dataset(data_controller:DataController):
    name = input('Dataset name: ')
    path = input('Dataset path: ')
    data_controller.add_dataset(name, path)

def handle_create_subsets(data_controller:DataController):
    dataset_name = input("Name of the dataset to be split: ")
    n_subsets = int(input("Number of subsets: "))
    split_method = input("Split method ('uniform' / 'manual'): ")
    
    if split_method == 'manual':
        partitions_str = input("Enter partitions as comma separated percentages (e.g. 0.5, 0.25, 0.25): ")
        partitions = [float(p.strip()) for p in partitions_str.split(',')]
        data_controller.create_datasubsets_from_dataset(dataset_name=dataset_name, n_subsets=n_subsets, split_method=split_method, partitions=partitions)
    else:
        data_controller.create_datasubsets_from_dataset(dataset_name=dataset_name, n_subsets=n_subsets, split_method=split_method)

def handle_create_harmful_datasubset(data_controller: DataController):
    source_name = input("Source datasubset name: ")
    print("Available techniques: label_noise, gaussian_blur")
    technique = input("Technique: ").strip()
    
    params = {}
    if technique == 'label_noise':
        p_str = input("Noise probability p (0.0 to 1.0): ")
        params['p'] = float(p_str)
    elif technique == 'gaussian_blur':
        k_str = input("Kernel size (odd integer >= 3): ")
        params['kernel_size'] = int(k_str)
    
    data_controller.create_harmful_datasubset(source_name, technique, params)

def handle_create_node(data_controller:DataController):
    new_node_name = input('Node name: ')
    new_node_description = input('Node description: ')
    data_controller.add_node(new_node_name, new_node_description)
        
def handle_associate_subset_to_note(data_controller:DataController):
    node_name = input('Node name: ')
    subset_name = input('Subset name: ')
    data_controller.add_subset_to_node(datasubset_name=subset_name, node_name=node_name)

def handle_remove_node(data_controller:DataController):

    node_name = input('Node name: ')
    data_controller.remove_node(node_name=node_name)

def handle_remove_dataset(data_controller:DataController):
    dataset_name = input('Dataset name: ')
    data_controller.remove_dataset(dataset_name=dataset_name)

def handle_remove_datasubset(data_controller:DataController):
    subset_name = input('Datasubset name: ')
    data_controller.remove_datasubset(datasubset_name=subset_name)

def handle_remove_datasubset_from_node(data_controller:DataController):
    node_name = input('Node name: ')
    subset_name = input('Datasubset name: ')
    data_controller.remove_datasubset_from_node(node_name=node_name, datasubset_name=subset_name)
    
def handle_edit_dataset(data_controller:DataController):
    old_name = input('Old dataset name: ')
    new_name = input('New dataset name: ')
    desc = input('New description (leave empty to keep current): ')
    data_controller.edit_dataset(old_name, new_name, desc if desc else None)

def handle_edit_datasubset(data_controller:DataController):
    old_name = input('Old datasubset name: ')
    new_name = input('New datasubset name: ')
    desc = input('New description (leave empty to keep current): ')
    data_controller.edit_datasubset(old_name, new_name, desc if desc else None)

def handle_edit_node(data_controller:DataController):
    old_name = input('Old node name: ')
    new_name = input('New node name: ')
    desc = input('New description (leave empty to keep current): ')
    data_controller.edit_node(old_name, new_name, desc if desc else None)

# SCENARIO CALLS HANDLERS
def handle_create_scenario(sc: ScenarioController):
    name            = input('Scenario name: ')
    description     = input('Scenario description: ')
    traversal_order = input("Traversal order ('sequential' / 'random'): ").strip() or 'sequential'
    sc.create_scenario(name, description, traversal_order)

def handle_remove_scenario(sc: ScenarioController):
    name = input('Scenario name: ')
    sc.remove_scenario(name)

def handle_add_node_to_scenario(sc: ScenarioController):
    scenario_name = input('Scenario name: ')
    node_name     = input('Node name: ')
    sc.add_node_to_scenario(scenario_name, node_name)

def handle_remove_node_from_scenario(sc: ScenarioController):
    scenario_name = input('Scenario name: ')
    node_name     = input('Node name: ')
    sc.remove_node_from_scenario(scenario_name, node_name)

def handle_set_active_scenario(sc: ScenarioController):
    name = input('Scenario name to activate: ')
    sc.set_active_scenario(name)

def handle_edit_scenario(sc: ScenarioController):
    old_name = input('Old scenario name: ')
    new_name = input('New scenario name: ')
    desc = input('New description (leave empty to keep current): ')
    sc.edit_scenario(old_name, new_name, desc if desc else None)


def handle_set_node_order(sc:ScenarioController):

    scenario_name = input('Scenario name: ')

    scenario = sc._get_scenario(scenario_name)

    node_names = ', '.join(n.name for n in scenario.nodes)
    print(f'Nodes in the scenario: [{node_names}]')

    new_order = [n.strip() for n in input('Name the nodes in order, separated by commas: ').split(',')]

    sc.set_node_order(scenario_name, new_order)
    


# TRAINING CALLS HANDLERS

def handle_add_model(tc: TrainingController):
    name = input('Model name: ')
    model_path = input('Model path (Keras/TF format): ')
    description = input('Model description: ')
    tc.add_model(name=name, model_path=model_path, description=description)

def handle_remove_model(tc: TrainingController):
    name = input('Model name: ')
    tc.remove_model(name)

def handle_set_active_model(tc: TrainingController):
    name = input('Model name to activate: ')
    tc.set_active_model(name)

def handle_reset_active_model_weights(tc: TrainingController):
    ans = input("Revert weights to initial state from .keras file? (y/n, default 'n' for randomization): ")
    revert_to_file = ans.strip().lower() == 'y'
    tc.reset_active_model_weights(revert_to_file=revert_to_file)

def handle_get_architecture(tc: TrainingController):
    model = tc.get_active_model()
    print(model.get_architecture())

def handle_edit_model(tc: TrainingController):
    old_name = input('Old model name: ')
    new_name = input('New model name: ')
    desc = input('New description (leave empty to keep current): ')
    tc.edit_model(old_name, new_name, desc if desc else None)

def handle_train_current_node(tc: TrainingController):
    epochs_str = input("Epochs [default 1]: ").strip()
    epochs = int(epochs_str) if epochs_str.isdigit() else 1
    
    batch_size_str = input("Batch size [default 32]: ").strip()
    batch_size = int(batch_size_str) if batch_size_str.isdigit() else 32
    
    while True:
        report_name = input("Enter experiment/report name (without .md): ").strip()
        if not report_name:
            print("Name cannot be empty.")
            continue
        break
            
    tc.set_experiment_name(report_name)
    
    print(f"[INFO] Starting training phase for {epochs} epoch(s) with batch size {batch_size}...")
    tc.train_current_node(epochs=epochs, batch_size=batch_size)
    print("[INFO] Training on current node completed.")

def handle_train_scenario_cycle(tc: TrainingController):
    epochs_str = input("Epochs per node [default 1]: ").strip()
    epochs = int(epochs_str) if epochs_str.isdigit() else 1
    
    batch_size_str = input("Batch size per node [default 32]: ").strip()
    batch_size = int(batch_size_str) if batch_size_str.isdigit() else 32
    
    while True:
        report_name = input("Enter experiment/report name (without .md): ").strip()
        if not report_name:
            print("Name cannot be empty.")
            continue
        break
            
    tc.set_experiment_name(report_name)

    eda_ans = input("Generate EDA charts for this report? [y/N]: ").strip().lower()
    tc.set_eda_enabled(eda_ans == 'y')

    print("[INFO] Starting full scenario training cycle...")
    tc.train_scenario_cycle(epochs, batch_size)

def handle_train_n_scenario_cycles(tc: TrainingController):
    cycles_str = input("Number of cycles [default 1]: ").strip()
    cycles = int(cycles_str) if cycles_str.isdigit() else 1

    epochs_str = input("Epochs per node [default 1]: ").strip()
    epochs = int(epochs_str) if epochs_str.isdigit() else 1
    
    batch_size_str = input("Batch size per node [default 32]: ").strip()
    batch_size = int(batch_size_str) if batch_size_str.isdigit() else 32
    
    while True:
        report_name = input("Enter experiment/report name (without .md): ").strip()
        if not report_name:
            print("Name cannot be empty.")
            continue
        
        report_path = os.path.join("reports", f"{report_name}.md")
        if os.path.exists(report_path):
            print(f"File '{report_path}' already exists. Please choose a different name.")
        else:
            break
            
    tc.set_experiment_name(report_name)
    
    eda_ans = input("Generate EDA charts for this report? [y/N]: ").strip().lower()
    tc.set_eda_enabled(eda_ans == 'y')
    
    print(f"[INFO] Starting {cycles} full scenario training cycles...")
    # Reset scenario, model, and eval history
    tc.clear_training_and_evaluation_history()
    tc.reset_active_model_weights(revert_to_file=True)
    tc.reset_scenario()

    try:
        for i in range(1, cycles + 1):
            print(f"\n[INFO] ------------------ Starting Cycle {i}/{cycles} ------------------")
            success = tc.train_scenario_cycle(epochs, batch_size)
            if not success:
                print("[ERROR] Aborting remaining cycles.")
                break
        print("[INFO] Finished all scheduled cycles.")
    except Exception as e:
        print(f"[ERROR] Training aborted unexpectedly: {e}")
    finally:
        prompt_report = input("\nGenerate a final Markdown report? (y/n, default 'y'): ").strip().lower()
        if prompt_report != 'n':
            try:
                tc.generate_markdown_report(report_name)
            except Exception as e:
                print(f"[ERROR] Failed to generate report: {e}")

def handle_run_from_file(tc: TrainingController):
    eda_ans = input("Generate EDA charts for reports? [y/N]: ").strip().lower()
    tc.set_eda_enabled(eda_ans == 'y')
    file_path = input("Enter the path to the CSV file: ").strip()
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return
        
    experiments = []
    seen_reports = set()
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            required_cols = ['cycles', 'epochs', 'scenario', 'model', 'report_name']
            if not reader.fieldnames:
                print("[ERROR] CSV file is empty or missing headers.")
                return
                
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                print(f"[ERROR] CSV missing required columns: {', '.join(missing_cols)}")
                return
                
            for i, row in enumerate(reader, start=2):
                try:
                    cycles = int(row['cycles'])
                    if cycles < 1: raise ValueError("cycles must be >= 1")
                    
                    epochs = int(row['epochs'])
                    if epochs < 1: raise ValueError("epochs must be >= 1")
                    
                    scenario_name = row['scenario'].strip()
                    if scenario_name not in tc.scenario_controller.scenarios:
                        raise ValueError(f"Scenario '{scenario_name}' not found.")
                        
                    model_name = row['model'].strip()
                    if model_name not in tc.models:
                        raise ValueError(f"Model '{model_name}' not found.")
                        
                    report_name = row['report_name'].strip()
                    if not report_name:
                        raise ValueError("Report name cannot be empty.")
                        
                    report_path = os.path.join("reports", f"{report_name}.md")
                    if os.path.exists(report_path) or report_name in seen_reports:
                        raise ValueError(f"Report file '{report_name}' already exists or is duplicated in the CSV.")
                    seen_reports.add(report_name)
                        
                    experiments.append({
                        'cycles': cycles,
                        'epochs': epochs,
                        'scenario': scenario_name,
                        'model': model_name,
                        'report_name': report_name,
                        'row_idx': i
                    })
                except Exception as e:
                    print(f"[ERROR] Validation failed on row {i}: {e}")
                    return
    except Exception as e:
        print(f"[ERROR] Failed to read or parse file: {e}")
        return

    if not experiments:
        print("[INFO] No valid experiments found in file.")
        return
        
    print(f"[INFO] File validated completely. {len(experiments)} experiment(s) ready to run.")
    confirm = input("Do you want to proceed with execution? (y/n): ").strip().lower()
    if confirm != 'y':
        print("[INFO] Execution aborted by user.")
        return
        
    for exp in experiments:
        print("\n=======================================================")
        print(f"[INFO] RUNNING EXPERIMENT {exp['row_idx'] - 1} OF {len(experiments)}")
        print(f"[INFO] Scenario: {exp['scenario']}, Model: {exp['model']}, Cycles: {exp['cycles']}, Epochs/Cycle: {exp['epochs']}")
        print("=======================================================")
        
        try:
            # Set metavariables for the experiment
            tc.scenario_controller.set_active_scenario(exp['scenario'])
            tc.set_active_model(exp['model'])
            tc.set_experiment_name(exp['report_name'])
            
            # To isolate experiments, reverting weights to initial state is conceptually right.
            # Assuming file allows reverting back to original. 
            tc.reset_active_model_weights(revert_to_file=True)
            
            # Prepare fresh state for this experiment
            tc.reset_scenario()
            tc.clear_training_and_evaluation_history()
            
            # Execution loop
            cycles = exp['cycles']
            epochs = exp['epochs']
            for i in range(1, cycles + 1):
                print(f"\n[INFO] --- Starting Cycle {i}/{cycles} ---")
                success = tc.train_scenario_cycle(epochs=epochs, batch_size=32)
                if not success:
                    print(f"[ERROR] Cycle {i} failed. Aborting experiment.")
                    break
                    
            print(f"[INFO] Finished training cycles for experiment {exp['row_idx'] - 1}.")
            
            tc.generate_markdown_report(exp['report_name'])
            print(f"[INFO] Created report: {exp['report_name']}.md")
            
        except Exception as e:
            print(f"[ERROR] Experiment at row {exp['row_idx']} aborted unexpectedly: {e}")
            print("[INFO] Proceeding to the next experiment if any.")
            continue
            
    print("\n[INFO] All experiments from file completed.")

def handle_configure_evaluation(tc: TrainingController):
    print("Supported metrics: Accuracy, Recall, Precision, AUC, F1, FBeta, NPV")
    metrics_str = input("Select metrics to evaluate (comma separated): ").strip()
    metrics_list = [m.strip().lower() for m in metrics_str.split(',') if m.strip()]
    
    display_metrics = []
    
    for metric in metrics_list:
        if metric in ['accuracy', 'acc']:
            display_metrics.append('accuracy')
        elif metric in ['recall', 'sensitivity']:
            display_metrics.append('recall')
        elif metric in ['precision', 'ppv']:
            display_metrics.append('precision')
        elif metric == 'auc':
            display_metrics.append('auc')
        elif metric == 'f1':
            display_metrics.append('f1_score')
        elif metric == 'fbeta':
            beta_str = input("Enter Beta value for FBetaScore [default 1.0]: ").strip()
            beta = float(beta_str) if beta_str else 1.0
            display_metrics.append(f'fbeta_{beta}')
        elif metric == 'npv':
            display_metrics.append('npv')
        else:
            print(f"[WARNING] Unknown metric '{metric}' ignored.")

    print("Supported timings: pre-node, post-node, pre-cycle, post-cycle")
    timings_str = input("Evaluate when? (comma separated): ").strip()
    timings_list = [t.strip().lower() for t in timings_str.split(',') if t.strip()]

    valid_timings = []
    for t in timings_list:
        if t in ['pre-node', 'post-node', 'pre-cycle', 'post-cycle']:
            valid_timings.append(t)
        else:
            print(f"[WARNING] Unknown timing '{t}' ignored.")

    tc.evaluator.set_active_metrics(display_metrics)
    tc.configure_evaluation_timings(valid_timings)
    tc._metrics_compiled = False
    
    print("[INFO] Evaluation configuration saved. Metrics will be dynamically compiled when executing evaluation operations.")

def handle_view_metrics(tc: TrainingController):
    metrics = tc.evaluator.active_metrics_config
    timings = tc.evaluation_timings
    
    print("\n=== Evaluation Configuration ===")
    if metrics:
        print(f"Active Metrics: {', '.join(metrics)}")
    else:
        print("Active Metrics: None configured (using default Keras metrics)")
        
    if timings:
        print(f"Extraction Timings: {', '.join(timings)}")
    else:
        print("Extraction Timings: None configured (evaluations will not trigger automatically)")
    print("================================\n")

def handle_explain_current_node(tc: TrainingController):
    print("For CLI demonstration, this naturally requires loading your data to pass as 'node_x' and 'node_y'.")
    print("In a real environment, this extracts activations and plots them.")
    current_node = tc.scenario_controller.get_current_node()
    active_model = tc.get_active_model()
    
    method = input("Projection method (tsne/umap) [default tsne]: ").strip() or 'tsne'
    print(f"Ready to explain model '{active_model.name}' activations on node '{current_node.name}' using {method.upper()}.")
    # Example direct call: tc.explain_current_node(node_x, node_y, method=method)

def handle_configure_explanation(tc: TrainingController):
    print("=== Configure Explanation Feature Space ===")
    method = input("Projection method (umap/tsne) [default umap]: ").strip().lower() or 'umap'
    layer_index_str = input("Layer index [default -2]: ").strip()
    layer_index = int(layer_index_str) if layer_index_str.lstrip('-').isdigit() else -2
    
    config_dict = {
        'method': method,
        'layer_index': layer_index,
    }
    
    if method == 'umap':
        n_neighbors_str = input("n_neighbors [default 50]: ").strip()
        config_dict['n_neighbors'] = int(n_neighbors_str) if n_neighbors_str.isdigit() else 50
        
        min_dist_str = input("min_dist [default 0.1]: ").strip()
        try:
            config_dict['min_dist'] = float(min_dist_str) if min_dist_str else 0.1
        except ValueError:
            config_dict['min_dist'] = 0.1
            
        n_epochs_str = input("n_epochs [default 300]: ").strip()
        config_dict['n_epochs'] = int(n_epochs_str) if n_epochs_str.isdigit() else 300
    
    data_fraction_str = input("Data fraction to use (0.0 to 1.0) [default 1.0]: ").strip()
    try:
        config_dict['data_fraction'] = float(data_fraction_str) if data_fraction_str else 1.0
    except ValueError:
        config_dict['data_fraction'] = 1.0
        
    tc.configure_explanation(config_dict)

def handle_set_explanation_timings(tc: TrainingController):
    print("Supported timings: pre-node, post-node, pre-cycle, post-cycle")
    timings_str = input("Generate visualizations when? (comma separated, empty to disable): ").strip()
    
    if timings_str:
        timings_list = [t.strip().lower() for t in timings_str.split(',') if t.strip()]
        valid_timings = [t for t in timings_list if t in ['pre-node', 'post-node', 'pre-cycle', 'post-cycle']]
    else:
        valid_timings = []
        
    tc.configure_explanation_timings(valid_timings)

def handle_view_explanation_config(tc: TrainingController):
    print("\n=== Explanation Configuration ===")
    print(f"Timings: {tc.explanation_timings}")
    print("Config:")
    for k, v in tc.explanation_config.items():
        print(f"  {k}: {v}")
    print("=================================\n")

def handle_advance_scenario(tc: TrainingController):
    advanced = tc.advance_scenario()
    node = tc.get_current_node()
    if advanced:
        print(f"Advanced scenario. Current node: {node.name}")
    else:
        print(f"Cycle completed. Starting a new cycle. Current node: {node.name}")

def handle_reset_scenario(tc: TrainingController):
    tc.reset_scenario()

# Map commands to their handler functions
DATA_COMMANDS = {
    'ad': handle_add_dataset,
    'add dataset': handle_add_dataset,
    
    'create subsets': handle_create_subsets,
    'cs': handle_create_subsets,
    
    'create harmful datasubset': handle_create_harmful_datasubset,
    'chds': handle_create_harmful_datasubset,
    
    'create node': handle_create_node,
    'cn': handle_create_node,
    
    'add subset to node': handle_associate_subset_to_note,
    'as': handle_associate_subset_to_note,

    'remove node': handle_remove_node,
    'rn': handle_remove_node,

    'remove dataset': handle_remove_dataset,
    'rd': handle_remove_dataset,

    'remove datasubset': handle_remove_datasubset,
    'rds': handle_remove_datasubset,

    'remove datasubset from node': handle_remove_datasubset_from_node,
    'rdsfn': handle_remove_datasubset_from_node,

    'edit dataset': handle_edit_dataset,
    'edd': handle_edit_dataset,

    'edit datasubset': handle_edit_datasubset,
    'edds': handle_edit_datasubset,

    'edit node': handle_edit_node,
    'edn': handle_edit_node,

    'view data': lambda c: print(c),
    'vd': lambda c: print(c),
}

SCENARIO_COMMANDS ={
    'create scenario' : handle_create_scenario,
    'csc'             : handle_create_scenario,

    'remove scenario' : handle_remove_scenario,
    'rsc'             : handle_remove_scenario,

    'add node to scenario'      : handle_add_node_to_scenario,
    'ans'                       : handle_add_node_to_scenario,

    'remove node from scenario' : handle_remove_node_from_scenario,
    'rns'                       : handle_remove_node_from_scenario,

    'set active scenario' : handle_set_active_scenario,
    'sas'                 : handle_set_active_scenario,

    'edit scenario'       : handle_edit_scenario,
    'edsc'                : handle_edit_scenario,

    'view scenarios' : lambda sc: print(sc),
    'vs' : lambda sc: print(sc),

    'set node order' : handle_set_node_order,
    'sno' : handle_set_node_order
}

TRAINING_COMMANDS = {
    'add model'            : handle_add_model,
    'am'                   : handle_add_model,

    'remove model'         : handle_remove_model,
    'rm'                   : handle_remove_model,

    'edit model'           : handle_edit_model,
    'edm'                  : handle_edit_model,

    'set active model'     : handle_set_active_model,
    'sam'                  : handle_set_active_model,

    'reset model weights'  : handle_reset_active_model_weights,
    'rmw'                  : handle_reset_active_model_weights,

    'get architecture'     : handle_get_architecture,
    'ga'                   : handle_get_architecture,

    'train current node'   : handle_train_current_node,
    'tcn'                  : handle_train_current_node,

    'train scenario cycle' : handle_train_scenario_cycle,
    'tsc'                  : handle_train_scenario_cycle,

    'train n scenario cycles' : handle_train_n_scenario_cycles,
    'tnc'                     : handle_train_n_scenario_cycles,

    'run from file'           : handle_run_from_file,
    'rff'                     : handle_run_from_file,

    'configure evaluation'    : handle_configure_evaluation,
    'ce'                      : handle_configure_evaluation,

    'view metrics'            : handle_view_metrics,
    'vm'                      : handle_view_metrics,
    
    'configure explanation'   : handle_configure_explanation,
    'cexp'                    : handle_configure_explanation,
    
    'set explanation timings' : handle_set_explanation_timings,
    'set'                     : handle_set_explanation_timings,
    
    'view explanation config' : handle_view_explanation_config,
    'vec'                     : handle_view_explanation_config,

    'explain current node' : handle_explain_current_node,
    'ecn'                  : handle_explain_current_node,

    'advance scenario'     : handle_advance_scenario,
    'adv'                  : handle_advance_scenario,

    'reset scenario'       : handle_reset_scenario,
    'rs'                   : handle_reset_scenario,

    'view training'  : lambda tc: print(tc),
    'vt' : lambda sc: print(sc),
}

# Map commands to the controllers
CONTROLLER_FOR_COMMAND = {
    command: 'data_controller' if DATA_COMMANDS.get(command, None) else ('scenario_controller' if SCENARIO_COMMANDS.get(command, None) else 'training_controller')
    for command in list((DATA_COMMANDS | SCENARIO_COMMANDS | TRAINING_COMMANDS).keys())    
}


def verify_files_integrity(state_dir:str, data_state_filename:str, scenario_state_filename:str, training_state_filename:str):

    if not os.path.isdir(state_dir):
        print(f'[INFO] {state_dir} dir created')
        os.mkdir(state_dir)
    
    if not os.path.exists(f"{state_dir}{data_state_filename}"):
        with open(f"{state_dir}{data_state_filename}", 'wb') as f:
            pickle.dump({}, f)
        f.close()
        print(f'[INFO] {data_state_filename} file created')
    
    if not os.path.exists(f"{state_dir}{scenario_state_filename}"):
        with open(f"{state_dir}{scenario_state_filename}", 'wb') as f:
            pickle.dump({}, f)
        f.close()
        print(f'[INFO] {scenario_state_filename} file created')

    if not os.path.exists(f"{state_dir}{training_state_filename}"):
        with open(f"{state_dir}{training_state_filename}", 'wb') as f:
            pickle.dump({}, f)
        f.close()
        print(f'[INFO] {training_state_filename} file created')

    print('[INFO] state data verified')

def load_controllers(state_dir:str, data_state_filename:str, scenario_state_filename:str, training_state_filename:str):
    from classes.training import TrainingController

    # Load the controllers (which contain all state data)
    data_controller = DataController.load(f"{state_dir}{data_state_filename}")
    scenario_controller = ScenarioController.load(f"{state_dir}{scenario_state_filename}")
    try:
        training_controller = TrainingController.load_state(f"{state_dir}{training_state_filename}")
    except Exception:
        training_controller = {}

    # If the save file did not exist, create a new controller from scratch
    if data_controller == {}:
        data_controller = DataController()
    # If it already existed, check their integrity
    else:
        print('[INFO] Found existing data controller. Proceeding to integrity check...')
        data_controller.verify_integrity()


    if scenario_controller == {}:
        scenario_controller = ScenarioController(data_controller)
    else:
        print('[INFO] Found existing scenario controller. Proceeding to integrity check...')
        # Re-link the data controller before checking integrity so reference nodes can be updated
        scenario_controller._data_controller = data_controller
        scenario_controller.verify_integrity()
        
    if training_controller == {} or type(training_controller) is dict:
        training_controller = TrainingController(scenario_controller=scenario_controller, base_path=state_dir)
    else:
        print('[INFO] Found existing training controller. Proceeding to integrity check...')
        # Update scenario controller reference to the current valid singleton
        training_controller.scenario_controller = scenario_controller
        training_controller.verify_integrity()
    
    return {
        'data_controller' : data_controller,
        'scenario_controller' : scenario_controller,
        'training_controller' : training_controller
        }

if __name__ == '__main__':

    state_dir='./system_state/'
    data_state_filename='data_state.pkl'
    scenario_state_filename='scenario_state.pkl'
    training_state_filename='training_state.pkl'
    data_state_path = state_dir + data_state_filename
    scenario_state_path = state_dir + scenario_state_filename
    
    verify_files_integrity(state_dir=state_dir, data_state_filename=data_state_filename, scenario_state_filename=scenario_state_filename, training_state_filename=training_state_filename)
    controllers = load_controllers(state_dir=state_dir, data_state_filename=data_state_filename, scenario_state_filename=scenario_state_filename, training_state_filename=training_state_filename)
    
    controllers['scenario_controller']._data_controller = controllers['data_controller']

    # Mainloop
    while True:
        try:
            ui = input(': ').lower().strip()
            if ui in ['exit', 'q']: break 
                
            handler = DATA_COMMANDS.get(ui, None) or SCENARIO_COMMANDS.get(ui, None) or TRAINING_COMMANDS.get(ui, None)
            
            if handler:
                handler(controllers[CONTROLLER_FOR_COMMAND[ui]])
            else:
                if ui:
                    print("Unknown command.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
    
    try:
        controllers['data_controller'].save_all_data(data_state_path)   
        controllers['scenario_controller'].save_all_data(scenario_state_path)
        
        file_path = os.path.join(controllers['training_controller'].base_path, training_state_filename)
        with open(file_path, 'wb') as f:
            pickle.dump(controllers['training_controller'], f)
            
        print("[INFO] System state saved explicitly before exit.")
    except Exception as e:
        print(f"[ERROR] Failed to save system state: {e}") 
