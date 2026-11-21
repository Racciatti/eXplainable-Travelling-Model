from classes.scenarios import ScenarioController

import os
import sys
import pickle
import matplotlib

from typing import Any, Dict
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

matplotlib.use('Agg')


# Lazy loading of libraries for quicker software initialization
def _get_tf():
    import tensorflow as tf
    return tf

def _get_keras():
    from tensorflow import keras
    return keras

class Model:
    """
    The model class is responsible for handling models, such as loading and saving them, providing attributes and methods
    that interface with the native model class (.keras, .tf) in order to get the architecture, 
    a description of the architecture, initiate training, reset weights, evaluate the model on a given set of metrics, 
    etc. as well as containing basic information for the system itself, such as model path, name, description, status, etc.
    Everything that is needed on the ML training and testing lifecycle.  
    """
    def __init__(self, name: str, model_path: str, description: str = ""):
        self.name = name
        self.model_path = os.path.abspath(model_path)
        self.description = description
        
        self.keras_model = None
        self.status = "uninitialized"
        self.load()

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        try:
            keras = _get_keras()
            from classes.metrics import F1Score, FBetaScore, NPV
            self.keras_model = keras.models.load_model(self.model_path, custom_objects={"F1Score": F1Score, "FBetaScore": FBetaScore, "NPV": NPV})
            self.status = "loaded"
        except Exception as e:
            self.status = "error"
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'keras_model' in state:
            state['keras_model'] = None  # Do not pickle the underlying keras model object
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.status = 'deferred'
        # We defer loading until explicitly requested or verified by the controller's integrity checks.

    def save_model(self, path: str = None):
        save_path = path if path else self.model_path
        if not self.keras_model:
            raise RuntimeError("No native model to save.")
        
        try:
            self.keras_model.save(save_path)
            self.model_path = os.path.abspath(save_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {save_path}: {e}")

    def get_architecture(self) -> str:
        if not self.keras_model:
            raise RuntimeError("Model is not initialized.")
        string_list = []
        self.keras_model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)

    def compile(self, optimizer, loss, metrics):
        if getattr(self, 'status', None) == 'deferred':
            self.load()
        if not self.keras_model:
            raise RuntimeError("Model is not initialized.")
        self.keras_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x, y, **kwargs):
        if getattr(self, 'status', None) == 'deferred':
            self.load()
        if not self.keras_model:
            raise RuntimeError("Model is not initialized.")
        return self.keras_model.fit(x, y, **kwargs)

    def evaluate(self, x, y, **kwargs):
        if getattr(self, 'status', None) == 'deferred':
            self.load()
        if not self.keras_model:
            raise RuntimeError("Model is not initialized.")
        return self.keras_model.evaluate(x, y, **kwargs)
        
    def get_weights(self):
        if getattr(self, 'status', None) == 'deferred':
            self.load()
        if not self.keras_model:
            raise RuntimeError("Model is not initialized.")
        return self.keras_model.get_weights()
        
    def set_weights(self, weights):
        if getattr(self, 'status', None) == 'deferred':
            self.load()
        if not self.keras_model:
            raise RuntimeError("Model is not initialized.")
        self.keras_model.set_weights(weights)

    def __str__(self):
        return f"""
        Model Name: {self.name}
        Path: {self.model_path}
        Status: {self.status}
        Description: {self.description}
        """

class Evaluator:
    """
    The evaluator class is responsible for evaluating a model. 
    More specifically, given a model and train / test data, it is responsible for managing the testing,
    extracting whatever metrics are requested (accuracy, precision, recall, sensitivity, NPV, FPV, f-1 score, f-beta score, etc.)
    The class shall be able to evaluate the model on any set of supported metrics, register that data, and generate
    a report that summarizes all of the results / stats / metrics.   
    """

    def __init__(self):
        self.evaluation_history = []
        self.active_metrics_config = []

    def set_active_metrics(self, user_metrics_config):
        self.active_metrics_config = user_metrics_config

    def evaluate_step(self, model: Model, test_x, test_y, cycle: int, node_name: str, timing: str, **kwargs) -> Dict[str, Any]:
        """
        Runs the Keras `evaluate()` pass on the provided global testing data and logs the 
        structured dictionary into `evaluation_history`.
        """
        if 'return_dict' not in kwargs:
            kwargs['return_dict'] = True
            
        try:
            results = model.evaluate(test_x, test_y, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error during model evaluation: {e}")
            
        extracted_metrics = {}
        if isinstance(results, dict):
            for name, value in results.items():
                if name == 'compile_metrics': continue # Skip compile_metrics placeholder
                if name == 'recall': extracted_metrics['Sensitivity (Recall)'] = float(value)
                elif name == 'precision': extracted_metrics['PPV (Precision)'] = float(value)
                else: extracted_metrics[name] = float(value)
        else:
            # Fallback for older keras versions returning lists without dict support
            metrics_names = model.keras_model.metrics_names
            for name, value in zip(metrics_names, results):
                if name == 'compile_metrics': continue
                if name == 'recall': extracted_metrics['Sensitivity (Recall)'] = float(value)
                elif name == 'precision': extracted_metrics['PPV (Precision)'] = float(value)
                else: extracted_metrics[name] = float(value)

        evaluation_record = {
            "cycle": cycle,
            "node": node_name,
            "timing": timing,
            "metrics": extracted_metrics
        }

        self.evaluation_history.append(evaluation_record)
        return evaluation_record

    def generate_report(self) -> str:
        if not self.evaluation_history:
            return "No evaluations performed yet."
            
        report = "=== Evaluation Report ===\n"
        for i, record in enumerate(self.evaluation_history):
            report += f"Run {i+1} | Cycle {record.get('cycle', 'N/A')} | Node: {record.get('node', 'N/A')} | Timing: {record.get('timing', 'N/A')}\n"
            for metric, value in record['metrics'].items():
                report += f"  {metric}: {value}\n"
        return report

class Explainer:
    """
    Generates visualizations of a model's feature space using multidimensional 
    projection techniques like t-SNE and UMAP on model activations.
    Highly modular: accepts any model and arrays.
    """
    def __init__(self, base_save_dir: str = './reports/visualizations'):
        self.base_save_dir = base_save_dir
        if not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)

    def extract_activations(self, model: Model, x, layer_index: int = -2):
        """
        Extracts activations from a specified layer of the Keras model.
        Defaults to the second-to-last layer (-2).
        """
        if getattr(model, 'status', None) == 'deferred':
            model.load()
        if not model.keras_model:
            raise RuntimeError("Model is not initialized.")
        
        keras_model = model.keras_model
        try:
            print(f"[EXPLAINER] Extracting activations from layer {layer_index}... (this may take a moment)")
            keras = _get_keras()
            intermediate_model = keras.models.Model(
                inputs=keras_model.inputs,
                outputs=keras_model.layers[layer_index].output
            )
            # Suppress verbose output
            activations = intermediate_model.predict(x, verbose=0)
            
            # Flatten activations if multi-dimensional
            if len(activations.shape) > 2:
                activations = activations.reshape(activations.shape[0], -1)
            
            print(f"[EXPLAINER] Activations extracted. Shape: {activations.shape}")
            return activations
        except Exception as e:
            raise RuntimeError(f"Error extracting activations from layer {layer_index}: {e}")

    def generate_visualization(self, model: Model, x, y, method: str = 'tsne', 
                               layer_index: int = -2, save_filename: str = None, 
                               projection_kwargs: dict = None, plot_kwargs: dict = None):
        """
        Extracts activations, applies multidimensional projection, and generates a scatterplot.
        """
        if projection_kwargs is None:
            projection_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}

        activations = self.extract_activations(model, x, layer_index=layer_index)

        print(f"[EXPLAINER] Running {method.upper()} projection on {activations.shape[0]} samples with {activations.shape[1]} features...")
        if method.lower() == 'tsne':
            if TSNE is None:
                raise ImportError("scikit-learn is required for t-SNE (pip install scikit-learn).")
            
            if 'perplexity' not in projection_kwargs:
                projection_kwargs['perplexity'] = min(30, max(1, activations.shape[0] // 3))
            if 'n_components' not in projection_kwargs:
                projection_kwargs['n_components'] = 2

            projector = TSNE(**projection_kwargs)
            projections = projector.fit_transform(activations)
            
        elif method.lower() == 'umap':
            try:
                import umap
            except ImportError:
                raise ImportError("umap-learn is required for UMAP (pip install umap-learn).")
            
            if 'n_components' not in projection_kwargs:
                projection_kwargs['n_components'] = 2

            projector = umap.UMAP(verbose=False, **projection_kwargs)
            projections = projector.fit_transform(activations)
        else:
            raise ValueError(f"Unsupported projection method: {method}. Choose 'tsne' or 'umap'.")

        print("[EXPLAINER] Projection complete. Generating scatter plot...")
        self._plot_projections(projections, y, method=method, save_filename=save_filename, **plot_kwargs)

    def _plot_projections(self, projections, labels, method: str, save_filename: str = None, **plot_kwargs):
        figsize = plot_kwargs.get('figsize', (10, 8))
        cmap_name = plot_kwargs.get('cmap', 'tab10')
        point_size = plot_kwargs.get('s', 15)
        alpha = plot_kwargs.get('alpha', 0.7)
        title = plot_kwargs.get('title', f"Feature Space Projection ({method.upper()})")

        plt.figure(figsize=figsize)
        
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap(cmap_name)
        
        for index, label in enumerate(unique_labels):
            idx = np.where(labels == label)
            plt.scatter(projections[idx, 0], projections[idx, 1], 
                        label=str(label), s=point_size, alpha=alpha, 
                        color=cmap(index % cmap.N))

        plt.title(title)
        plt.legend(title="Classes")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(plot_kwargs.get('grid', True), linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save_filename:
            save_path = os.path.join(self.base_save_dir, save_filename)
            plt.savefig(save_path, dpi=plot_kwargs.get('dpi', 300))
            print(f"[EXPLAINER] Visualization saved: {save_path}")
        
        if plot_kwargs.get('show', False):
            plt.show()
        else:
            plt.close()

class TrainingController:
    """
    The training controller is responsible for managing the complete lifecycle of train / test in a modular way,
    i.e. scenarios can be chosen dynamically, evaluating and testing models can be cnofigured to work at each node
    or cycle, observability regarding the models and their names / paths, complete control of the travelling model 
    training simulation. 
    It is worth noting that the training controller operates based on a given scenario, i.e. the active scenario is
    the one used by the training controller. 
    """   
    
    def __init__(self, scenario_controller: ScenarioController, base_path: str = './system_state/'):
        self.scenario_controller = scenario_controller
        self.base_path = base_path
        
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        self.models: Dict[str, Model] = {}
        self._active_model_name: str | None = None

        self.eda_enabled: bool = False
        
        self.evaluator = Evaluator()
        self.explainer = Explainer(base_save_dir=os.path.join('reports', 'visualizations'))
        self.training_history = []
        
        self.evaluation_timings = []  # e.g. 'pre-node', 'post-node', 'pre-cycle', 'post-cycle'
        self.explanation_timings = []
        self.explanation_config = {
            'method': 'umap',
            'layer_index': -2,
            'n_neighbors': 50,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'n_epochs': 300,
            'random_state': 42,
            'n_components': 2,
            'data_fraction': 1.0
        }
        self.explanation_history = []
        
        self.current_cycle = 1
        self.global_test_x = None
        self.global_test_y = None
        self._metrics_compiled = False
        self.experiment_name = None

    def set_eda_enabled(self, enabled: bool) -> None:
        self.eda_enabled = enabled

    def _generate_eda_charts(self, report_name: str) -> list[str]:
        post_node_records = [r for r in self.evaluator.evaluation_history if r.get('timing') == 'post-node']
        if not post_node_records:
            print("[EDA] No post-node evaluation records found. Skipping EDA charts.")
            return []

        charts_dir = os.path.join('visualizations', 'eda', report_name)
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)

        # Accuracy
        has_accuracy = any('accuracy' in r.get('metrics', {}) for r in post_node_records)
        # Loss
        has_loss = any('loss' in r.get('metrics', {}) for r in post_node_records)

        saved_files = []

        # Chart A - Accuracy by Cycle
        if has_accuracy:
            cycle_accs = {}
            for r in post_node_records:
                c = r.get('cycle')
                if c not in cycle_accs:
                    cycle_accs[c] = []
                acc = r.get('metrics', {}).get('accuracy')
                if acc is not None:
                    cycle_accs[c].append(acc)
            
            cycles = sorted(list(cycle_accs.keys()))
            avg_acc = [np.mean(cycle_accs[c]) if cycle_accs[c] else 0.0 for c in cycles]
            
            plt.figure()
            plt.plot(cycles, avg_acc, marker='o')
            plt.xlabel('Cycle')
            plt.ylabel('Accuracy')
            plt.ylim(0.0, 1.0)
            plt.xticks(cycles)
            plt.grid(True, alpha=0.3)
            plt.title('Model Accuracy per Training Cycle')
            
            path_a = os.path.join(charts_dir, 'eda_accuracy_by_cycle.png')
            plt.savefig(path_a, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(path_a)
        else:
            print("[EDA] Missing accuracy metric. Skipping accuracy charts.")
            saved_files.append("") # Placeholder for skipped chart

        # Chart B - Accuracy by Node per Cycle
        if has_accuracy:
            cycle_node_acc = {}
            for r in post_node_records:
                c = r.get('cycle')
                n = str(r.get('node', ''))
                acc = r.get('metrics', {}).get('accuracy')
                if acc is not None:
                    if c not in cycle_node_acc:
                        cycle_node_acc[c] = []
                    cycle_node_acc[c].append((n, acc))
                    
            plt.figure()
            cycles = sorted(list(cycle_node_acc.keys()))
            cmap = cm.get_cmap('tab10' if len(cycles) <= 10 else 'tab20')
            ax = plt.gca()
            
            all_nodes = set()
            for c in cycles:
                for (n, _) in cycle_node_acc[c]:
                    all_nodes.add(n)
            all_nodes = sorted(list(all_nodes))
            
            node_idx_map = {n: i for i, n in enumerate(all_nodes)}

            for i, c in enumerate(cycles):
                nodes_ordered = []
                accs_ordered = []
                for n in all_nodes:
                    # find acc for node n in this cycle
                    match = [v for nd, v in cycle_node_acc[c] if nd == n]
                    if match:
                        nodes_ordered.append(n)
                        accs_ordered.append(match[0])
                        
                plt.plot(nodes_ordered, accs_ordered, marker='o', alpha=0.75, linewidth=1.8, 
                         color=cmap(i % cmap.N), label=f'Cycle {c}')
                         
            plt.xlabel('Node')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            plt.title('Model Accuracy per Node (by Cycle)')
            if len(cycles) <= 10:
                plt.legend(loc='best')
            else:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(cycles), vmax=max(cycles)))
                sm.set_array([])
                plt.colorbar(sm, label='Cycle')
                
            path_b = os.path.join(charts_dir, 'eda_accuracy_by_node.png')
            plt.savefig(path_b, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(path_b)
        else:
            saved_files.append("")

        # Chart C - Loss by Cycle
        if has_loss:
            cycle_loss = {}
            for r in post_node_records:
                c = r.get('cycle')
                if c not in cycle_loss:
                    cycle_loss[c] = []
                loss = r.get('metrics', {}).get('loss')
                if loss is not None:
                    cycle_loss[c].append(loss)
            
            cycles = sorted(list(cycle_loss.keys()))
            avg_loss = [np.mean(cycle_loss[c]) if cycle_loss[c] else 0.0 for c in cycles]
            
            plt.figure()
            plt.plot(cycles, avg_loss, marker='o')
            plt.xlabel('Cycle')
            plt.ylabel('Loss')
            bottom_lim = 0
            plt.ylim(bottom=bottom_lim)
            plt.xticks(cycles)
            plt.grid(True, alpha=0.3)
            plt.title('Model Loss per Training Cycle')
            
            path_c = os.path.join(charts_dir, 'eda_loss_by_cycle.png')
            plt.savefig(path_c, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(path_c)
        else:
            print("[EDA] Missing loss metric. Skipping loss charts.")
            saved_files.append("")

        # Chart D - Loss by Node per Cycle
        if has_loss:
            cycle_node_loss = {}
            for r in post_node_records:
                c = r.get('cycle')
                n = str(r.get('node', ''))
                loss = r.get('metrics', {}).get('loss')
                if loss is not None:
                    if c not in cycle_node_loss:
                        cycle_node_loss[c] = []
                    cycle_node_loss[c].append((n, loss))
                    
            plt.figure()
            cycles = sorted(list(cycle_node_loss.keys()))
            cmap = cm.get_cmap('tab10' if len(cycles) <= 10 else 'tab20')
            
            all_nodes = set()
            for c in cycles:
                for (n, _) in cycle_node_loss[c]:
                    all_nodes.add(n)
            all_nodes = sorted(list(all_nodes))

            for i, c in enumerate(cycles):
                nodes_ordered = []
                loss_ordered = []
                for n in all_nodes:
                    # find loss for node n in this cycle
                    match = [v for nd, v in cycle_node_loss[c] if nd == n]
                    if match:
                        nodes_ordered.append(n)
                        loss_ordered.append(match[0])
                        
                plt.plot(nodes_ordered, loss_ordered, marker='o', alpha=0.75, linewidth=1.8, 
                         color=cmap(i % cmap.N), label=f'Cycle {c}')
                         
            plt.xlabel('Node')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.title('Model Loss per Node (by Cycle)')
            if len(cycles) <= 10:
                plt.legend(loc='best')
            else:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(cycles), vmax=max(cycles)))
                sm.set_array([])
                plt.colorbar(sm, label='Cycle')
                
            path_d = os.path.join(charts_dir, 'eda_loss_by_node.png')
            plt.savefig(path_d, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(path_d)
        else:
            saved_files.append("")

        return saved_files

    def set_experiment_name(self, name: str):
        self.experiment_name = name
        target_dir = os.path.join('reports', 'visualizations', name)
        self.explainer.base_save_dir = target_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        print(f"[INFO] Experiment name set to '{name}'. Visualizations will be saved to '{target_dir}'.")

    def configure_evaluation_timings(self, timings: list[str]):
        self.evaluation_timings = timings
        print(f"[INFO] Evaluation timings configured to: {self.evaluation_timings}")

    def _encode_labels(self, y, num_classes):
        """
        Converts a sequence of sparse categorical labels into a one-hot encoded array.
        Returns the original array if it's already one-hot.
        """
        if y is None or len(y) == 0:
            return y
        # Check if already one hot encoded
        if len(y.shape) > 1 and y.shape[-1] == num_classes:
            return y
            
        y = np.squeeze(y)
        tf = _get_tf()
        return tf.keras.utils.to_categorical(y, num_classes=num_classes)

    def generate_global_test_set(self):
        scenario = self.scenario_controller.get_active_scenario()
        if not scenario:
            print("[WARNING] No scenario for global test set genearation")
            return
            
        data_controller = getattr(self.scenario_controller, '_data_controller', None)
        if not data_controller:
            print("[WARNING] No DataController linked. Cannot generate global test set.")
            return
            
        active_node_names = [n.name for n in scenario.nodes]
        self.global_test_x, self.global_test_y = data_controller.get_global_test_set(active_node_names) 
        
        # Apply one-hot encoding dynamically based on the model's output size
        if self._active_model_name:
            model_obj = self.get_active_model()
            if getattr(model_obj, 'status', None) == 'deferred':
                model_obj.load()
            if model_obj and model_obj.keras_model:
                try:
                    num_classes = model_obj.keras_model.output_shape[-1]
                    self.global_test_y = self._encode_labels(self.global_test_y, num_classes)
                except Exception as e:
                    print(f"[WARNING] Could not encode global test labels: {e}")
                    
        print(f"[INFO] Global test set generated: X shape: {self.global_test_x.shape}, Y shape: {self.global_test_y.shape}")

    def trigger_evaluation(self, timing: str):
        if timing not in self.evaluation_timings:
            return
            
        if self.global_test_x is None or len(self.global_test_x) == 0:
            print("[INFO] Global test set not available. Generating before explanation...")
            self.generate_global_test_set()

        active_model = self.get_active_model()
        current_node = self.scenario_controller.get_current_node()
        node_name = current_node.name if current_node else "N/A"
        
        # Ensure custom metrics are compiled before running evaluate()
        self._lazy_compile_model()
        
        # Re-verify and enforce encoding alignment on evaluate() call dynamically safely
        try:
            if getattr(active_model, 'status', None) == 'deferred':
                active_model.load()
            num_classes = active_model.keras_model.output_shape[-1]
            self.global_test_y = self._encode_labels(self.global_test_y, num_classes)
        except Exception as e:
            print(f"[WARNING] Could not verify test label encoding during evaluation: {e}")

        print(f"[INFO] Running {timing} evaluation...")
        self.evaluator.evaluate_step(
            model=active_model, 
            test_x=self.global_test_x, 
            test_y=self.global_test_y, 
            cycle=self.current_cycle, 
            node_name=node_name, 
            timing=timing,
            verbose=0
        )

    def configure_explanation_timings(self, timings: list[str]):
        self.explanation_timings = timings
        print(f"[INFO] Explanation timings configured to: {self.explanation_timings}")

    def configure_explanation(self, config_dict: dict):
        if not hasattr(self, 'explanation_config'):
            self.explanation_config = {}
        self.explanation_config.update(config_dict)
        print(f"[INFO] Explanation config updated: {self.explanation_config}")

    def trigger_explanation(self, timing: str):
        if not hasattr(self, 'explanation_timings') or timing not in self.explanation_timings:
            return
            
        active_model = self.get_active_model()
        current_node = self.scenario_controller.get_current_node()
        node_name = current_node.name if current_node else "N/A"

        if self.global_test_x is None or len(self.global_test_x) == 0:
            print("[INFO] Global test set not available. Generating before explanation...")
            self.generate_global_test_set()

        if self.global_test_x is None or len(self.global_test_x) == 0:
            print(f"[WARNING] Skipping {timing} explanation: global test set is empty.")
            return

        node_x = self.global_test_x
        node_y = self.global_test_y
        
        fraction = self.explanation_config.get('data_fraction', 1.0)
        if fraction < 1.0:
            np.random.seed(self.explanation_config.get('random_state', 42))
            indices = np.random.choice(len(node_x), size=int(len(node_x) * fraction), replace=False)
            node_x = node_x[indices]
            node_y = node_y[indices]
            
        # Explainer expects integer labels for scatter plot grouping
        node_y = np.squeeze(node_y)
        if len(node_y.shape) > 1:
            node_y = np.argmax(node_y, axis=-1)

        save_filename = f"cycle{self.current_cycle}_{node_name}_{timing}.png"
        
        method = self.explanation_config.get('method', 'umap')
        layer_index = self.explanation_config.get('layer_index', -2)
        
        print(f"[EXPLANATION] Generating {timing} visualization for node '{node_name}' (Cycle {self.current_cycle})...")
        print(f"[EXPLANATION] Dataset: global test set | Samples: {len(node_x)} | Method: {method.upper()} | Layer index: {layer_index}")
        print(f"[EXPLANATION] Saving to: {save_filename}")

        proj_keys = ['n_neighbors', 'min_dist', 'metric', 'n_epochs', 'random_state', 'n_components', 'perplexity']
        projection_kwargs = {k: self.explanation_config[k] for k in proj_keys if k in self.explanation_config}

        # Need to disable plot showing
        plot_kwargs = {'show': False}
        
        try:
            self.explainer.generate_visualization(
                model=active_model,
                x=node_x,
                y=node_y,
                method=method,
                layer_index=layer_index,
                save_filename=save_filename,
                projection_kwargs=projection_kwargs,
                plot_kwargs=plot_kwargs
            )
            
            if not hasattr(self, 'explanation_history'):
                self.explanation_history = []
            
            self.explanation_history.append({
                'cycle': self.current_cycle,
                'node': node_name,
                'timing': timing,
                'file_path': os.path.join(self.explainer.base_save_dir, save_filename),
                'method': method,
                'config': self.explanation_config.copy()
            })
        except Exception as e:
            print(f"[WARNING] Error generating explanation visualization: {e}")

    def clear_training_and_evaluation_history(self):
        """
        Clears all training and evaluation logs to ensure reports start fresh before multi-cycle runs.
        """
        print("[INFO] Clearing prior training and evaluation history...")
        self.training_history = []
        self.explanation_history = []
        if hasattr(self, 'evaluator'):
            self.evaluator.evaluation_history = []
        self.current_cycle = 1
    def reset_model(self):
        """
        Loads a fresh model so multi-experiment runs go on uncorrupted by previous experiments
        """


    def verify_integrity(self):
        # Handle unpickled configurations from older versions 
        if not hasattr(self, 'evaluation_timings'):
            self.evaluation_timings = []
        if not hasattr(self, 'explanation_timings'):
            self.explanation_timings = []
        if not hasattr(self, 'explanation_history'):
            self.explanation_history = []
        if not hasattr(self, 'explanation_config'):
            self.explanation_config = {
                'method': 'umap',
                'layer_index': -2,
                'n_neighbors': 50,
                'min_dist': 0.1,
                'metric': 'euclidean',
                'n_epochs': 300,
                'random_state': 42,
                'n_components': 2,
                'data_fraction': 1.0
            }
        if not hasattr(self, 'current_cycle'):
            self.current_cycle = 1
        if not hasattr(self, 'global_test_x'):
            self.global_test_x = None
        if not hasattr(self, 'global_test_y'):
            self.global_test_y = None
            
        if hasattr(self, 'explainer'):
            target_dir = os.path.join('reports', 'visualizations')
            if hasattr(self, 'experiment_name') and self.experiment_name:
                target_dir = os.path.join(target_dir, self.experiment_name)
            if self.explainer.base_save_dir != target_dir:
                self.explainer.base_save_dir = target_dir
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

        self._metrics_compiled = False
        
        is_tf_loaded = 'tensorflow' in sys.modules
        
        invalid_models = []
        for name, model_obj in self.models.items():
            if not os.path.exists(model_obj.model_path):
                print(f"[WARNING] Model '{name}' path no longer exists: {model_obj.model_path}. It will be removed.")
                invalid_models.append(name)
            else:
                if model_obj.keras_model is None:
                    if not is_tf_loaded:
                        model_obj.status = 'deferred'
                    else:
                        try:
                            model_obj.load()
                        except Exception as e:
                            print(f"[WARNING] Model '{name}' failed re-validation and will be removed: {e}")
                            invalid_models.append(name)
                        
        for name in invalid_models:
            del self.models[name]
            if self._active_model_name == name:
                self._active_model_name = None
                print(f"[WARNING] Active model '{name}' was removed due to integrity checks.")
        print('Training controller integrity verified.')

    def add_model(self, name: str, model_path: str, description: str = ""):
        if name in self.models:
            raise ValueError(f"Model '{name}' already exists.")
        new_model = Model(name, model_path, description)
        self.models[name] = new_model
        print(f"Model '{name}' added successfully.")

    def remove_model(self, name: str):
        if name not in self.models:
            raise ValueError(f"Model '{name}' does not exist.")
        del self.models[name]
        
        if self._active_model_name == name:
            self._active_model_name = None
            print(f"[INFO] Active model '{name}' was removed. No model is currently active.")
            
        print(f"Model '{name}' removed.")

    def edit_model(self, old_name: str, new_name: str, new_description: str):
        if old_name not in self.models:
            raise NameError(f"Model '{old_name}' does not exist.")
            
        if new_name != old_name and new_name in self.models:
            raise ValueError(f"Model '{new_name}' already exists.")

        model = self.models.pop(old_name)
        model.name = new_name
        if new_description is not None:
            model.description = new_description
            
        self.models[new_name] = model
        
        if self._active_model_name == old_name:
            self._active_model_name = new_name
            
        print(f"Model '{old_name}' updated to '{new_name}'.")

    def set_active_model(self, name: str):
        if name not in self.models:
            raise ValueError(f"Model '{name}' does not exist.")
        self._active_model_name = name
        print(f"Model '{name}' is now active.")

    def reset_active_model_weights(self, revert_to_file: bool = False):
        if not self._active_model_name:
            raise RuntimeError("No active model is set.")
        model_obj = self.models[self._active_model_name]
        
        if revert_to_file:
            model_obj.load()
            self._metrics_compiled = False
            self._lazy_compile_model()
            print(f"Active model '{self._active_model_name}' weights have been reverted to file state.")
        else:
            if getattr(model_obj, 'status', None) == 'deferred':
                model_obj.load()
            if not model_obj.keras_model:
                raise RuntimeError("Active model has no underlying Keras model to reset.")
            tf = _get_tf()
            model_obj.keras_model = tf.keras.models.clone_model(model_obj.keras_model)
            self._metrics_compiled = False
            self._lazy_compile_model()
            print(f"Active model '{self._active_model_name}' weights have been reset (randomized).")

    def get_active_model(self) -> Model:
        if not self._active_model_name:
            raise RuntimeError("No active model is set.")
        return self.models[self._active_model_name]

    def _resolve_loss(self, model_obj: 'Model'):
        """
        Returns a correctly configured loss object for the given model.

        Detects whether the final layer has a softmax activation to determine
        whether from_logits should be True or False. Converts sparse categorical
        CE to categorical CE when labels are one-hot encoded (as is the case
        throughout this system), while correctly preserving the from_logits flag.
        """
        tf = _get_tf()
        raw_loss = getattr(model_obj.keras_model, 'loss', None)

        # Normalize to a string name for comparison
        if isinstance(raw_loss, str):
            loss_name = raw_loss
        elif hasattr(raw_loss, 'name'):
            loss_name = raw_loss.name
        elif hasattr(raw_loss, '__name__'):
            loss_name = raw_loss.__name__
        else:
            # Unknown loss type — return as-is and let Keras handle it
            return raw_loss

        loss_name = loss_name.lower()

        # Detect whether the output layer applies softmax internally
        try:
            last_layer = model_obj.keras_model.layers[-1]
            activation = getattr(last_layer, 'activation', None)
            activation_name = getattr(activation, '__name__', '') if activation else ''
            has_softmax = activation_name == 'softmax'
        except Exception:
            # If detection fails, assume softmax is present (safer default)
            has_softmax = True

        from_logits = not has_softmax

        if 'sparse_categorical_crossentropy' in loss_name or 'categorical_crossentropy' in loss_name:
            return tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)

        # All other loss types (binary CE, MSE, etc.) are returned unchanged
        return raw_loss

    def _lazy_compile_model(self):
        """
        Dynamically recompiles the Keras model with configured active metric objects on-demand.
        """
        if getattr(self, '_metrics_compiled', False):
            return
            
        active_model = getattr(self, '_active_model_name', None)
        if not active_model: return
        
        real_model_obj = self.get_active_model()
        if getattr(real_model_obj, 'status', None) == 'deferred':
            real_model_obj.load()
        if not real_model_obj or not real_model_obj.keras_model:
            return

        configured_metrics = getattr(self.evaluator, 'active_metrics_config', [])
        if not configured_metrics:
            return
            
        
        
        num_classes = 1
        if real_model_obj:
            if getattr(real_model_obj, 'status', None) == 'deferred':
                real_model_obj.load()
        if real_model_obj and real_model_obj.keras_model:
            try:
                num_classes = real_model_obj.keras_model.output_shape[-1]
            except:
                pass
        
        pos_class_id = 1 if num_classes == 2 else None

        tf = _get_tf()
        AUC = tf.keras.metrics.AUC
        Precision = tf.keras.metrics.Precision
        Recall = tf.keras.metrics.Recall
        from classes.metrics import F1Score, FBetaScore, NPV

        keras_metrics = []
        for metric in configured_metrics:
            if metric == 'accuracy':
                keras_metrics.append('accuracy')
            elif metric == 'recall':
                keras_metrics.append(Recall(class_id=pos_class_id, name='recall'))
            elif metric == 'precision':
                keras_metrics.append(Precision(class_id=pos_class_id, name='precision'))
            elif metric == 'auc':
                keras_metrics.append(AUC(name='auc'))
            elif metric == 'f1_score':
                keras_metrics.append(F1Score(name='f1_score'))
            elif metric.startswith('fbeta_'):
                # Extract beta value
                try:
                    beta = float(metric.split('_')[1])
                except:
                    beta = 1.0
                keras_metrics.append(FBetaScore(beta=beta, name=f'f{beta}_score'))
            elif metric == 'npv':
                keras_metrics.append(NPV(name='npv'))
            else:
                keras_metrics.append(metric)

        try:
            print("[INFO] Lazy loading: Compiling metrics into the active model...")
            if getattr(real_model_obj, 'status', None) == 'deferred':
                real_model_obj.load()
                
            optimizer = real_model_obj.keras_model.optimizer
            if optimizer is None:
                print("[WARNING] Active model has no optimizer during lazy compile. Skipping recompile.")
                return
                
            loss = self._resolve_loss(real_model_obj)
            
            real_model_obj.keras_model.compile(optimizer=optimizer, loss=loss, metrics=keras_metrics)
            self._metrics_compiled = True
        except Exception as e:
            print(f"[ERROR] Could not lazily recompile active model automatically: {e}")

    def _log_optimizer_state(self, optimizer, label: str = ""):
        """
        Diagnostic logger for optimizer internal state.
        Prints the step count and a snapshot of the first few slot variables
        (e.g. Adam's momentum/variance accumulators) so we can verify whether
        .compile() resets them across node boundaries.
        """
        # try:
        #     prefix = f"[OPT-DIAG] {label}"
        #     opt_name = optimizer.__class__.__name__
            
        #     # Step count — tells us how many gradient updates have occurred
        #     step = None
        #     if hasattr(optimizer, 'iterations'):
        #         step = int(optimizer.iterations.numpy())
        #     elif hasattr(optimizer, '_iterations'):
        #         step = int(optimizer._iterations.numpy())

        #     # Learning rate
        #     lr = None
        #     try:
        #         if hasattr(optimizer, 'learning_rate'):
        #             lr_attr = optimizer.learning_rate
        #             lr = float(lr_attr.numpy() if hasattr(lr_attr, 'numpy') else lr_attr)
        #     except Exception:
        #         pass

        #     print(f"{prefix}")
        #     print(f"  Optimizer : {opt_name}")
        #     print(f"  Step      : {step}")
        #     print(f"  LR        : {lr:.2e}" if lr is not None else "  LR        : N/A")

        #     # Slot variables (momentum, variance, etc.) — sample the first 3 weights only
        #     # to keep logs readable. We print the L2 norm of each slot tensor as a fingerprint.
        #     variables = optimizer.variables() if hasattr(optimizer, 'variables') else []
        #     slot_vars = [v for v in variables if 'iteration' not in v.name and 'learning_rate' not in v.name]
            
        #     if slot_vars:
        #         import numpy as np
        #         print(f"  Slot vars : {len(slot_vars)} total — showing L2 norm of first 3:")
        #         for v in slot_vars[:3]:
        #             try:
        #                 norm = float(np.linalg.norm(v.numpy()))
        #                 print(f"    {v.name:<50} norm={norm:.6f}")
        #             except Exception:
        #                 print(f"    {v.name:<50} (could not read value)")
        #     else:
        #         print("  Slot vars : none found (optimizer may not have been used yet)")

        # except Exception as e:
        #     print(f"[OPT-DIAG] Could not log optimizer state: {e}")

    def train_current_node(self, node_x=None, node_y=None, epochs: int = 1, batch_size: int = 32, **kwargs):
        active_model = self.get_active_model()
        current_node = self.scenario_controller.get_current_node()
        if not current_node:
            raise RuntimeError("No active node found in the current scenario.")

        print(f"Training model '{active_model.name}' on node: {current_node.name}")
        
        self.trigger_evaluation('pre-node')
        self.trigger_explanation('pre-node')

        # If data is not provided explicitly, attempt to load it from the node's datasets
        if node_x is None or node_y is None:
            print(f"current_node.datasets: {current_node.datasets}")
            if not current_node.datasets:
                raise RuntimeError(f"The current node ('{current_node.name}') has no datasets to train on.")
                
            all_x, all_y = [], []
            for ds_name, ds in current_node.datasets.items():
                print(f"[INFO] Loading data from subset: '{ds.name}'")
                if ds.structure == 'medmnist':
                    x_path = os.path.join(ds.path, 'train_images.npy')
                    y_path = os.path.join(ds.path, 'train_labels.npy')
                    if os.path.exists(x_path) and os.path.exists(y_path):
                        all_x.append(np.load(x_path))
                        all_y.append(np.load(y_path))
                        print(f"       -> Loaded {len(all_x[-1])} samples from '{ds.name}'")
                    else:
                        print(f"[WARNING] Missing training files in subset {ds_name} at {ds.path}")
                else:
                    print(f"[WARNING] Unsupported structure '{ds.structure}' for dataset '{ds_name}'. Only MedMNIST format is supported.")
                    
            if not all_x or not all_y:
                raise RuntimeError("No valid data could be aggregated for training from the current node.")
                
            node_x = np.concatenate(all_x)
            node_y = np.concatenate(all_y)
            
            try:
                if getattr(active_model, 'status', None) == 'deferred':
                    active_model.load()
                num_classes = active_model.keras_model.output_shape[-1]
                node_y = self._encode_labels(node_y, num_classes)
            except Exception as e:
                print(f"[WARNING] Could not encode node labels: {e}")
                
            print(f"[INFO] Data aggregation complete. Total samples: {len(node_x)}. X shape: {node_x.shape}, Y shape: {node_y.shape}")

        # Remove custom metrics from compile during training to speed up fitting
        basic_metrics = ['accuracy']
        if getattr(active_model, 'status', None) == 'deferred':
            active_model.load()
            
        loss = self._resolve_loss(active_model)
        
        keras_optimizer = active_model.keras_model.optimizer
        if keras_optimizer is None:
            tf = _get_tf()
            keras_optimizer = tf.keras.optimizers.Adam()
            print("[WARNING] Model had no optimizer set. Initialized a new Adam optimizer.")

        self._log_optimizer_state(keras_optimizer, label=f"PRE-COMPILE  | cycle={self.current_cycle} node={current_node.name}")
        active_model.keras_model.compile(optimizer=keras_optimizer, loss=loss, metrics=basic_metrics)
        self._log_optimizer_state(keras_optimizer, label=f"POST-COMPILE | cycle={self.current_cycle} node={current_node.name}")
        self._metrics_compiled = False # Force next evaluation to lazily compile properly


        # DATA BALANCING MECHANISM

        # 1. Fetch a reference subset from the current node
        # (Since distribution is invariant, querying the first available subset is sufficient)
        reference_subset = next(iter(current_node.datasets.values()))

        # 2. Extract cached population statistics
        n_classes = reference_subset.n_classes
        total_train_examples = reference_subset.n_examples_train
        class_frequencies = reference_subset.n_examples_per_class

        # 3. Construct the cost matrix mapping
        class_weight_dict = {
            int(cls_name): float(total_train_examples / (n_classes * count))
            for cls_name, count in class_frequencies.items()
        }

        print(f'[WARNING] Using data balancing mechanism under the assumption of uniform data partitioning! Weights: {class_weight_dict}')

        try:
            history = active_model.fit(
                node_x, 
                node_y, 
                epochs=epochs, 
                batch_size=batch_size, 
                class_weight=class_weight_dict,
                **kwargs
            )

            self._log_optimizer_state(keras_optimizer, label=f"POST-FIT     | cycle={self.current_cycle} node={current_node.name}")
            self.training_history.append({
                "cycle": self.current_cycle,
                "model": active_model.name,
                "node": current_node.name,
                "epochs": epochs,
                "batch_size": batch_size,
                "history": history.history
            })

            self._persist_layer_activations(node_name=current_node.name)

            self.trigger_evaluation('post-node')
            self.trigger_explanation('post-node')
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to train on node {current_node.name}: {e}")

    def evaluate_current_node(self, node_x, node_y, **kwargs):
        active_model = self.get_active_model()
        current_node = self.scenario_controller.get_current_node()
        if not current_node:
            raise RuntimeError("No active node found in the current scenario.")

        print(f"Evaluating model '{active_model.name}' on node: {current_node.name}")
        return self.evaluator.evaluate_on_data(active_model, node_x, node_y, **kwargs)

    def explain_current_node(self, node_x, node_y, method: str = 'tsne', layer_index: int = -2, 
                             save_filename: str = None, projection_kwargs: dict = None, plot_kwargs: dict = None):
        active_model = self.get_active_model()
        current_node = self.scenario_controller.get_current_node()
        if not current_node:
            raise RuntimeError("No active node found in the current scenario.")

        print(f"Generating {method.upper()} visualization for model '{active_model.name}' on node: {current_node.name}")
        
        if not save_filename:
            save_filename = f"{active_model.name}_{current_node.name}_{method}.png"

        self.explainer.generate_visualization(
            model=active_model, x=node_x, y=node_y, method=method, 
            layer_index=layer_index, save_filename=save_filename, 
            projection_kwargs=projection_kwargs, plot_kwargs=plot_kwargs
        )

    def get_current_node(self):
        return self.scenario_controller.get_current_node()

    def advance_scenario(self) -> bool:
        return self.scenario_controller.advance_to_next_node()

    def generate_markdown_report(self, report_name: str):
        """
        Builds a structured markdown report encompassing Data, Scenario, Active Model,
        and all accumulated metrics during training and evaluation.
        """
        
        os.makedirs('reports', exist_ok=True)
        report_path = os.path.join('reports', f"{report_name}.md")

        # Gather context
        active_model = self.get_active_model()
        scenario = self.scenario_controller.get_active_scenario()
        data_controller = getattr(self.scenario_controller, '_data_controller', None)

        md_content = []
        md_content.append(f"# Simulation Report: {report_name}")
        md_content.append("\n## 1. Data Information & Scenario Configuration")

        if data_controller and scenario:
            
            # Scenario basics
            md_content.append(f"- **Scenario Name:** `{scenario.name}`")
            if hasattr(scenario, 'description') and scenario.description:
                md_content.append(f"- **Scenario Description:** _{scenario.description}_")
            
            # Nodes and distributions
            md_content.append("\n### Node Distribution")
            md_content.append("| Node | Description | Subsets (Type/Ratio) | N° of Examples |")
            md_content.append("|---|---|---|---|")

            # Total number of train examples
            n_examples_tot = 0

            # Original dataset tracking, so we get the original number of train/test examples
            original_datasets = set()

            # Global test set number of examples 
            
            for node in scenario.nodes:
                subsets_str = ", ".join(node.datasets.keys()) if node.datasets else "None"
                node_desc = getattr(node, 'description', 'N/A')
                
                n_examples = node.get_n_examples()
                n_examples_tot += n_examples

                node_origin_dataset_names = set([node.datasets[ds].origin_dataset_name for ds in node.datasets])

                original_datasets = original_datasets | node_origin_dataset_names

                md_content.append(f"| {node.name} | {node_desc} | {subsets_str} | {n_examples} |")
            
            print('ORIGINAL DATASETS LIST: ', original_datasets)
            
            if len(list(original_datasets)) == 1:
                original_dataset = self.scenario_controller.get_dataset(original_datasets.pop())
                
                print('ORIGINAL DATASET RECEIVED: ', type(original_dataset),original_dataset.__str__() )
                original_n_test = original_dataset.n_examples_test  
                original_n_train = original_dataset.n_examples_train
            
            else:
                original_n_test =  {od: self.scenario_controller.get_dataset(od).n_examples_test for od in original_datasets}
                original_n_train = {od: self.scenario_controller.get_dataset(od).n_examples_train for od in original_datasets}

            # Synthesized data information
            md_content.append(f'\n Total scenario train examples count: {n_examples_tot}')
            md_content.append(f'\n Global test set  examples count: {self.global_test_x.shape[0]}')
            md_content.append(f'\n Original train examples count: {original_n_train}')
            md_content.append(f'\n Original dataset test examples count: {original_n_test}')


        else:
            md_content.append("\n*No strict scenario/data context available.*")

        md_content.append("\n## 2. Active Model Details")
        if active_model:
            md_content.append(f"- **Name:** `{active_model.name}`")
            if hasattr(active_model, 'description') and active_model.description:
                md_content.append(f"- **Model Description:** _{active_model.description}_")
            md_content.append(f"- **Path:** `{active_model.model_path}`")
            
            if getattr(active_model, 'status', None) == 'deferred':
                active_model.load()
            if active_model.keras_model:

                model = active_model.keras_model

                tf = _get_tf()
                ""
                if model.optimizer is not None:

                    # Optimizer
                    optimizer = type(model.optimizer).__name__

                    # Learning rate
                    lr = model.optimizer.learning_rate
                    lr_final = f"Dynamic Schedule ({type(lr).__name__})" if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule) else f"{tf.keras.backend.get_value(lr)}"
                        
                    # Weight decay
                    opt_config = model.optimizer.get_config()
                    weight_decay = f"{opt_config.get('weight_decay', 'None or Not Set')}"
                
                else: 
                    raise Exception('The model was still not compiled when report generation got called')
                
                # Weight initialization
                weight_initializers = set([f"{type(layer.kernel_initializer).__name__}" for layer in model.layers if hasattr(layer,'kernel_initializer')])

                md_content.append(f"- **Optimizer:** `{optimizer}`")
                md_content.append(f"- **Learning rate:** `{lr_final}`")
                md_content.append(f"- **Weight Decay:** `{weight_decay}`")
                md_content.append(f"- **Weight initializer(s):** `{weight_initializers}`")
                
                # We can dump model summary into string
                stringlist = []
                active_model.keras_model.summary(print_fn=lambda x: stringlist.append(x))
                model_summary = "\n".join(stringlist)
                md_content.append("\n### Architecture Summary")
                md_content.append("```text\n" + model_summary + "\n```")
            
            else:
                md_content.append("\n*Model architecture not loaded into memory.*")
        else:
            md_content.append("\n*No active model attached.*")

        md_content.append("\n## 3. Training & Evaluation Metrics")
        
        active_metrics = getattr(self.evaluator, 'active_metrics_config', [])
        if active_metrics:
            md_content.append(f"- **Selected Metrics:** `{', '.join(active_metrics)}`")
        
        # Check timings safely (backward compatibility)
        if hasattr(self, 'evaluation_timings'):
            md_content.append(f"- **Evaluation Timings Active:** `{', '.join(self.evaluation_timings)}`")

        eval_history = getattr(self.evaluator, 'evaluation_history', [])
        if eval_history:
            md_content.append("\n### Cycle Evaluations")
            
            # Group by cycle
            cycles = {}
            for record in eval_history:
                c = record.get('cycle', 0)
                if c not in cycles:
                    cycles[c] = []
                cycles[c].append(record)

            for c in sorted(cycles.keys()):
                md_content.append(f"\n#### Cycle {c}")
                
                # Fetch headers dynamically
                all_metrics = set()
                for rec in cycles[c]:
                    for k in rec.get('metrics', {}).keys():
                        all_metrics.add(k)
                
                # Default sorting to keep it readable, Accuracy/Loss first if available
                def sort_metrics(m):
                    if m.lower() == 'accuracy': return "0"
                    if m.lower() == 'loss': return "1"
                    return m
                
                all_metrics = sorted(list(all_metrics), key=sort_metrics)
                
                headers = ["Node", "Timing"] + all_metrics
                md_content.append("| " + " | ".join(headers) + " |")
                md_content.append("| " + " | ".join(["---"] * len(headers)) + " |")
                
                for rec in cycles[c]:
                    row = [str(rec.get('node')), str(rec.get('timing'))]
                    for m in all_metrics:
                        val = rec.get('metrics', {}).get(m, "N/A")
                        # Format floats
                        if isinstance(val, (float, int)):
                            row.append(f"{val:.4f}")
                        else:
                            row.append(str(val))
                    md_content.append("| " + " | ".join(row) + " |")
        else:
            md_content.append("\n*No evaluation records captured.*")
            
        train_history = getattr(self, 'training_history', [])
        if train_history:
            md_content.append("\n### Training Epoch Metrics (Basic)")
            md_content.append("| Cycle | Model | Node | Epochs | Batch Size | Batch Final Accuracy | Batch Final Loss |")
            md_content.append("|---|---|---|---|---|---|---|")
            for entry in train_history:
                hist = entry.get('history', {})
                # Try getting accuracy or loss properly handle keys like 'accuracy', 'acc'
                acc_list = hist.get('accuracy', hist.get('acc', ["N/A"]))
                loss_list = hist.get('loss', ["N/A"])
                acc = acc_list[-1] if isinstance(acc_list, list) and acc_list else "N/A"
                loss = loss_list[-1] if isinstance(loss_list, list) and loss_list else "N/A"
                
                acc_str = f"{acc:.4f}" if isinstance(acc, (float, int, np.floating)) else str(acc)
                loss_str = f"{loss:.4f}" if isinstance(loss, (float, int, np.floating)) else str(loss)
                md_content.append(f"| {entry.get('cycle', 'N/A')} | {entry.get('model')} | {entry.get('node')} | {entry.get('epochs', 'N/A')} | {entry.get('batch_size', 'N/A')} | {acc_str} | {loss_str} |")
                
        # --- 4. Feature Space Visualizations ---
        if hasattr(self, 'explanation_history') and self.explanation_history:
            md_content.append("\n## 4. Feature Space Visualizations\n")
            
            # Show configuration details from the latest entry or the config itself
            conf = self.explanation_config
            method_str = conf.get("method", "umap").upper()
            n_neighbors = conf.get("n_neighbors", 50)
            min_dist = conf.get("min_dist", 0.1)
            layer = conf.get("layer_index", -2)
            
            md_content.append(f"**Configuration:** {method_str} | n_neighbors={n_neighbors} | min_dist={min_dist} | layer={layer}\n")
            
            exp_cycles = {}
            for rec in self.explanation_history:
                c = rec.get("cycle", 0)
                if c not in exp_cycles:
                    exp_cycles[c] = {}
                node = rec.get("node", "N/A")
                if node not in exp_cycles[c]:
                    exp_cycles[c][node] = {}
                exp_cycles[c][node][rec.get("timing")] = rec.get("file_path")
                
            report_dir = os.path.abspath('reports')
                
            for c in sorted(exp_cycles.keys()):
                md_content.append(f"### Cycle {c}\n")
                for node in sorted(exp_cycles[c].keys()):
                    md_content.append(f"#### Node: {node}")
                    md_content.append("| Pre-Training | Post-Training |")
                    md_content.append("|---|---|")
                    
                    pre_path = exp_cycles[c][node].get("pre-node", "")
                    post_path = exp_cycles[c][node].get("post-node", "")
                    
                    pre_rel = os.path.relpath(pre_path, report_dir) if pre_path else ""
                    post_rel = os.path.relpath(post_path, report_dir) if post_path else ""
                    
                    pre_img = f"![]({pre_rel})" if pre_rel else "N/A"
                    post_img = f"![]({post_rel})" if post_rel else "N/A"
                    
                    md_content.append(f"| {pre_img} | {post_img} |\n")
        
        if getattr(self, 'eda_enabled', False):
            try:
                eda_charts = self._generate_eda_charts(report_name)
                if eda_charts:
                    md_content.append("\n---")
                    md_content.append("\n## Exploratory Data Analysis")
                    
                    chart_a, chart_b, chart_c, chart_d = eda_charts
                    
                    if chart_a:
                        rel_a = os.path.relpath(chart_a, start=os.path.dirname(report_path))
                        md_content.append("\n### Accuracy over Training Cycles")
                        md_content.append(f"![Accuracy by Cycle]({rel_a})")
                    
                    if chart_b:
                        rel_b = os.path.relpath(chart_b, start=os.path.dirname(report_path))
                        md_content.append("\n### Accuracy by Node per Cycle")
                        md_content.append(f"![Accuracy by Node]({rel_b})")
                        
                    if chart_c:
                        rel_c = os.path.relpath(chart_c, start=os.path.dirname(report_path))
                        md_content.append("\n### Loss over Training Cycles")
                        md_content.append(f"![Loss by Cycle]({rel_c})")
                        
                    if chart_d:
                        rel_d = os.path.relpath(chart_d, start=os.path.dirname(report_path))
                        md_content.append("\n### Loss by Node per Cycle")
                        md_content.append(f"![Loss by Node]({rel_d})")
            except Exception as e:
                print(f"[ERROR] Failed to generate EDA charts: {e}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_content))
            
        print(f"\n[INFO] Markdown report successfully saved to {report_path}")

    def train_scenario_cycle(self, epochs: int = 1, batch_size: int = 32) -> bool:
        """
        Runs a complete training cycle through all nodes in the active scenario until it loops.
        Returns True if the cycle completed successfully, False otherwise.
        """
        self.generate_global_test_set()

        self.trigger_evaluation('pre-cycle')
        self.trigger_explanation('pre-cycle')

        while True:
            try:
                self.train_current_node(epochs=epochs, batch_size=batch_size)
            except RuntimeError as e:
                print(f"[ERROR] Cycle aborted: {e}")
                return False
                
            advanced = self.advance_scenario()
            if not advanced:
                node = self.get_current_node()
                print(f"[INFO] Cycle completed. Starting a new cycle. Current node: {node.name}")
                self.trigger_evaluation('post-cycle')
                self.trigger_explanation('post-cycle')
                self.current_cycle += 1
                return True
            else:
                node = self.get_current_node()
                print(f"[INFO] Proceeding to next node: {node.name}")

    def is_scenario_complete(self) -> bool:
        return self.scenario_controller.is_scenario_complete()

    def reset_scenario(self):
        self.scenario_controller.reset_active_scenario()

    def _persist_layer_activations(self, node_name: str):
        """
        Extracts and saves activation tensors of the active model.
        """

        if self.global_test_x is None or len(self.global_test_x) == 0:
            self.generate_global_test_set()
            
        if self.global_test_x is None or len(self.global_test_x) == 0:
            raise Exception(f"[WARNING] Failed to generate global_test_x. Activation extraction aborted for node {node_name}.")

        active_model = self.get_active_model()
        
        try:
            activations = self.explainer.extract_activations(model=active_model, x=self.global_test_x, layer_index=-2)
        except Exception as e:
            raise Exception(f"[ERROR] Failed to extract activations for persistence: {e}")

        exp_name = self.experiment_name if self.experiment_name else "unnamed_experiment"
        save_dir = os.path.join('reports', 'activations', exp_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        filename = f"activations_cycle{self.current_cycle}_node_{node_name}.npy"
        filepath = os.path.join(save_dir, filename)
        
        np.save(filepath, activations)
        print(f"[INFO] Penultimate-layer activations persisted to: {filepath}")

    @staticmethod
    def load_state(file_path: str = './system_state/training_state.pkl') -> 'TrainingController':
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"State file {file_path} not found.")
            
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __str__(self) -> str:
        model_lines = '\n'.join(
            [f'        {"[ACTIVE]" if name == self._active_model_name else "        "} {name} ({m.model_path})'
             for name, m in self.models.items()]
        ) or '        (none)'
        
        return f"""
        TrainingController
        Base path:       {self.base_path}
        Active Scenario: {self.scenario_controller._active_scenario_name or '(none)'}
        Active Model:    {self._active_model_name or '(none)'}
        ===== MODELS =====
{model_lines}
        """
