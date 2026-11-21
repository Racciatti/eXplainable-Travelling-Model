# XTM architecture

XTM simulates the **travelling model** paradigm: one neural network is trained
sequentially on data partitions held by different nodes (e.g. institutions),
without the data ever being centralised. After each node the model's internal
representation can be extracted and projected to 2-D so you can *see* how the
representation drifts as the model travels.

## Components

| Module | Class / function | Responsibility |
|---|---|---|
| `classes/data.py` | `Dataset`, `DataSubset`, `Node`, `DataController` | Register datasets, split them into disjoint training partitions (`uniform` or `manual` proportions), build corrupted copies, group subsets into nodes, build the global test set |
| `classes/degradation.py` | `apply_label_noise`, `apply_gaussian_blur` | Stateless data perturbations used to simulate low-quality nodes |
| `classes/scenarios.py` | `Scenario`, `ScenarioController` | An ordered list of nodes plus a traversal pointer; exactly one active scenario |
| `classes/training.py` | `Model`, `Evaluator`, `Explainer`, `TrainingController` | Training loop, metric evaluation at configurable moments, activation extraction + UMAP/t-SNE projection, Markdown report generation |
| `classes/metrics.py` | `FBetaScore`, `F1Score`, `NPV` | Custom Keras metrics |
| `main.py` | REPL | Maps text commands to controller methods and persists controller state |

```
DataController
    -> ScenarioController(data_controller)
        -> TrainingController(scenario_controller)
              |- Evaluator   (metrics at pre/post node/cycle)
              |- Explainer   (activations -> UMAP / t-SNE -> PNG)
              `- Model       (lazy-loaded Keras model)
```

## Training cycle

`TrainingController.train_scenario_cycle()`:

1. Build the global test set from the *original* test split of each node's origin dataset
   (counted once per origin dataset).
2. Optional `pre-cycle` evaluation / explanation.
3. For every node of the active scenario: `pre-node` hooks, fit on the node's
   subsets (class-weighted), persist penultimate-layer activations, `post-node` hooks, advance.
4. Optional `post-cycle` hooks, increment the cycle counter.

`generate_markdown_report()` then writes data/scenario configuration, model
architecture, metric tables, training history and the generated plots to `reports/`.

## Data safety guards

* Datasubsets are created only from the training (and validation) arrays of a dataset;
  creation aborts if a `test*` file would be included.
* The `test_*.npy` arrays stored inside every subset are intentionally empty.
* Activating a scenario checks that its nodes share no datasubset.

## On-disk layout (created at runtime, git-ignored)

```
data/datasets/<name>/{train,val,test}_{images,labels}.npy
data/datasubsets/<name>_subset_<i>/...
data/models/*.keras
reports/<experiment>.md, reports/visualizations/, reports/activations/
system_state/*.pkl
```

All paths are relative to the current working directory.

## Extending

* **New perturbation:** add a function to `classes/degradation.py` and a branch in
  `DataController.create_harmful_datasubset()`.
* **New metric:** subclass `tf.keras.metrics.Metric` in `classes/metrics.py`, register it in
  `Model.load()` (`custom_objects`) and in `TrainingController._lazy_compile_model()`.
* **New CLI command:** write a `handle_*` function in `main.py` and add it to
  `DATA_COMMANDS`, `SCENARIO_COMMANDS` or `TRAINING_COMMANDS`.
