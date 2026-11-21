import os
import pickle
import numpy as np
from classes.degradation import apply_label_noise, apply_gaussian_blur

class Dataset:

    def __init__(self, name:str, path:str, description:str=""):

        self.name = name
        self.path = os.path.abspath(path)
        self.description = description

        dataset_metadata = self.__read_dataset()
        self.n_classes   = dataset_metadata['n_classes']
        self.n_examples_train  = dataset_metadata['n_examples_train']
        self.n_examples_test  = dataset_metadata['n_examples_test']
        self.class_names = dataset_metadata['class_names']
        self.n_examples_per_class = dataset_metadata['n_examples_per_class']
        self.structure   = dataset_metadata.get('structure', 'traditional')
    
    def __read_dataset(self):    
        """
        This functions validates the path passed as an argument to the constructor by attempting to 
        read the dataset on the provided path. 
        """
        
        if not os.path.isdir(self.path):
            raise ValueError(f'Invalid directory: {self.path}')
            
        # Check if it's a MedMNIST format dataset by looking for .npy files
        files_in_dir = os.listdir(self.path)
        npy_files = [f for f in files_in_dir if f.endswith('.npy')]
        
        required_medmnist_files = ['train_images.npy', 'train_labels.npy', 'test_images.npy', 'test_labels.npy']
        
        # If at least the required train and test files exist, treat as MedMNIST
        has_req_medmnist = all(req in npy_files for req in required_medmnist_files)
        
        if len(npy_files) > 0 and has_req_medmnist:
            return self.__read_medmnist_dataset()
        else:
            return self.__read_traditional_dataset()

    def __read_traditional_dataset(self):
        dataset_metadata = {}
        
        # Get the list of subdirectories in the main directory
        subdir_names = [name for name in os.listdir(self.path) if os.path.isdir(f"{self.path}/{name}")]
        
        # Register the number of classes, their names and the number of examples per class as dataset attributes
        dataset_metadata['class_names'] = subdir_names
        dataset_metadata['n_classes'] = len(subdir_names)
        dataset_metadata['n_examples_per_class'] = {class_name : 0 for class_name in dataset_metadata['class_names']}

        # Go through each example inside the classes, counting the number of examples (file extensions are not validated)
        for class_name in dataset_metadata['class_names']:
            examples_in_subdir = os.listdir(f"{self.path}/{class_name}")
            
            if len(examples_in_subdir) == 0:
                raise Exception(f'Empty subdirectory: {self.path}/{class_name}')
            
            
            dataset_metadata['n_examples_per_class'][class_name] = len(examples_in_subdir)

        # Define dataset attributes
        dataset_metadata['n_examples'] = sum(list(dataset_metadata['n_examples_per_class'].values()))
        dataset_metadata['structure'] = 'traditional'
        
        return dataset_metadata

    def __read_medmnist_dataset(self):
        
        dataset_metadata = {}
        has_val_data = False
        
        # Wipe the dir files
        label_files = [f for f in os.listdir(self.path) if f.endswith('_labels.npy')]
        image_files = [f for f in os.listdir(self.path) if f.endswith('_images.npy')]

        # Separate train/test splits and put the validation set together with the train set, if it exists
        test_label_files = [l for l in label_files if l.startswith('test')]
        test_image_files = [l for l in image_files if l.startswith('test')]

        train_label_files = [l for l in label_files if l.startswith('train')]
        train_image_files = [l for l in image_files if l.startswith('train')]

        val_label_files = [l for l in label_files if l.startswith('val')]
        val_image_files = [l for l in image_files if l.startswith('val')]

        lens = [
            len(test_image_files),
            len(test_label_files),
            len(train_image_files),
            len(train_label_files)
            ]

        if 0 in lens:
            raise ValueError('one of the test or train sets is empty!: ')
        
        if len(val_image_files) != 0 and len(val_label_files) != 0:

            print('[INFO] Validation set spotted! Adding it to the training set...')

            has_val_data = True
        

        # Checking if we have a 1:1 mapping between image and label files 
        missing_files = []

        for lf in label_files:
            prefix = lf.split('_labels.npy')[0]
            
            if f"{prefix}_images.npy" not in image_files:
                missing_files.append(f"{prefix}_images.npy")
        
        for imgf in image_files:
            
            prefix = imgf.split('_images.npy')[0]
            
            if f"{prefix}_labels.npy" not in label_files:
                missing_files.append(f"{prefix}_labels.npy")
                
        if missing_files:
            print(f"[WARNING] Missing corresponding image/label files in dataset {self.name}: {missing_files}")
            
        all_labels = {}
        for lf in label_files:
            
            try:
            
                # Load labels
                labels = np.load(os.path.join(self.path, lf), mmap_mode='r')
                
                # Verify corresponding images exist and match in size
                img_file = lf.replace('_labels.npy', '_images.npy')

                if img_file in image_files:
                    
                    images = np.load(os.path.join(self.path, img_file), mmap_mode='r')
                    
                    if len(images) != len(labels):
                        
                        print(f"[WARNING] Example count mismatch between {lf} ({len(labels)}) and {img_file} ({len(images)})")
                        
                if lf in test_label_files:
                    all_labels['test'] = labels.flatten()
                else:
                    all_labels['train'] = labels.flatten() if 'train' not in all_labels else np.concat([all_labels['train'], labels.flatten()])



            except Exception as e:
                print(f"[WARNING] Could not load labels from {lf}: {e}")
                
        if not all_labels:
            raise Exception(f"No valid label files could be read in {self.path}")
            
        concatenated_labels = np.concatenate([all_labels[name] for name in all_labels])

        unique_classes, class_counts = np.unique(concatenated_labels, return_counts=True)
        
        unique_classes = sorted(unique_classes.tolist())
        dataset_metadata['class_names'] = [str(cls) for cls in unique_classes]
        dataset_metadata['n_classes'] = len(unique_classes)
        dataset_metadata['n_examples_per_class'] = {
            str(cls): int(count) for cls, count in zip(unique_classes, class_counts)
        }

        dataset_metadata['n_examples_train'] = all_labels['train'].shape[0]
        dataset_metadata['n_examples_test'] =  all_labels['test'].shape[0] 
        
        dataset_metadata['structure'] = 'medmnist'
        
        return dataset_metadata

    def __str__(self):
        string = f""" 
        Name: {self.name}                
        Path: {self.path}                
        N° of train examples: {self.n_examples_train}
        N° of test examples: {self.n_examples_test}
        Classes: {self.class_names}
        Examples per class: {self.n_examples_per_class}
        """

        return string

class DataSubset(Dataset):

    def __init__(self, origin_dataset_name:str, origin_dataset_path:str, name:str, path:str, description:str = None):
        super().__init__(name=name, path=path,description=description)
        
        self.origin_dataset_name = origin_dataset_name
        self.origin_dataset_path = origin_dataset_path
        

    def __str__(self):
        string = f""" 
        Name: {self.name}                
        Path: {self.path}                
        N° of train examples: {self.n_examples_train}
        N° of test examples: {self.n_examples_test}
        Classes: {self.class_names}
        Examples per class: {self.n_examples_per_class}
        Origin dataset path: {self.origin_dataset_path}
        Origin dataset name: {self.origin_dataset_name}
        """

        return string

class Node:
    """
    A node is the logical representation of what would be a data silo in real life, such as medical clinics. 
    """
    def __init__(self, name:str, description:str):
        self.name = name
        self.description = description
        
        # For this object, it contains datasets. In the grand scheme, it is a subset of a dataset.
        self.datasets = {}

    def add_dataset(self, datasubset:DataSubset):
        self.datasets[datasubset.name] = datasubset
        
    def remove_dataset(self, datasubset_name:str):
        
        if datasubset_name not in list(self.datasets.keys()):
            raise ValueError('Datasubset could not be removed because it does not exist in this node')
        
        del self.datasets[datasubset_name]
        
    def discard_all_datasets(self):
        self.datasets = {}
    
    def get_n_examples(self)->int:
        # Get the total amount of train + test examples in this node (all subsets it contains)
        total_examples =  sum([self.datasets[dataset].n_examples_train + self.datasets[dataset].n_examples_test for dataset in self.datasets])
        return total_examples
            
    
    def __str__(self):
        return f"""
        Name: {self.name}
        Description:{self.description}
        Datasets contained:
        {'\n            '.join([dataset_name for dataset_name in self.datasets])} 
        """

class DataController:

    """Registry and on-disk manager for datasets, datasubsets and nodes.

    Datasets are canonical datasets (MedMNIST-style ``.npy`` files or one directory per
    class). Datasubsets are disjoint training partitions derived from a dataset and are
    written under ``<base_path>/<datasubsets_dir_name>``. Nodes group datasubsets and
    model a data silo (e.g. an institution).
    """

    def __init__(self, base_path:str='./data/', datasets_dir_name:str='datasets', datasubsets_dir_name:str='datasubsets'):

        self.SUPPORTED_SPLIT_METHODS = ['uniform', 'manual']

        self.base_path = base_path
        self.datasets_path = base_path + datasets_dir_name
        self.datasubsets_path = base_path + datasubsets_dir_name
        self.datasets = {}
        self.datasubsets = {}
        self.nodes = {}

        # Create the storage layout on first use so a fresh checkout works out of the box
        os.makedirs(self.datasets_path, exist_ok=True)
        os.makedirs(self.datasubsets_path, exist_ok=True)

    def __verify_path_integrity(self):
        "Verifies if the base paths and implicit folders contained in this class exist"
        
        if not os.path.isdir(self.base_path):
            raise FileNotFoundError(f'Base path does not exist: {self.base_path}')

        if not os.path.isdir(self.datasets_path):
            raise FileNotFoundError(f'Datasets directory does not exist: {self.datasets_path}')
        
        if not os.path.isdir(self.datasubsets_path):
            raise FileNotFoundError(f'Datasubsets directory does not exist: {self.datasubsets_path}')

    def __verify_datasets_integrity(self):
        """
        Verifies if the dataset objects match the available data on disk, regarding the names, structure, paths, etc.
        """
        
        # Since a Dataset object's constructor already validates the structure of a dataset
        # and defines its attributes based on that structure, we can just build another dataset object
        # and compare its attributes to the object we have 
        
        invalid_datasets = []

        for dataset_name in self.datasets:
            dataset_obj = self.datasets[dataset_name]

            try:
                new_dataset_obj = Dataset(name='obj_for_validation', path=dataset_obj.path, description=dataset_obj.description)
            except Exception as e:
                print(f'[WARNING] Dataset "{dataset_name}" failed re-validation and will be removed: {e}')
                invalid_datasets.append(dataset_name)
                continue

            dataset_obj_attrs = dataset_obj.__dict__.copy()
            new_dataset_obj_attrs = new_dataset_obj.__dict__.copy()

            # The name is known to be different 
            del dataset_obj_attrs['name']
            del new_dataset_obj_attrs['name']

            for attr_name in dataset_obj_attrs:
                if attr_name not in new_dataset_obj_attrs:
                    continue
                if dataset_obj_attrs[attr_name] != new_dataset_obj_attrs[attr_name]:
                    print(f'[WARNING] Dataset "{dataset_name}" has a mismatch on attribute "{attr_name}" and will be removed.')
                    invalid_datasets.append(dataset_name)
                    break

        for dataset_name in invalid_datasets:
            del self.datasets[dataset_name]

    def __verify_datasubsets_integrity(self):
        # Datasubsets' integrity can be verified similarly to the way datasets were verified
        # The only difference is we also need to check the origin dataset's path and object existence   

        invalid_datasubsets = []

        for datasubset_name in self.datasubsets:
            datasubset_obj = self.datasubsets[datasubset_name]

            # Check that the origin dataset path still exists on disk
            if not os.path.isdir(datasubset_obj.origin_dataset_path):
                print(f'[WARNING] Datasubset "{datasubset_name}" origin dataset path no longer exists and will be removed: {datasubset_obj.origin_dataset_path}')
                invalid_datasubsets.append(datasubset_name)
                continue

            # Check that the origin dataset object is still registered in memory
            if datasubset_obj.origin_dataset_name not in self.datasets:
                print(f'[WARNING] Datasubset "{datasubset_name}" origin dataset "{datasubset_obj.origin_dataset_name}" is not loaded and will be removed.')
                invalid_datasubsets.append(datasubset_name)
                continue

            # Re-validate the datasubset's own structure against what is currently on disk
            try:
                new_datasubset_obj = DataSubset(
                    origin_dataset_name=datasubset_obj.origin_dataset_name,
                    origin_dataset_path=datasubset_obj.origin_dataset_path,
                    name='obj_for_validation',
                    path=datasubset_obj.path,
                    description=datasubset_obj.description
                )
            except Exception as e:
                # If any datasubsets are invalid, raise a warning and remove them from memory
                print(f'[WARNING] Datasubset "{datasubset_name}" failed re-validation and will be removed: {e}')
                invalid_datasubsets.append(datasubset_name)
                continue

            datasubset_obj_attrs = datasubset_obj.__dict__.copy()
            new_datasubset_obj_attrs = new_datasubset_obj.__dict__.copy()

            # The name is known to be different 
            del datasubset_obj_attrs['name']
            del new_datasubset_obj_attrs['name']

            for attr_name in datasubset_obj_attrs:
                if attr_name not in new_datasubset_obj_attrs:
                    continue
                if datasubset_obj_attrs[attr_name] != new_datasubset_obj_attrs[attr_name]:
                    # If any datasubsets are invalid, raise a warning and remove them from memory
                    print(f'[WARNING] Datasubset "{datasubset_name}" has a mismatch on attribute "{attr_name}" and will be removed.')
                    invalid_datasubsets.append(datasubset_name)
                    break

        for datasubset_name in invalid_datasubsets:
            del self.datasubsets[datasubset_name]

    def __verify_nodes_integrity(self):
        # Verifying nodes integrity is essentially making sure the datasubsets they point to exist

        for node_name in self.nodes:
            node_obj = self.nodes[node_name]
            invalid_datasets = []

            for datasubset in node_obj.datasets:
                # Check that the datasubset the node points to is still registered in memory
                if node_obj.datasets[datasubset].name not in self.datasubsets:
                    print('[WARNING] Node references a datasubset which no longer exists and will be removed from the node.')
                    invalid_datasets.append(datasubset)

            for datasubset in invalid_datasets:
                del node_obj.datasets[datasubset]

    def verify_integrity(self):
        self.__verify_path_integrity()
        self.__verify_datasets_integrity()
        self.__verify_datasubsets_integrity()
        self.__verify_nodes_integrity()
        
        print('Integrity verified')

    def save_all_data(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def get_dataset(self, name:str)->Dataset:
        if name not in list(self.datasets.keys()):
            raise ValueError('REQUESTED DATASET NOT FOUND')

        return self.datasets[name]

    def add_dataset(self, name:str, path:str):
        new_dataset = Dataset(name=name, path=path)
        self.datasets[new_dataset.name] = new_dataset

    def add_node(self, node_name:str, node_description:str):
        
        if node_name in self.nodes: 
            raise ValueError('[WARN] Node name already exists! Aborting operation...')
        
        self.nodes[node_name] = Node(node_name, node_description)
    
    def add_subset_to_node(self, datasubset_name:str, node_name:str):

        if datasubset_name not in list(self.datasubsets.keys()):
            raise ValueError(f'Datasubset "{datasubset_name}" does not exist')
        
        if node_name not in list(self.nodes.keys()):
            raise ValueError(f'Node "{node_name}" does not exist')
        
        self.nodes[node_name].datasets

        self.nodes[node_name].add_dataset(self.datasubsets[datasubset_name])

    def remove_node(self, node_name:str):
        if node_name not in list(self.nodes.keys()):
            raise NameError(f'Node name {node_name} does not exist')

        del self.nodes[node_name]

    def edit_dataset(self, old_name:str, new_name:str, new_description:str):
        if old_name not in self.datasets:
            raise NameError(f'Dataset name {old_name} does not exist')
        if new_name != old_name and new_name in self.datasets:
            raise ValueError(f'Dataset with name {new_name} already exists')
        
        dataset = self.datasets.pop(old_name)
        dataset.name = new_name
        if new_description is not None:
            dataset.description = new_description
        self.datasets[new_name] = dataset
        
        for sub in self.datasubsets.values():
            if sub.origin_dataset_name == old_name:
                sub.origin_dataset_name = new_name

    def edit_datasubset(self, old_name:str, new_name:str, new_description:str):
        if old_name not in self.datasubsets:
            raise NameError(f'Datasubset name {old_name} does not exist')
        if new_name != old_name and new_name in self.datasubsets:
            raise ValueError(f'Datasubset with name {new_name} already exists')
        
        subset = self.datasubsets.pop(old_name)
        subset.name = new_name
        if new_description is not None:
            subset.description = new_description
        self.datasubsets[new_name] = subset
        
        for node in self.nodes.values():
            if old_name in node.datasets:
                node.datasets[new_name] = node.datasets.pop(old_name)

    def edit_node(self, old_name:str, new_name:str, new_description:str):
        if old_name not in self.nodes:
            raise NameError(f'Node name {old_name} does not exist')
        if new_name != old_name and new_name in self.nodes:
            raise ValueError(f'Node with name {new_name} already exists')
        
        node = self.nodes.pop(old_name)
        node.name = new_name
        if new_description is not None:
            node.description = new_description
        self.nodes[new_name] = node

    def remove_dataset(self, dataset_name:str):
        if dataset_name not in self.datasets:
            raise NameError(f'Dataset name {dataset_name} does not exist')
        subsets_to_remove = [k for k, v in self.datasubsets.items() if v.origin_dataset_name == dataset_name]
        for sub in subsets_to_remove:
            self.remove_datasubset(sub)
        del self.datasets[dataset_name]

    def remove_datasubset(self, datasubset_name:str):
        if datasubset_name not in self.datasubsets:
            raise NameError(f'Datasubset name {datasubset_name} does not exist')
        obj = self.datasubsets[datasubset_name]
        for node in self.nodes.values():
            if obj.name in node.datasets:
                del node.datasets[obj.name]
        del self.datasubsets[datasubset_name]

    def remove_datasubset_from_node(self,node_name:str, datasubset_name:str):
        if node_name not in self.nodes:
            raise NameError(f'Node name {node_name} does not exist')
        
        node = self.nodes[node_name]

        if datasubset_name not in node.datasets:
            raise NameError(f'Datasubset name {datasubset_name} does not exist')
        
        del node.datasets[datasubset_name]

    def create_harmful_datasubset(self, source_datasubset_name: str, technique: str, technique_params: dict) -> DataSubset:
        if source_datasubset_name not in self.datasubsets:
            raise ValueError(f"Origin datasubset '{source_datasubset_name}' does not exist.")
            
        source_subset = self.datasubsets[source_datasubset_name]
        origin_dataset = self.get_dataset(source_subset.origin_dataset_name)
        
        if origin_dataset.structure != 'medmnist':
            raise NotImplementedError("Creating harmful datasubsets is currently only supported for medmnist datasets (.npy format).")
            
        if technique == 'label_noise':
            p = technique_params.get('p', 0.0)
            if not isinstance(p, float) or not (0.0 < p <= 1.0):
                raise ValueError("Noise probability 'p' must be a float in (0.0, 1.0]")
            suffix = f"lnoise_p{p}"
        elif technique == 'gaussian_blur':
            kernel_size = technique_params.get('kernel_size', 0)
            if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
                raise ValueError("Kernel size must be an odd integer >= 3")
            suffix = f"mblur_k{kernel_size}"
        else:
            raise ValueError(f"Technique '{technique}' is not supported.")
            
        new_name = f"{source_datasubset_name}_{suffix}"
        if new_name in self.datasubsets:
            raise ValueError(f"Datasubset '{new_name}' already exists.")
            
        new_path = os.path.join(self.datasubsets_path, new_name)
        
        print(f"[INFO] Applying technique \"{technique}\" to datasubset \"{source_datasubset_name}\"...")
        
        os.mkdir(new_path)
        
        train_images_path = os.path.join(source_subset.path, "train_images.npy")
        train_labels_path = os.path.join(source_subset.path, "train_labels.npy")
        test_images_path = os.path.join(source_subset.path, "test_images.npy")
        test_labels_path = os.path.join(source_subset.path, "test_labels.npy")
        
        train_images = np.load(train_images_path) if os.path.exists(train_images_path) else np.array([])
        train_labels = np.load(train_labels_path) if os.path.exists(train_labels_path) else np.array([])
        test_images = np.load(test_images_path) if os.path.exists(test_images_path) else np.array([])
        test_labels = np.load(test_labels_path) if os.path.exists(test_labels_path) else np.array([])
        
        if technique == 'label_noise':
            train_labels = apply_label_noise(train_labels, p)
        elif technique == 'gaussian_blur':
            if train_images.size > 0:
                train_images = apply_gaussian_blur(train_images, kernel_size)
                
        np.save(os.path.join(new_path, "train_images.npy"), train_images)
        np.save(os.path.join(new_path, "train_labels.npy"), train_labels)
        np.save(os.path.join(new_path, "test_images.npy"), test_images)
        np.save(os.path.join(new_path, "test_labels.npy"), test_labels)
        
        new_datasubset = DataSubset(
            origin_dataset_name=source_subset.origin_dataset_name,
            origin_dataset_path=source_subset.origin_dataset_path,
            name=new_name,
            path=new_path,
            description=f"Degraded version of {source_datasubset_name} using {technique}."
        )
        
        self.datasubsets[new_name] = new_datasubset
        print(f"[INFO] Harmful datasubset \"{new_name}\" created successfully.")
        
        return new_datasubset

    def create_datasubsets_from_dataset(self, dataset_name:str, n_subsets:int, split_method:str='uniform', data_subsets_name:str = None, partitions:list = None)->list:

        # PARAMETER VALIDATION
        if split_method not in self.SUPPORTED_SPLIT_METHODS:
            raise (ValueError(f'Split method "{split_method}" not supported.'))

        if self.get_dataset(dataset_name) == None:
            raise Exception(f'dataset with name "{dataset_name}" does not exist')
        
        if data_subsets_name is None:
            data_subsets_name = f"{dataset_name}_{split_method}"

        # GET SPLIT INDEXES 
        if split_method == 'uniform':
            split_indexes = self.__get_uniform_split_indexes(dataset_name, n_subsets)
        elif split_method == 'manual':
            split_indexes = self.__get_manual_split_indexes(dataset_name, n_subsets, partitions)
        
        # CREATE THE DATASUBSETS THEMSELVES
        new_datasubsets = self.__create_datasubsets(n_subsets=n_subsets, split_indexes=split_indexes, dataset_name=dataset_name, datasubsets_name=data_subsets_name)

        # SAVE THEM IN THE DATASUBSETS HASH TABLE (attr)
        for new_datasubset in new_datasubsets:
            self.datasubsets[new_datasubset.name] = new_datasubset

    def __get_uniform_split_indexes(self, dataset_name:str, n_subsets:int)->dict:
        
        dataset = self.get_dataset(dataset_name)
        dataset_splits = {}

        if dataset.structure == 'traditional':
            for class_name in dataset.class_names:
                
                # Get the number of examples for this class in the dataset #200
                n_examples = dataset.n_examples_per_class[class_name]

                # Define the split indexes for this class
                split_start_indexes = [round(n_examples / n_subsets*i) for i in range(n_subsets+1)] # 0, 40, 80, 120, 160, 200

                split_indexes = [(split_start_indexes[i], split_start_indexes[i+1]) for i in range(len(split_start_indexes)-1)]

                # Save the split indexes in order to create the datasubsets later on
                dataset_splits[class_name] = split_indexes
                
        elif dataset.structure == 'medmnist':
            
            label_files = [f for f in os.listdir(dataset.path) if (f.endswith('_labels.npy')) and not f.startswith('test')]
            label_files.sort()
            
            all_labels = []
            
            for lf in label_files:
                labels = np.load(os.path.join(dataset.path, lf), mmap_mode='r')
                all_labels.append(labels.flatten())
            
            all_labels_concat = np.concatenate(all_labels)

            for class_name in dataset.class_names:
                class_int = int(class_name)
                class_indices = np.where(all_labels_concat == class_int)[0]
                n_examples = len(class_indices)
                
                split_start_indexes = [round(n_examples / n_subsets*i) for i in range(n_subsets+1)]
                
                subset_indices = []
                for i in range(n_subsets):
                    start_idx = split_start_indexes[i]
                    end_idx = split_start_indexes[i+1]
                    subset_indices.append(class_indices[start_idx:end_idx].tolist())
                dataset_splits[class_name] = subset_indices
        
        return dataset_splits

    def __get_manual_split_indexes(self, dataset_name:str, n_subsets:int, partitions:list, min_examples_per_datasubset:int=0)->dict:
        if not partitions or len(partitions) != n_subsets:
            raise ValueError("Partitions list must be provided and its length must equal the number of subsets.")
        if abs(sum(partitions) - 1.0) > 1e-5:
            raise ValueError("The sum of partitions must be approximately 1.0.")
            
        dataset = self.get_dataset(dataset_name)
        dataset_splits = {}

        if dataset.structure == 'traditional':
            for class_name in dataset.class_names:
                n_examples = dataset.n_examples_per_class[class_name]
                
                split_start_indexes = [0]
                current_idx = 0
                for i in range(n_subsets - 1):
                    count = round(n_examples * partitions[i])
                    current_idx += count
                    split_start_indexes.append(current_idx)
                
                # The last index should be the total number of examples to avoid rounding issues
                split_start_indexes.append(n_examples)
                
                split_indexes = [(split_start_indexes[i], split_start_indexes[i+1]) for i in range(len(split_start_indexes)-1)]
                dataset_splits[class_name] = split_indexes
                
        elif dataset.structure == 'medmnist':
            
            label_files = [f for f in os.listdir(dataset.path) if (f.endswith('_labels.npy') and not f.startswith('test'))]

            # ASSERT no files that were selected contain the word 'test'
            if any(['test' in l.lower() for l in label_files]):
                raise Exception('[WARNING] TEST DATA was being loaded into a datasubset! Aborted operation.')
            
            label_files.sort()
            all_labels = []

            for lf in label_files:
                labels = np.load(os.path.join(dataset.path, lf), mmap_mode='r')
                all_labels.append(labels.flatten())
            all_labels_concat = np.concatenate(all_labels)

            for class_name in dataset.class_names:
                class_int = int(class_name)
                class_indices = np.where(all_labels_concat == class_int)[0]
                n_examples = len(class_indices)
                
                split_start_indexes = [0]
                current_idx = 0
                for i in range(n_subsets - 1):
                    count = round(n_examples * partitions[i])
                    current_idx += count
                    split_start_indexes.append(current_idx)
                
                split_start_indexes.append(n_examples)
                
                subset_indices = []
                for i in range(n_subsets):
                    start_idx = split_start_indexes[i]
                    end_idx = split_start_indexes[i+1]
                    subset_indices.append(class_indices[start_idx:end_idx].tolist())
                dataset_splits[class_name] = subset_indices

        return dataset_splits

    def __create_datasubsets(self, n_subsets:int, split_indexes:dict, dataset_name:str, datasubsets_name:str):

        dataset = self.get_dataset(dataset_name)
        datasubsets = []

        if dataset.structure == 'traditional':
            # Validate the indexes and classes based on the files and dirs and dataset attributes
            for class_name in dataset.class_names:
                assert class_name in list(split_indexes.keys())
                assert dataset.n_examples_per_class[class_name] == split_indexes[class_name][-1][1]

            # For each datasubset to be created
            for i in range(n_subsets):

                # Create its subdirectory
                os.mkdir(f"{self.datasubsets_path}/{datasubsets_name}_subset_{i}")
            
                # For each class
                for class_name in dataset.class_names:

                    # Create the class dir inside the subset dir
                    os.mkdir(f"{self.datasubsets_path}/{datasubsets_name}_subset_{i}/{class_name}")

                    # Get the file names inside the class directory
                    example_names = os.listdir(f"{dataset.path}/{class_name}")

                    assert len(example_names) == dataset.n_examples_per_class[class_name]

                    # Get the split indexes of this particular subset
                    start_idx, end_idx = split_indexes[class_name][i]
                    examples_to_copy = example_names[start_idx:end_idx]

                    # And copy the indexed data from the dataset to the subset
                    for example_name in examples_to_copy:
                        os.system(f'cp "{dataset.path}/{class_name}/{example_name}" "{self.datasubsets_path}/{datasubsets_name}_subset_{i}/{class_name}/{example_name}"')

                # Create a datasubset object
                new_datasubset = DataSubset(
                    origin_dataset_name=dataset_name, 
                    origin_dataset_path=dataset.path, 
                    name=f"{datasubsets_name}_subset_{i}", 
                    path=f"{self.datasubsets_path}/{datasubsets_name}_subset_{i}"
                    )

                # And store it
                datasubsets.append(new_datasubset)
                
        elif dataset.structure == 'medmnist':

            # Get the label file names for all sets **except** the test set 
            label_files = [f for f in os.listdir(dataset.path) if (f.endswith('_labels.npy') and not f.startswith('test'))]

            # ASSERT no files that were selected contain the word 'test'
            if any(['test' in l.lower() for l in label_files]):
                raise Exception('[WARNING] TEST DATA was being loaded into a datasubset! Aborted operation.')

            label_files.sort()
            
            all_images = []
            all_labels = []
            
            # For each label file name, get the corresponding image file, load both of them and add the data to their list  
            for lf in label_files:
                img_file = lf.replace('_labels.npy', '_images.npy')
                images = np.load(os.path.join(dataset.path, img_file), mmap_mode='r')
                labels = np.load(os.path.join(dataset.path, lf), mmap_mode='r')
                all_images.append(images)
                all_labels.append(labels.flatten())
            
            all_images_concat = np.concatenate(all_images)
            all_labels_concat = np.concatenate(all_labels)

            for i in range(n_subsets):
                subset_dir = f"{self.datasubsets_path}/{datasubsets_name}_subset_{i}"
                os.mkdir(subset_dir)
                
                subset_indices = []
                for class_name in dataset.class_names:
                    subset_indices.extend(split_indexes[class_name][i])
                    
                subset_indices = sorted(subset_indices)
                
                subset_images = all_images_concat[subset_indices]
                subset_labels = all_labels_concat[subset_indices].reshape(-1, 1) if len(all_labels[0].shape) > 1 else all_labels_concat[subset_indices]
                
                # We save train subsets. In order to pass Dataset format check later on, we make empty test sets.
                np.save(os.path.join(subset_dir, "train_images.npy"), subset_images)
                np.save(os.path.join(subset_dir, "train_labels.npy"), subset_labels)
                np.save(os.path.join(subset_dir, "test_images.npy"), np.array([], dtype=subset_images.dtype).reshape((0,) + subset_images.shape[1:]))
                np.save(os.path.join(subset_dir, "test_labels.npy"), np.array([], dtype=subset_labels.dtype).reshape((0,) + subset_labels.shape[1:]))
                
                new_datasubset = DataSubset(
                    origin_dataset_name=dataset_name, 
                    origin_dataset_path=dataset.path, 
                    name=f"{datasubsets_name}_subset_{i}", 
                    path=subset_dir
                )
                datasubsets.append(new_datasubset)

        # After saving and creating all datasubsets, return the objects (that contain references to the dirs 
        return datasubsets

    def get_global_test_set(self, active_node_names: list) -> tuple:
        """
        Loads and concatenates the testing subsets of all active nodes in the provided scenario.
        For datasubsets that originate from MedMNIST, it uses the origin dataset's 
        test data to avoiding data leakage from the training partitions.
        Returns:
            global_test_x (np.ndarray), global_test_y (np.ndarray)
        """

        print('[INFO] Getting global test set...')

        all_test_x = []
        all_test_y = []
        
        # Track which origin datasets have defined test sets to avoid duplicating the
        # testing data when multiple node datasubsets come from the same origin.
        included_origins = set()
        
        for node_name in active_node_names:
            if node_name not in self.nodes:

                raise Exception(f'Node "{node_name}" in TrainingController but not in DataController!')

            node = self.nodes[node_name]

            for datasubset_name, datasubset in node.datasets.items():
                origin_name = datasubset.origin_dataset_name
                
                if origin_name in included_origins:
                    continue
                    
                origin_dataset = self.get_dataset(origin_name)
                
                if origin_dataset.structure == 'medmnist':

                    # Use the original untouched test set
                    test_x_path = os.path.join(origin_dataset.path, "test_images.npy")
                    test_y_path = os.path.join(origin_dataset.path, "test_labels.npy")
                    
                    if os.path.exists(test_x_path) and os.path.exists(test_y_path):
                        print(f'[INFO] Loading data from {test_x_path} and {test_y_path} into the global test set')
                        test_x = np.load(test_x_path, mmap_mode='r')
                        test_y = np.load(test_y_path, mmap_mode='r')
                        
                        all_test_x.append(test_x)
                        all_test_y.append(test_y)
                        
                elif origin_dataset.structure == 'traditional':
                    print(f"[WARNING] Traditional dataset testing concatenation not natively supported yet for {origin_name}.")
                    
                included_origins.add(origin_name)
                
        if not all_test_x:
            # If nothing was found, raise
            raise Exception("[ERROR] No test data was loaded!")
            
        global_test_x = np.concatenate(all_test_x, axis=0)
        global_test_y = np.concatenate(all_test_y, axis=0)
        
        return global_test_x, global_test_y

    @staticmethod
    def load(file_path: str = './data_state.pkl') -> 'DataController':
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        string = f"""
        base path: {self.base_path}
        datasets path: {self.datasets_path}
        subdatasets path: {self.datasubsets_path}
        ===== DATASETS LOADED =====
        {"\n\n".join([self.datasets[name].__str__() for name in self.datasets])}
        ===== DATASUBSETS LOADED =====
        {"\n\n".join([self.datasubsets[name].__str__() for name in self.datasubsets])}
        ===== NODES BUILT =====
        {"\n\n".join([self.nodes[name].__str__() for name in self.nodes])}

        """
        return string
