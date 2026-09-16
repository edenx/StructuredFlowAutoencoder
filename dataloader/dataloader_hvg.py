import os

import numpy as np
import numpy.random as npr
import pandas as pd
import torch
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix, issparse
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler, Subset

# --- 0. Define Classes and Functions ---

class AnnDataFeaturesLabels:
    """
    Loads an AnnData object from an .h5ad file and prepares features (X or layer)
    and labels (from obs) for use in a PyTorch Dataset.
    """
    def __init__(self, h5ad_path, label_obs_key, feature_layer=None):
        """
        Args:
            h5ad_path (str): Path to the .h5ad file.
            label_obs_key (str): The key in adata.obs to use for labels (e.g., 'cell_type').
            feature_layer (str, optional): The key in adata.layers to use for features.
                                           If None, adata.X will be used. Defaults to None.
        """
        print(f"Loading AnnData object from: {h5ad_path}")
        try:
            self.adata = ad.read_h5ad(h5ad_path)
            print("AnnData object loaded successfully.")
        except FileNotFoundError:
            print(f"ERROR: File not found at {h5ad_path}. Please check the path.")
            print("Using dummy data for demonstration instead.")
            self._create_dummy_adata() # Fallback to dummy data if file not found

        print(f"Using '{label_obs_key}' from adata.obs as labels.")
        if label_obs_key not in self.adata.obs.columns:
            raise KeyError(f"Label key '{label_obs_key}' not found in adata.obs. Available keys: {list(self.adata.obs.columns)}")

        # Prepare features (X or a layer)
        if feature_layer:
            if feature_layer not in self.adata.layers.keys():
                raise KeyError(f"Feature layer '{feature_layer}' not found in adata.layers. Available layers: {list(self.adata.layers.keys())}")
            feature_data_source = self.adata.layers[feature_layer]
            print(f"Using layer '{feature_layer}' for features.")
        else:
            feature_data_source = self.adata.X
            print("Using adata.X for features.")

        if issparse(feature_data_source):
            self.features = feature_data_source.toarray().astype(np.float32)
        else:
            self.features = np.asarray(feature_data_source, dtype=np.float32) # Ensure it's a NumPy array

        # Numerically encode labels
        unique_labels = self.adata.obs[label_obs_key].unique()
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        self.labels = self.adata.obs[label_obs_key].map(self.label_to_int).values.astype(np.int64)

        print("\n--- Data Preparation within AnnDataFeaturesLabels ---")
        print(f"Shape of features: {self.features.shape}")
        print(f"Shape of labels: {self.labels.shape}")
        print(f"Label to integer mapping: {self.label_to_int}")
        print(f"First 5 integer labels: {self.labels[:5]}")
        print(f"Original labels for first 5: {self.adata.obs[label_obs_key].values[:5]}")
        print("-" * 30)

    def _create_dummy_adata(self):
        """Creates a dummy AnnData object if the specified file is not found."""
        n_cells = 200
        n_genes = 100
        X_data = csr_matrix(np.random.poisson(1, size=(n_cells, n_genes)).astype(np.float32))
        obs_data = pd.DataFrame({
            'cell_type': np.random.choice(['T cell', 'B cell', 'Monocyte', 'NK cell'], size=n_cells),
            # Ensure the dummy split key matches what might be expected later
            'split_condition': np.random.choice(['train', 'valid', 'test'], size=n_cells, p=[0.6, 0.2, 0.2])
        })
        obs_data.index = [f"Cell_{i}" for i in range(n_cells)]
        var_data = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)])
        self.adata = ad.AnnData(X=X_data, obs=obs_data, var=var_data)
        print("--- Dummy AnnData Object Created for AnnDataFeaturesLabels ---")
        print(self.adata)


class TorchDatasetFromPrepared(Dataset):
    """
    Custom PyTorch Dataset that takes pre-processed features and labels.
    """
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_dataloaders_from_adata_splits(
    adata_prepared_obj: AnnDataFeaturesLabels,
    split_obs_key: str,
    train_val: str = "train",
    test_val: str = "test",
    valid_val: str = None, # Optional: if you have a separate validation set
    batch_size: int = 32,
    num_workers: int = 0
):
    """
    Creates train, validation (optional), and test DataLoaders from an AnnDataFeaturesLabels
    object based on a split column in adata.obs.

    Args:
        adata_prepared_obj (AnnDataFeaturesLabels): An instance of AnnDataFeaturesLabels
                                                   containing the loaded adata, features, and labels.
        split_obs_key (str): The key in adata.obs that contains the split labels
                             (e.g., 'split_B', 'split_condition').
        train_val (str): The value in `split_obs_key` identifying training samples.
        test_val (str): The value in `split_obs_key` identifying test samples.
        valid_val (str, optional): The value in `split_obs_key` identifying validation samples.
                                   If None, no validation DataLoader is created.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of workers for DataLoaders.

    Returns:
        tuple: (train_dataloader, valid_dataloader, test_dataloader)
               If valid_val is None, valid_dataloader will be None.
    """
    if split_obs_key not in adata_prepared_obj.adata.obs.columns:
        raise KeyError(f"Split key '{split_obs_key}' not found in adata.obs. Available keys: {list(adata_prepared_obj.adata.obs.columns)}")

    split_column = adata_prepared_obj.adata.obs[split_obs_key]
    full_pytorch_dataset = TorchDatasetFromPrepared(adata_prepared_obj.features, adata_prepared_obj.labels)

    # Get indices
    train_indices = np.where(split_column == train_val)[0]
    test_indices = np.where(split_column == test_val)[0]

    if not train_indices.size:
        print(f"Warning: No training samples found for split_key='{split_obs_key}' with value='{train_val}'. Train DataLoader will be empty.")
    if not test_indices.size:
        print(f"Warning: No test samples found for split_key='{split_obs_key}' with value='{test_val}'. Test DataLoader will be empty.")

    print(f"\n--- Indices for Splits (from '{split_obs_key}') ---")
    print(f"Number of training samples ('{train_val}'): {len(train_indices)}")
    print(f"Number of test samples ('{test_val}'): {len(test_indices)}")

    # Samplers and Subsets
    train_sampler = SubsetRandomSampler(train_indices)
    test_subset = Subset(full_pytorch_dataset, test_indices)

    train_dataloader = DataLoader(full_pytorch_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valid_dataloader = None

    if valid_val:
        valid_indices = np.where(split_column == valid_val)[0]
        if not valid_indices.size:
            print(f"Warning: No validation samples found for split_key='{split_obs_key}' with value='{valid_val}'. Validation DataLoader will be None.")
        else:
            print(f"Number of validation samples ('{valid_val}'): {len(valid_indices)}")
            valid_subset = Subset(full_pytorch_dataset, valid_indices)
            valid_dataloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("-" * 30)
    return train_dataloader, valid_dataloader, test_dataloader


def get_hvg_dataloaders(dat_dir, batch_size, num_workers):
	h5ad_file_path = os.path.join(dat_dir, "kang_normalized_hvg.h5ad")
	label_column = 'cell_type'       # Column in adata.obs for labels
	split_info_column = 'split_B' # Column in adata.obs for train/valid/test splits

	train_set_label = "train"
	validation_set_label = "valid" # Set to None if you don't have/want a validation set
	test_set_label = "ood"    # Or "ood" if that's what you use

	try:
	    adata_processor = AnnDataFeaturesLabels(
	        h5ad_path=h5ad_file_path,
	        label_obs_key=label_column,
            # add_feature = "condition"
	        # feature_layer="counts" # Example: if you want to use adata.layers['counts']
	    )

	    # Get the DataLoaders
	    train_loader, valid_loader, test_loader = get_dataloaders_from_adata_splits(
	        adata_prepared_obj=adata_processor,
	        split_obs_key=split_info_column,
	        train_val=train_set_label,
	        valid_val=validation_set_label,
	        test_val=test_set_label,
	        batch_size=batch_size, # Example batch size
	        num_workers=num_workers
	    )

	    print("\n--- DataLoaders Ready ---")
	    print(f"Train DataLoader: {len(train_loader)} batches")
	    if valid_loader:
	        print(f"Validation DataLoader: {len(valid_loader)} batches")
	    else:
	        print("Validation DataLoader: Not created.")
	    print(f"Test DataLoader: {len(test_loader)} batches")
	    print("-" * 30)

	    # Example: Iterate through the train_loader
	    print("\n--- Iterating through train_loader (first few batches) ---")
	    if train_loader and len(train_loader) > 0:
	        for i, (features, labels) in enumerate(train_loader):
	            print(f"Train Batch {i+1}: Features shape {features.shape}, Labels shape {labels.shape}")
	            if i >= 2: # Show first 3 batches
	                break
	    else:
	        print("Train DataLoader is empty or not created.")

	    # Example: Iterate through the test_loader
	    print("\n--- Iterating through test_loader (first batch if not empty) ---")
	    if test_loader and len(test_loader) > 0:
	        for i, (features, labels) in enumerate(test_loader):
	            print(f"Test Batch {i+1}: Features shape {features.shape}, Labels shape {labels.shape}")
	            break # Show first batch
	    else:
	        print("Test DataLoader is empty or not created.")

	except KeyError as e:
	    print(f"\n--- ERROR during setup ---")
	    print(f"A KeyError occurred: {e}")
	    print("This often means a specified column (for labels or splits) was not found in the AnnData object's .obs attribute.")
	    print("Please check your 'label_column' and 'split_info_column' against the actual columns in your .h5ad file.")
	except Exception as e:
	    print(f"\n--- An unexpected error occurred ---")
	    print(e)

	return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # Define the path to your AnnData file
    # IMPORTANT: Replace this with the actual path to your .h5ad file.
    # The dummy data generation in AnnDataFeaturesLabels will be triggered if this file isn't found.
    h5ad_file_path = "data/kang_normalized_hvg.h5ad" # <--- YOUR FILE PATH HERE

    # Define keys from your AnnData object
    label_column = 'cell_type'       # Column in adata.obs for labels
    split_info_column = 'split_B' # Column in adata.obs for train/valid/test splits
                                     # (Change if your dummy/real data uses a different key like 'split_B')

    # Values in the split_info_column that define your sets
    train_set_label = "train"
    validation_set_label = "valid" # Set to None if you don't have/want a validation set
    test_set_label = "ood"    # Or "ood" if that's what you use

    # Instantiate the data preparation class
    # This will load the h5ad file (or create dummy data if not found)
    # and prepare features and labels.
    try:
        adata_processor = AnnDataFeaturesLabels(
            h5ad_path=h5ad_file_path,
            label_obs_key=label_column
            # feature_layer="counts" # Example: if you want to use adata.layers['counts']
        )

        # Get the DataLoaders
        train_loader, valid_loader, test_loader = get_dataloaders_from_adata_splits(
            adata_prepared_obj=adata_processor,
            split_obs_key=split_info_column,
            train_val=train_set_label,
            valid_val=validation_set_label,
            test_val=test_set_label,
            batch_size=64, # Example batch size
            num_workers=0
        )

        print("\n--- DataLoaders Ready ---")
        print(f"Train DataLoader: {len(train_loader)} batches")
        if valid_loader:
            print(f"Validation DataLoader: {len(valid_loader)} batches")
        else:
            print("Validation DataLoader: Not created.")
        print(f"Test DataLoader: {len(test_loader)} batches")
        print("-" * 30)

        # Example: Iterate through the train_loader
        print("\n--- Iterating through train_loader (first few batches) ---")
        if train_loader and len(train_loader) > 0:
            for i, (features, labels) in enumerate(train_loader):
                print(f"Train Batch {i+1}: Features shape {features.shape}, Labels shape {labels.shape}")
                if i >= 2: # Show first 3 batches
                    break
        else:
            print("Train DataLoader is empty or not created.")

        # Example: Iterate through the test_loader
        print("\n--- Iterating through test_loader (first batch if not empty) ---")
        if test_loader and len(test_loader) > 0:
            for i, (features, labels) in enumerate(test_loader):
                print(f"Test Batch {i+1}: Features shape {features.shape}, Labels shape {labels.shape}")
                break # Show first batch
        else:
            print("Test DataLoader is empty or not created.")

    except KeyError as e:
        print(f"\n--- ERROR during setup ---")
        print(f"A KeyError occurred: {e}")
        print("This often means a specified column (for labels or splits) was not found in the AnnData object's .obs attribute.")
        print("Please check your 'label_column' and 'split_info_column' against the actual columns in your .h5ad file.")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(e)

# if __name__ == "__main__":
# 	dat_path = "../data/kang_normalized_hvg.h5ad"
# 	adata = sc.read(dat_path)

# 	# sc.pp.normalize_total(adata)
# 	# sc.pp.log1p(adata)
# 	# sc.pp.highly_variable_genes(
# 	#     adata,
# 	#     n_top_genes=5000,
# 	#     batch_key="batch",
# 	#     subset=True)
# 	# print(adata)
# 	# print(adata.layers['counts'])
# 	# print(adata.obs['cell_type'])
# 	adata.obs['dose'] = adata.obs['condition'].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))
# 	print(adata.obs["dose"].value_counts())
# 	print(adata.obs["cell_type"].value_counts())
# 	print(adata.obs['condition'].value_counts())
# 	print(adata.obs['split_B'].value_counts())
# 	# Assuming 'adata' is your AnnData object
# 	print(f"adata.X data type: {adata.X.dtype}")

# 	# Print a small sample of non-zero values if possible
# 	if hasattr(adata.X, "data"): # For sparse matrices
# 	    print(f"Sample of non-zero values in adata.X: {adata.X.data[:20]}")
# 	else: # For dense matrices
# 	    non_zero_sample = adata.X[adata.X > 0]
# 	    if len(non_zero_sample) > 0:
# 	        print(f"Sample of non-zero values in adata.X: {non_zero_sample.flatten()[:20]}")
# 	    else:
# 	        print("adata.X contains all zeros (or is empty of non-zeros in sample).")

# 	print(f"Min value in adata.X: {adata.X.min()}")
# 	print(f"Max value in adata.X: {adata.X.max()}")

	# sc.pp.neighbors(adata)
	# sc.tl.umap(adata)
	# sc.pl.umap(adata,
	#            color=['condition', 'cell_type'],
	#            frameon=False,
	#            wspace=0.5)

