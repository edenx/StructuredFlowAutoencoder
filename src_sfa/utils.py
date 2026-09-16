import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure



def first_eigen_proj(x):
    x_centered = x - x.mean(0, keepdims=True)
    cov_matrix = np.cov(x_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    first_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    projected_array = np.dot(x, first_eigenvector)

    return projected_array


def sample_and_reconstruct_pca(
    X_original: torch.Tensor,
    mean_original: torch.Tensor, # Shape (n_features,) or (1, n_features)
    Vh_from_svd: torch.Tensor,   # Full Vh from SVD of centered data
    k_components: int,
    num_new_samples: int = 5,
    sampling_strategy: str = "standard_normal" # or "standard_normal"
):
    """
    Samples points in the PCA reduced space and transforms them back to the original space.

    Args:
        X_original: The original data tensor (n_samples_orig, n_features).
        mean_original: The mean vector of the original data.
        Vh_from_svd: The Vh matrix (V.T) from torch.linalg.svd(X_original - mean_original).
        k_components: The number of principal components defining the reduced space.
        num_new_samples: Number of new points to sample and reconstruct.
        sampling_strategy:
            "mimic_projection": Samples based on mean/std of original data projected to reduced space.
            "standard_normal": Samples from N(0,1) in each reduced dimension.

    Returns:
        Tuple of (new_points_reduced, reconstructed_original_data)
    """
    device = X_original.device
    n_features = X_original.shape[1]

    if mean_original.ndim == 1:
        mean_original_bc = mean_original.unsqueeze(0) # For broadcasting
    else:
        mean_original_bc = mean_original

    # 1. Define principal components and get stats from original data projection
    X_centered = X_original - mean_original_bc
    principal_components = Vh_from_svd[:k_components, :] # Shape: (k_components, n_features)

    scores_original_data = X_centered @ principal_components.T # Shape: (n_samples_orig, k_components)
    scores_mean = scores_original_data.mean(dim=0)
    scores_std = scores_original_data.std(dim=0, unbiased=True)

    # 2. Sample from the reduced dimension space
    if sampling_strategy == "mimic_projection":
        sampled_scores_normalized = torch.randn(num_new_samples, k_components, device=device)
        new_points_reduced = sampled_scores_normalized * scores_std + scores_mean
    elif sampling_strategy == "standard_normal":
        new_points_reduced = torch.randn(num_new_samples, k_components, device=device)
    else:
        raise ValueError("Unknown sampling_strategy")

    # 3. Transform back to original high-dimensional space
    reconstructed_centered_data = new_points_reduced @ principal_components
    reconstructed_original_data = reconstructed_centered_data + mean_original_bc

    return new_points_reduced, reconstructed_original_data



class TSNEWithTransform:
    def __init__(self, n_components=2, perplexity=30, random_state=None, neighbors=20):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.embedding_ = None
        self.X_fit = None
        self.neighbors = neighbors
        
    def fit(self, X):
        """Fit TSNE to data X"""
        self.X_fit = X
        tsne = TSNE(n_components=self.n_components, 
                    perplexity=self.perplexity,
                    random_state=self.random_state)
        self.embedding_ = tsne.fit_transform(X)
        return self
        
    def transform(self, X):
        """Transform new data using nearest neighbors in the original space"""
        if self.embedding_ is None:
            raise ValueError("Model not fitted yet.")
            
        # Find k nearest neighbors in the original space
        k = min(self.neighbors, self.X_fit.shape[0])
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.X_fit)
        
        # Get distances and indices of nearest neighbors
        distances, indices = nn.kneighbors(X)
        
        # Normalize distances to weights
        weights = 1 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Weighted average of the embeddings of nearest neighbors
        result = np.zeros((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            for j in range(k):
                result[i] += weights[i, j] * self.embedding_[indices[i, j]]
                
        return result

def tsne_map(references, trajectories, k, random_state=14, neighbors=10):
    """
    Fit t-SNE on the first time point and transform all other time points.
    
    Args:
        trajectories: numpy array of shape (nbatch, total_t, feature_dim)
    """
    n_samples, n_timesteps, feature_dim = trajectories.shape
    trajectories_2d = np.zeros((n_samples, n_timesteps, k))
    
    # Initialize the wrapper
    tsne_wrapper = TSNEWithTransform(n_components=k, random_state=random_state, neighbors=neighbors)
    
    # Fit on first time point
    tsne_wrapper.fit(references)
    # trajectories_2d[:, 0, :] = tsne_wrapper.embedding_
    
    # Transform remaining time points
    for t in range(0, n_timesteps):
        trajectories_2d[:, t, :] = tsne_wrapper.transform(trajectories[:, t, :])
    
    return trajectories_2d


from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np

class PCAWithTransform:
    def __init__(self, n_components=2, random_state=None, neighbors=20):
        self.n_components = n_components
        self.random_state = random_state
        self.embedding_ = None
        self.X_fit = None
        self.neighbors = neighbors
        self.pca = None
        
    def fit(self, X):
        """Fit PCA to data X"""
        self.X_fit = X
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.embedding_ = self.pca.fit_transform(X)
        return self
        
    def transform(self, X):
        """Transform new data directly using PCA projection"""
        if self.pca is None:
            raise ValueError("Model not fitted yet.")
            
        # Direct transformation using PCA
        direct_projection = self.pca.transform(X)
        return direct_projection
    
    def transform_with_neighbors(self, X):
        """Transform using nearest neighbors approach (alternative method)"""
        if self.embedding_ is None:
            raise ValueError("Model not fitted yet.")
            
        # Find k nearest neighbors in the original space
        k = min(self.neighbors, self.X_fit.shape[0])
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.X_fit)
        
        # Get distances and indices of nearest neighbors
        distances, indices = nn.kneighbors(X)
        
        # Normalize distances to weights
        weights = 1 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Weighted average of the embeddings of nearest neighbors
        result = np.zeros((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            for j in range(k):
                result[i] += weights[i, j] * self.embedding_[indices[i, j]]
                
        return result


def pca_map(references, trajectories, k, random_state=42, use_neighbors=False):
    """
    Fit PCA on reference data and transform all trajectory time points.
    
    Args:
        references: numpy array of reference points to fit PCA on
        trajectories: numpy array of shape (nbatch, total_t, feature_dim)
        k: number of components to use in PCA
        random_state: random state for reproducibility
        use_neighbors: whether to use nearest neighbor approach for transformation
    
    Returns:
        Transformed trajectories with shape (nbatch, total_t, k)
    """
    n_samples, n_timesteps, feature_dim = trajectories.shape
    trajectories_pca = np.zeros((n_samples, n_timesteps, k))
    
    # Initialize the wrapper
    pca_wrapper = PCAWithTransform(n_components=k, random_state=random_state)
    
    # Fit on reference data
    pca_wrapper.fit(references)
    
    # Transform each time point
    for t in range(n_timesteps):
        if use_neighbors:
            trajectories_pca[:, t, :] = pca_wrapper.transform_with_neighbors(trajectories[:, t, :])
        else:
            trajectories_pca[:, t, :] = pca_wrapper.transform(trajectories[:, t, :])
    
    return trajectories_pca
    

def soft_nmi(probs, labels, epsilon=1e-10):
    """
    Compute Soft Normalized Mutual Information (NMI) for soft clustering.

    Parameters:
    - probs: (N, K) Soft cluster assignments (each row sums to 1).
    - labels: (N,) Ground truth hard labels (integer values).

    Returns:
    - Soft NMI score (0 to 1).
    """
    N, K = probs.shape  # Number of samples, number of clusters

    # Convert labels to one-hot encoding
    one_hot_labels = np.eye(K)[labels]

    # Compute marginal entropies (average over batch)
    H_y = np.mean(entropy(one_hot_labels + epsilon, axis=0))  # True label entropy
    H_p = np.mean(entropy(probs + epsilon, axis=0))  # Soft clustering entropy

    # Compute soft joint probability matrix P_ij
    P_joint = (one_hot_labels.T @ probs) / N  # (K, K) matrix

    # Normalize joint probability
    P_joint = P_joint / np.sum(P_joint)  # Ensure it's a valid probability distribution

    # Compute joint entropy correctly
    H_joint = -np.sum(P_joint * np.log(P_joint + epsilon))

    # Debugging: Check entropy values
    print(f"H_y: {H_y}, H_p: {H_p}, H_joint: {H_joint}")

    # Ensure proper normalization
    denominator = H_y + H_p # max(H_y, H_p)
    if denominator == 0:  # Avoid division by zero
        return 0

    # Compute Soft NMI with proper normalization
    soft_nmi_score = max(0, min((H_y + H_p - H_joint) / denominator, 1))  # Clamp to [0,1]

    return soft_nmi_score


def soft_ari(probs, labels):
    """Compute Soft ARI by converting soft assignments to pairwise probabilities."""
    one_hot_labels = np.eye(probs.shape[1])[labels]  # Convert labels to one-hot encoding
    
    # Compute pairwise distances in soft assignment space
    pred_dists = squareform(pdist(probs, metric='cosine'))
    true_dists = squareform(pdist(one_hot_labels, metric='cosine'))

    # Convert distances to pseudo-labels (closest cluster)
    pred_labels = np.argmin(pred_dists, axis=1)
    true_labels = np.argmin(true_dists, axis=1)

    return adjusted_rand_score(true_labels, pred_labels)


def GLscore(genx, trainx):
    ssim = StructuralSimilarityIndexMeasure()
    ssim_score = []

    genx = genx.repeat(1, 3, 1, 1).to(torch.float32)
    trainx = trainx.repeat(1, 3, 1, 1).to(torch.float32)
    for i in range(len(trainx)):
        ssim_score.append(ssim(genx, trainx[i]))

    pass




def plot_image_sequence_and_trajectory(image_sequence, latent_trajectory, figsize=(12, 5)):
    """
    Plots a sequence of images in the first row and a latent trajectory
    in the second row, aligned by step.

    Args:
        image_sequence (list or np.ndarray or torch.Tensor):
            A list/array containing the sequence of images.
            Each image should be suitable for plt.imshow (e.g., HxW, HxWxC).
            If PyTorch tensors, expects shape like (N, C, H, W) or (N, H, W).
        latent_trajectory (np.ndarray or torch.Tensor):
            The latent trajectory data. Expected shape (n_steps, latent_dim).
        img_title (str, optional):
            Title for the image sequence row. Defaults to "Image Sequence".
        traj_title (str, optional):
            Y-axis label for the trajectory plot. Defaults to "Latent Trajectory (z)".
        figsize (tuple, optional):
            The figure size for the plot. Defaults to (12, 5).
    """
    # --- Input Validation and Setup ---
    if isinstance(image_sequence, torch.Tensor):
        # If sequence is a single tensor (N, C, H, W) or (N, H, W), convert to list
        if image_sequence.ndim >= 3:
             image_sequence = [img for img in image_sequence] # Iterate over the 0th dimension

    n_steps = len(image_sequence)
    if n_steps == 0:
        print("Input image sequence is empty, nothing to plot.")
        return

    # Process latent trajectory
    z = latent_trajectory
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()

    if not isinstance(z, np.ndarray) or z.ndim != 2:
         raise ValueError("latent_trajectory must be a 2D numpy array or tensor (n_steps, latent_dim).")

    if z.shape[0] != n_steps:
        raise ValueError(f"Number of steps in image_sequence ({n_steps}) must match "
                         f"latent_trajectory ({z.shape[0]}).")

    latent_dim = z.shape[1]

    # --- Plotting Setup ---
    fig = plt.figure(figsize=figsize)
    # Create a grid: 2 rows, n_steps columns. Give more height to images.
    gs = gridspec.GridSpec(2, n_steps, height_ratios=[1, 1], hspace=0.3)

    # --- Plot Image Sequence (First Row) ---
    for i in range(n_steps):
        ax_img = fig.add_subplot(gs[0, i])

        # Process image
        img = image_sequence[i]
        # Handle PyTorch Tensors within the list
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        # Handle channel dimension (e.g., C, H, W -> H, W, C or H, W if C=1)
        if img.ndim == 3 and img.shape[0] in [1, 3]: # Check if first dim is channel
             img = np.squeeze(img) # Remove channel dim if 1
             if img.ndim == 3: # If still 3D (RGB), move channel to last axis
                 img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 3 and img.shape[-1] not in [1, 3]: # Check if last dim is not channel
             # Handle cases like (H, W, C) where C is not 1 or 3, assume grayscale
             if img.shape[-1] > 3:
                 img = img[..., 0] # Take the first channel if unsure

        # Display image
        ax_img.imshow(img, cmap='gray' if img.ndim == 2 else None, aspect='equal')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        if i == 0:
            ax_img.set_ylabel(r"$x$") # Set row title on the first image's y-axis

    # --- Plot Latent Trajectory (Second Row) ---
    ax_traj = fig.add_subplot(gs[1, :]) # Span all columns in the second row

    steps_axis = np.arange(n_steps)
    # Plot each dimension of the latent trajectory
    for d in range(latent_dim):
        ax_traj.plot(steps_axis, z[:, d], label=f"z[{d}]", marker='.', linestyle='-')

    ax_traj.set_xlabel("Step")
    ax_traj.set_ylabel(r"$z$")
    # Ensure x-axis ticks match the number of steps visually
    if n_steps <= 15: # Show all step ticks if not too many
        ax_traj.set_xticks(steps_axis)
    ax_traj.set_xlim(steps_axis.min() - 0.5, steps_axis.max() + 0.5) # Align x-axis limits
    ax_traj.grid(True, axis='x', linestyle=':') # Add vertical grid lines for alignment


    # Overall figure title (optional)
    # fig.suptitle("Image Sequence and Latent Trajectory", fontsize=14)

    # Adjust layout - may need manual tweaking depending on titles/legends
    # plt.tight_layout(rect=[0, 0.03, 0.95 if latent_dim > 1 else 1, 0.95]) # Leave space for legend/title
    fig.tight_layout()
    # plt.show()


def dict2namespace(config):
    """Recursively convert a dict to an argparse.Namespace."""
    import argparse
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def remap_checkpoint_state_dict(state_dict):
    """Remap checkpoint state dict keys for compatibility with updated model structure."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "Rt.fc.0." in key:
            parts = key.split(".")
            new_parts = parts[:2] + parts[3:]
            new_key = ".".join(new_parts)
            new_state_dict[new_key] = value
        elif "flow_matching_loss.Rt.fc.0." in key:
            parts = key.split(".")
            new_parts = parts[:3] + parts[4:]
            new_key = ".".join(new_parts)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
