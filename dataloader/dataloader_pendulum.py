import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

from dataloader.Pendulum import *
from dataloader.dataloader_mnist import LogitTrans, SigmoidTrans

class normalize_for_tanh():
    def __init__(self):
        pass

    def __call__(self, image):
        return 2 * image - 1

transform = transforms.Compose([
    # transforms.Resize((14, 14)),  # Pad 2 pixels on each side (left, top, right, bottom)
    # transforms.ToTensor(),         # Convert image to PyTorch tensor
    # transforms.Lambda(lambda x: x.double())
    # transforms.Normalize((0.5,), (0.5,)) # normalize to between (-1,1)
    LogitTrans(),
    # normalize_for_tanh(),
])

inv_transform = transforms.Compose([
    # transforms.Resize((14, 14)),  # Pad 2 pixels on each side (left, top, right, bottom)
    # transforms.Normalize((0.5,), (0.5,)) # normalize to between (-1,1)
    SigmoidTrans(),
])


class PENDULUM(Dataset):
    def __init__(self, N, p, S, dat_dir, window_size=9, stride=1, use_sliding_window=True, 
                 samples_per_batch=64, batch_from_same_trajectory=False,
                 state_dim=2, gen=False, plot=False, seed=111, transform=None):
        self.N = N
        self.n = state_dim  # latent dim
        self.p = p  # ambient dim
        self.S = S
        self.dat_dir = dat_dir
        self.seed = seed
        self.transform = transform
        
        # Sliding window parameters
        self.window_size = window_size
        self.stride = stride
        self.use_sliding_window = use_sliding_window
        self.samples_per_batch = samples_per_batch
        self.batch_from_same_trajectory = batch_from_same_trajectory

        # Generate new data if required
        file_path = os.path.join(self.dat_dir, 'pendulum.npz')

        if gen:
            all_data, all_states = self.sample()
            np.savez_compressed(file_path, states=all_states, data=all_data)

        # Load data
        loaded = np.load(file_path)
        self.states = torch.tensor(loaded['states'], dtype=torch.float32)
        self.data = (torch.tensor(loaded['data'], dtype=torch.float32))/255.0
        
        # Calculate windows per trajectory
        if self.use_sliding_window:
            self.windows_per_trajectory = max(0, self.S - self.window_size + 1) // self.stride
            
            if self.batch_from_same_trajectory:
                # Each trajectory will produce N// (samples_per_batch) batches
                # We need longer trajectories for this approach
                required_length = self.window_size + (self.samples_per_batch - 1) * self.stride
                if self.S < required_length:
                    raise ValueError(
                        f"Trajectory length {self.S} is too short. For {self.samples_per_batch} samples "
                        f"with window size {self.window_size} and stride {self.stride}, "
                        f"you need at least {required_length} frames per trajectory."
                    )
                
                # Calculate how many complete batches we can get from each trajectory
                self.batches_per_trajectory = (self.windows_per_trajectory // self.samples_per_batch)
                self.total_batches = self.N * self.batches_per_trajectory
                self.total_windows = self.total_batches * self.samples_per_batch
            else:
                # Original approach: all windows from all trajectories
                self.total_windows = self.N * self.windows_per_trajectory
        
        if plot:
            self._plot_samples(loaded['data'], loaded['states'])

    def sample(self, gen_se=False):
        pend_params = Pendulum.pendulum_default_params()
        pend_params[Pendulum.FRICTION_KEY] = .001
        pend_params[Pendulum.LENGTH_KEY] = .12 # .09
        pendulum = Pendulum(self.p, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                        transition_noise_std=0., observation_noise_std=1e-5,
                        seed=42, pendulum_params=pend_params,
                        state_dim=self.n)

        obs, targets, _, _ = pendulum.sample_data_set(self.N, self.S, full_targets=False)
        if gen_se:  # add random noise to the observed
            obs, _ = pendulum.add_observation_noise(obs, first_n_clean=5, r=0.2, 
                                                    t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
        obs = np.expand_dims(obs, -3)

        return obs, targets

    def __len__(self):
        if self.use_sliding_window:
            if self.batch_from_same_trajectory:
                # Return number of complete batches we can create
                return self.total_batches
            else:
                # Original approach: return total windows
                return self.total_windows
        else:
            return self.N

    def __getitem__(self, idx):
        if self.use_sliding_window:
            if self.batch_from_same_trajectory:
                # Here, idx represents a batch index
                # Each batch consists of samples_per_batch consecutive windows from the same trajectory
                trajectory_idx = idx // self.batches_per_trajectory
                batch_within_trajectory = idx % self.batches_per_trajectory
                
                # Starting position for this batch within the trajectory
                start_pos = batch_within_trajectory * self.samples_per_batch * self.stride
                
                # Create a batch of windows from consecutive positions
                batch_data = []
                batch_states = []
                batch_indices = []
                
                for i in range(self.samples_per_batch):
                    window_start = start_pos + (i * self.stride)
                    window_end = window_start + self.window_size
                    
                    # Extract window
                    data_window = self.data[trajectory_idx, window_start:window_end]
                    state_window = self.states[trajectory_idx, window_start:window_end]
                    
                    if self.transform:
                        data_window = self.transform(data_window)
                    
                    batch_data.append(data_window)
                    batch_states.append(state_window)
                    batch_indices.append(torch.arange(window_start, window_end))
                # Stack all windows into a batch
                batch_data = torch.stack(batch_data)
                batch_states = torch.stack(batch_states)
                batch_indices = torch.stack(batch_indices)
                
                return batch_data, batch_states, batch_indices/self.S
            else:
                # Original approach: single window indexing
                # Convert flat index to trajectory and window indices
                trajectory_idx = idx // self.windows_per_trajectory
                window_idx = (idx % self.windows_per_trajectory) * self.stride
                
                # Extract the window
                end_idx = window_idx + self.window_size
                data_window = self.data[trajectory_idx, window_idx:end_idx]
                state_window = self.states[trajectory_idx, window_idx:end_idx]
                
                original_indices = torch.arange(window_idx, end_idx)

                if self.transform:
                    data_window = self.transform(data_window)
                    
                # return data_window, state_window
                return data_window, state_window, original_indices/self.S
        else:
            # Original behavior: return entire trajectory
            data, state = self.data[idx], self.states[idx]
            if self.transform:
                data = self.transform(data)
            return data, state, torch.arange(self.S)/self.S
    
    def _plot_samples(self, all_data, all_states):
        # Plotting code from original implementation
        # (Simplified for brevity)
        stacked = all_data[12][:49].reshape(7, self.p*7, self.p)
        imgrid = stacked.swapaxes(0, 1).reshape(self.p * 7, self.p * 7)

        fig, axes = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [3, 1]})
        axes[0].imshow(imgrid, cmap="gray", origin="lower", aspect=.2)
        axes[0].set_xlabel("data")
        axes[0].set_xmargin(0)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].plot(np.arange(49), all_states[12][:49])
        axes[1].set_xmargin(0)
        axes[1].set_xlabel("latent")
        plt.tight_layout()
        plt.savefig("../gen_figs/pendulum{}_1.png".format(self.S))
        plt.close()


def get_pendulum_dataloader(Ntrain, Ntest, p, S, dat_dir, window_size=9, stride=1, batch_size=64,
                           samples_per_batch=64, batch_from_same_trajectory=False,
                           state_dim=2, gen=False, plot=False, shuffle=True, num_workers=4,
                           logtran=True):
    """
    Creates and returns dataloaders for training and testing with the sliding window approach.
    
    Parameters:
    -----------
    N : int
        Number of trajectories
    p : int
        Ambient dimension
    S : int
        Sequence length for each trajectory
    dat_dir : str
        Directory to save/load data
    window_size : int
        Size of sliding window
    stride : int
        Step size between consecutive windows
    batch_size : int
        Batch size for the dataloader (only used when batch_from_same_trajectory=False)
    samples_per_batch : int
        Number of samples per batch when using batch_from_same_trajectory=True
    batch_from_same_trajectory : bool
        If True, each batch will contain samples only from the same trajectory
    state_dim : int
        Dimension of the latent state
    gen : bool
        Whether to generate new data
    plot : bool
        Whether to plot samples
    shuffle : bool
        Whether to shuffle the data
    num_workers : int
        Number of worker threads for the dataloader
        
    Returns:
    --------
    tuple
        (train_dataloader, test_dataloader)
    """
    # transform = transforms.Compose([LogitTrans()])
    if logtran:
        tran = transform
    else:
        tran = None
    
    # Create training dataset with sliding window
    train_dataset = PENDULUM(
        N=Ntrain, p=p, S=S, dat_dir=dat_dir, 
        window_size=window_size, stride=stride, use_sliding_window=True,
        samples_per_batch=samples_per_batch, batch_from_same_trajectory=batch_from_same_trajectory,
        state_dim=state_dim, gen=gen, plot=plot, transform=tran
    )
    
    # Create test dataset (optionally with sliding window)
    test_dataset = PENDULUM(
        N=Ntest, p=p, S=S, dat_dir=dat_dir + "_test", 
        window_size=window_size, stride=stride, use_sliding_window=True,
        samples_per_batch=samples_per_batch, batch_from_same_trajectory=batch_from_same_trajectory,
        state_dim=state_dim, gen=gen, transform=tran
    )
    
    # If batch_from_same_trajectory is True, each __getitem__ call returns a full batch
    # so we set the DataLoader batch_size to 1
    effective_batch_size = 1 if batch_from_same_trajectory else batch_size
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=shuffle, 
        num_workers=num_workers, pin_memory=False,
        # persistent_worker=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=effective_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    N = 20          # Number of trajectories
    dat_dir = "../data"
    n = 2           # State dimension
    p = 10          # Ambient dimension
    S = 100         # Length of each trajectory
    
    window_size = 9  # Size of sliding window
    stride = 1       # Step size between consecutive windows
    samples_per_batch = 64  # Samples per batch from same trajectory
    
    # Required trajectory length calculation
    required_length = window_size + (samples_per_batch - 1) * stride
    print(f"Required trajectory length for {samples_per_batch} samples with window size {window_size} and stride {stride}: {required_length}")
    
    # Check if current trajectory length is sufficient
    if S < required_length:
        print(f"WARNING: Current trajectory length {S} is too short. Adjusting to minimum required: {required_length}")
        S = required_length
    
    # Generate data and create dataloader with sliding window approach
    # Option 1: Traditional approach (samples can come from different trajectories)
    train_loader_traditional, test_loader_traditional = get_pendulum_dataloader(
        N=N, p=p, S=S, dat_dir=dat_dir,
        window_size=window_size, stride=stride, batch_size=32,
        batch_from_same_trajectory=False,
        gen=True, plot=True
    )
    
    # # Option 2: Batch from same trajectory approach
    # train_loader_same_traj, test_loader_same_traj = get_pendulum_dataloader(
    #     N=N, p=p, S=S, dat_dir=dat_dir,
    #     window_size=window_size, stride=stride, 
    #     samples_per_batch=samples_per_batch, batch_from_same_trajectory=True,
    #     gen=False, plot=False
    # )
    
    # Calculate total samples in the dataset for traditional approach
    windows_per_trajectory = (S - window_size) // stride + 1
    print("\nTraditional Approach (samples can come from different trajectories):")
    print(f"Total trajectories: {N}")
    print(f"Windows per trajectory: {windows_per_trajectory}")
    print(f"Total sliding windows: {N * windows_per_trajectory}")
    
    # Visualize a batch from the traditional sliding window dataloader
    for batch_data, batch_states in train_loader_traditional:
        print(f"Traditional batch shape - Data: {batch_data.shape}, States: {batch_states.shape}")
        break
    
    # Calculate total samples in the dataset for same-trajectory approach
    batches_per_trajectory = windows_per_trajectory // samples_per_batch
    print("\nSame-Trajectory Approach (all samples in a batch come from the same trajectory):")
    print(f"Total trajectories: {N}")
    print(f"Batches per trajectory: {batches_per_trajectory}")
    print(f"Total batches: {N * batches_per_trajectory}")
    
    # Visualize a batch from the same-trajectory sliding window dataloader
    for batch_data, batch_states in train_loader_same_traj:
        # Since batch_size=1 in the dataloader, we need to squeeze the batch dimension
        batch_data = batch_data.squeeze(0)
        batch_states = batch_states.squeeze(0)
        print(f"Same-trajectory batch shape - Data: {batch_data.shape}, States: {batch_states.shape}")
        
        # Check that all samples are from the same trajectory (they should be consecutive)
        if batch_states.size(0) >= 2:
            state_diffs = torch.abs(batch_states[1:, 0, :] - batch_states[:-1, 0, :]).mean()
            print(f"Average state difference between consecutive samples: {state_diffs.item():.6f}")
            print(f"(Small value confirms samples are from the same trajectory)")
        break

# class PENDULUM(Dataset):
# 	def __init__(self, N, p, S, dat_dir, state_dim=2, gen=False, plot=False, seed=111, transform=None):
# 		self.N = N
# 		self.n = state_dim  # latent dim
# 		self.p = p  # ambient dim
# 		self.S = S
# 		self.dat_dir = dat_dir
# 		self.seed = seed
# 		self.transform = transform

# 		# Generate new data if required
# 		file_path = os.path.join(self.dat_dir, 'pendulum.npz')


# 		if gen:
# 			all_data, all_states = self.sample()
# 			# test_data, test_states = self.sample(train=False)
# 			np.savez_compressed(file_path, states=all_states, data=all_data)
# 			# np.savez_compressed(test_file_path, states=test_states, data=test_data)

# 		# Load data
# 		# subsample, add observation noise and normalize
# 		loaded = np.load(file_path)
# 		self.states = torch.tensor(loaded['states'], dtype=torch.float32)
		
# 		self.data = (torch.tensor(loaded['data'], dtype=torch.float32))/255.0
		
# 		if plot:
# 			# plot first 100 frames
# 			stacked = all_data[12][:49].reshape(7, self.p*7, self.p)
# 			imgrid = stacked.swapaxes(0, 1).reshape(self.p * 7, self.p * 7)

# 			fig, axes = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [3, 1]})
# 			axes[0].imshow(imgrid, cmap="gray", origin="lower", aspect=.2)
# 			axes[0].set_xlabel("data")
# 			axes[0].set_xmargin(0)
# 			axes[0].set_xticks([])
# 			axes[0].set_yticks([])
# 			# axes[0].plot(np.arange(len(all_data[2])), all_data[2])
# 			axes[0].set_xmargin(0)
# 			axes[1].plot(np.arange(49), all_states[12][:49])
# 			axes[1].set_xmargin(0)
# 			axes[1].set_xlabel("latent")
# 			plt.tight_layout()
# 			plt.savefig("../gen_figs/pendulum{}_1.png".format(self.S))
# 			plt.close()


# 			stacked_image = np.hstack(all_data[0][:50, 0, :, :])  # Shape: (24, 240)
# 			stacked_states = all_states[0][:50,:]
# 			# Display the stacked image
# 			# plt.figure(figsize=(10, 4))  # Adjust figure size for better visualization
# 			# plt.imshow(stacked_image, cmap='gray')
# 			# plt.axis('off')  # Turn off axis for better visualization
# 			# plt.savefig("gen_figs/pendulum_img{}.png".format(10))
# 			# plt.close()

# 			# fig, axes = plt.subplots(2, 1, figsize=(8, 4))
# 			fig, axes = plt.subplots(2, 1, figsize=(10, 4), gridspec_kw={'height_ratios': [3, 1]})
# 			axes[0].imshow(stacked_image, cmap="gray", origin="lower", aspect=5.)
# 			axes[0].set_xlabel("data")
# 			# axes[0].plot(np.arange(len(all_data[11])), all_data[11])
# 			axes[0].set_xmargin(0)
# 			axes[1].plot(np.arange(len(stacked_states)), stacked_states)
# 			axes[1].set_xmargin(0)
# 			axes[1].set_xlabel("latent")
# 			plt.tight_layout()
# 			plt.savefig("../gen_figs/pendulum_img{}.png".format(10))
# 			plt.close()



# 	def sample(self, gen_se=False):
# 		pend_params = Pendulum.pendulum_default_params()
# 		pend_params[Pendulum.FRICTION_KEY] = .001 # 0.1
# 		pend_params[Pendulum.LENGTH_KEY] = .09
# 		pendulum = Pendulum(self.p, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
# 		                transition_noise_std=0.1, observation_noise_std=1e-5,
# 		                seed=42, pendulum_params=pend_params,
# 		                state_dim=self.n)

# 		obs, targets, _, _ = pendulum.sample_data_set(self.N, self.S, full_targets=False)
# 		if gen_se: # add random noise to the observed
# 			obs, _ = pendulum.add_observation_noise(obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
# 		obs = np.expand_dims(obs, -3)

# 		return obs, targets


# 	def __len__(self):
# 		return self.N

# 	def __getitem__(self, idx):
# 		data, state = self.data[idx], self.states[idx]
# 		if self.transform:
# 			data = self.transform(data)

# 		return data, state


# def plot_img_grid(recon):
#     # plt.figure(figsize=(8,8))
#     # Show the sequence as a block of images
#     stacked = recon.reshape(10, 24 * 10, 24)
#     imgrid = stacked.swapaxes(0, 1).reshape(24 * 10, 24 * 10)
#     # plt.imshow(imgrid, vmin=0, vmax=1)
#     return imgrid



# def get_pendulum(N, p, S, dat_dir, state_dim=2, gen=True, plot=False):
# 	dataset = PENDULUM(N, p, S, dat_dir, state_dim=state_dim, gen=gen, plot=plot, transform=transform)
# 	test_dataset = PENDULUM(N, p, S, dat_dir + "_test", state_dim=state_dim, gen=gen, transform=transform)
# 	return dataset, test_dataset


# if __name__ == "__main__":
#     N = 20
#     dat_dir = "../data"
#     n = 2
#     p = 10
#     S = 100

#     batch_size = 20
#     dataloader = DataLoader(
#         PENDULUM(N, p, S, dat_dir, seed=123, 
#             gen=True, plot=True), batch_size=batch_size, shuffle=True)




