import os

import numpy as np
import numpy.random as npr
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt



def rand_rotation(seed, n, theta=None):
    rng = np.random.default_rng(seed)  # Initialize NumPy random generator

    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * rng.uniform()

    if n == 1:
        return rng.uniform() * np.eye(1)

    # Define the 2D rotation matrix
    rot = np.array([[np.cos(theta), -np.sin(theta)], 
                    [np.sin(theta), np.cos(theta)]])
    
    # Create an identity matrix and embed the 2D rotation in the top-left corner
    out = np.eye(n)
    out[:2, :2] = rot

    # Generate a random matrix and compute its QR decomposition
    random_matrix = rng.uniform(size=(n, n))
    q, _ = np.linalg.qr(random_matrix)

    # Perform the similarity transformation
    return q @ out @ q.T


def rand_psd(dim):
    """
    Generate a random positive semi-definite (PSD) matrix.

    Parameters:
    - dim: The dimension of the matrix.

    Returns:
    - A positive semi-definite matrix of shape (dim, dim).
    """
    A = np.random.normal(size=(dim, dim))
    psd_matrix = np.dot(A, A.T)  # Ensure symmetric and PSD
    return psd_matrix

def rand_stable(n):
    """
    Generate a stable matrix A for LDS.

    Parameters:
    - n: Dimension of the matrix.

    Returns:
    - A stable matrix of shape (n, n).
    """
    A = np.random.normal(size=(n, n))
    eigvals = np.linalg.eigvals(A)
    max_abs_eig = np.max(np.abs(eigvals))
    A /= max_abs_eig + 0.1  # Scale to ensure stability
    assert np.all(np.abs(np.linalg.eigvals(A)) < 1.)
    return A

def rand_prj(p, n):
    # Randomized approach: Entries from standard normal distribution
    R = np.random.randn(p, n)
    # Optionally normalize columns
    R = R / np.linalg.norm(R, axis=0)
    # Perform QR decomposition to ensure orthogonality in columns
    Q, _ = np.linalg.qr(R)
    # Take the first p rows of Q.T to form the projection matrix (p x n)
    A = Q.T[:p, :]
    
    return A.T


def rand_lds(n, p, S=None, seed=42):
    """
    Generate random parameters for a Linear Dynamical System (LDS).

    Parameters:
    - n: Dimension of the latent states.
    - p: Dimension of the observations.
    - S: Optional, number of time steps (for time-varying models).
    - seed: Random seed for reproducibility.

    Returns:
    - LDS parameters: mu_init, sigma_init, A, sigma_states, C, sigma_obs
    """
    np.random.seed(seed)
    homog = S is None

    mu_init = np.random.normal(size=n)
    sigma_init = rand_psd(n)

    A = rand_stable(n)
    B = np.random.normal(size=(n, n))

    if homog:
        C = np.random.normal(size=(p, n))
        D = np.random.normal(size=(p, p))
    else:
        C = np.random.normal(size=(S, p, n))
        D = np.random.normal(size=(S, p, p))

    sigma_states = np.dot(B, B.T)
    sigma_obs = np.dot(D, D.T) if homog else np.einsum('tij,tkj->tik', D, D)

    return mu_init, sigma_init, A, sigma_states, C, sigma_obs

def sin_lds(n, p, theta=0.1, S=None, seed=42):
    """
    Generate a sinusoidal LDS.

    Parameters:
    - n: Dimension of the latent states.
    - p: Dimension of the observations.
    - theta: Rotation angle.
    - S: Optional, number of time steps (for time-varying models).
    - seed: Random seed for reproducibility.

    Returns:
    - LDS parameters: mu_init, sigma_init, A, sigma_states, C, sigma_obs
    """
    np.random.seed(seed)
    homog = S is None

    mu_init = np.random.normal(size=n)
    # mu_init = np.zeros(n)
    sigma_init = rand_psd(n)
    # sigma_init = np.random.normal(size=n) * np.diag(np.ones(n))

    # A = np.array([[np.cos(theta), -np.sin(theta)], 
    #               [np.sin(theta),  np.cos(theta)]])

    A = rand_rotation(seed, n=n, theta=theta)
    B = .01 * np.diag(np.ones(n))

    if homog:
        C = np.random.normal(size=(p, n))
        # C = rand_prj(p, n)
        D = np.random.normal(size=(p, p)) * 0.001
    else:
        C = np.random.normal(size=(S, p, n))
        # C = rand_prj(p, n)
        D = np.random.normal(size=(S, p, p)) * 0.001

    sigma_states = np.dot(B, B.T)
    sigma_obs = np.dot(D, D.T) if homog else np.einsum('tij,tkj->tik', D, D)

    return mu_init, sigma_init, A, sigma_states, C, sigma_obs



class LDS(Dataset):
    def __init__(self, N, n, p, S, dat_dir, sin=True, gen=False, theta=0.03, plot=False, seed=111):
        # if sin:
        #     assert(n == 2)
        self.n = n  # latent dim
        self.p = p  # ambient dim
        self.length = N
        self.S = S
        self.dat_dir = dat_dir
        self.seed = seed

        # Initialize
        if sin:
            self.mu_init, self.sigma_init, self.A, self.sigma_states, self.C, self.sigma_obs = sin_lds(n, p, theta)
        else:
            self.mu_init, self.sigma_init, self.A, self.sigma_states, self.C, self.sigma_obs = rand_lds(n, p)

        # Generate new data if required
        file_path = os.path.join(self.dat_dir, 'lds.npz')
        if gen:
            all_states, all_data = self.sample()
            np.savez_compressed(file_path, states=all_states, data=all_data)

            if plot:
                fig, axes = plt.subplots(2, 1, figsize=(8, 4))
                axes[0].imshow(all_data[0].T, cmap="gray", origin="lower", aspect=.3)
                axes[0].set_xlabel("data")
                # axes[0].plot(np.arange(len(all_data[2])), all_data[2])
                axes[0].set_xmargin(0)
                # axes[1].plot(np.arange(len(all_states[22])), all_states[22])
                axes[1].plot(all_states[0][:,0], all_states[0][:,1])
                axes[1].set_xmargin(0)
                axes[1].set_xlabel("latent")
                plt.tight_layout()
                plt.savefig("gen_figs/lds{}_1.png".format(self.S))
                plt.close()

                # fig, axes = plt.subplots(2, 1, figsize=(8, 4))
                # # axes[0].plot(np.arange(len(all_data[11])), all_data[11])
                # axes[0].imshow(all_data[1].T, cmap="gray", origin="lower", aspect=.3)
                # axes[0].set_xmargin(0)
                # axes[1].plot(all_states[1][:,0], all_states[1][:,1])
                # axes[1].set_xlabel("latent")
                # axes[1].set_xmargin(0)
                # plt.tight_layout()
                # plt.savefig("gen_figs/lds{}_2.png".format(self.S))
                # plt.close()

        # Load data
        loaded = np.load(file_path)
        self.states = torch.tensor(loaded['states'], dtype=torch.float32)
        self.data = torch.tensor(loaded['data'], dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [self.data[idx], self.states[idx]]

    def sample(self):
        np.random.seed(self.seed)
        all_states = []
        all_data = []

        for _ in range(self.length):
            states, data = self._sample(self.S, self.mu_init, self.sigma_init, self.A, 
                                        self.sigma_states, self.C, self.sigma_obs)
            all_states.append(states)
            all_data.append(data)

        return np.array(all_states), np.array(all_data)

    @staticmethod
    def _sample(S, mu_init, sigma_init, A, sigma_states, C, sigma_obs):
        """
        Sample states and observations using numpy.

        Parameters:
        - S: Number of timesteps.
        - mu_init: Initial mean for the latent states.
        - sigma_init: Covariance matrix for the initial state.
        - sigma_states: Covariance matrix for the state transitions.
        - sigma_obs: Covariance matrix for the observations.
        - A: State transition matrix.
        - C: Observation matrix.

        Returns:
        - states: Latent states of shape (S, n).
        - data: Observed data of shape (S, p).
        """
        p, n = C.shape[-2:]
        states = np.zeros((S, n))
        data = np.zeros((S, p))

        # Cholesky decompositions
        B = np.linalg.cholesky(sigma_states)
        D = np.linalg.cholesky(sigma_obs)

        # Initial state
        states[0] = mu_init + np.dot(np.linalg.cholesky(sigma_init), np.random.normal(size=n))

        # Initial observation
        data[0] = np.dot(C, states[0]) + np.dot(D, np.random.normal(size=p))

        # Generate states and observations for each time step
        for t in range(1, S):
            states[t] = np.dot(A, states[t - 1]) + np.dot(B, np.random.normal(size=n))
            data[t] = np.dot(C, states[t]) + np.dot(D, np.random.normal(size=p))

        return states, data

def get_lds(N, n, p, S, theta, dat_dir, sin=True, gen=True, plot=False):
    dataset = LDS(N, n, p, S, dat_dir, theta=theta, sin=sin, gen=gen, plot=plot)
    test_dataset = LDS(N, n, p, S, dat_dir + "_test", theta=theta, sin=sin, gen=gen)
    return dataset, test_dataset



if __name__ == "__main__":
    N = 500
    dat_dir = "../data"
    n = 2
    p = 8
    # PINWHEEL(num_per_class, dat_dir, gen=True, plot=True)
    S = 15
    theta = 1.1

    batch_size = 20
    dataloader = DataLoader(
        LDS(N, n, p, S, dat_dir, sin=True, seed=123, theta=theta,
            gen=True, plot=True), batch_size=batch_size, shuffle=True)

    # for i, (Xbatch, ybatch) in enumerate(dataloader):
    #     print(Xbatch[0])
    #     print(ybatch[0])


    # # Initialize variables
    # timesteps = 80  # Number of steps to generate the sine wave
    # z_dim = 2  # Dimensionality of the state vector
    # # Define the mean and covariance for the initial state z_0
    # mu = torch.tensor([0.5, -0.5])  # Mean of the initial state
    # C = torch.tensor([[0.1, 0.05], [0.05, 0.1]])  # Covariance matrix for the initial state

    # # Sample the initial state z_0 ~ N(mu, C)
    # z_0_dist = torch.distributions.MultivariateNormal(mu, C)
    # z = z_0_dist.sample().unsqueeze(-1)  # (2, 1) vector for the initial state

    # # Define the rotation matrix A (2x2 matrix)
    # theta = torch.Tensor([0.02])  # Angle of rotation per timestep
    # A = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
    #                   [torch.sin(theta), torch.cos(theta)]])

    # # Define the covariance matrix D (for noise)
    # D = torch.diag(torch.tensor([0.001, 0.001]))

    # # Storage for the results
    # z_values = []

    # # Simulate the process z_{t+1} = A z_t + e_t
    # for t in range(timesteps):
    #     # Sample noise from N(0, D)
    #     e = torch.distributions.MultivariateNormal(torch.zeros(z_dim), D).sample().unsqueeze(-1)
        
    #     # Update the state z_{t+1} = A z_t + e_t
    #     z = A @ z + e
        
    #     # Store the first component (which will resemble a sine wave)
    #     z_values.append(z[0].item())  # The first component of z corresponds to sin(x)

    # # Convert the stored values to a numpy array for plotting
    # z_values = np.array(z_values)

    # # Plot the generated sine wave
    # plt.plot(z_values)
    # plt.title("Generated Sinusoidal Curve")
    # plt.xlabel("Time")
    # plt.ylabel("z[0] (Sinusoidal Component)")
    # plt.show()

