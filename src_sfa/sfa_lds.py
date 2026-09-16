import math
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from typing import *

import torch
from torch import Tensor, vmap
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.nn.utils as nn_utils
from torch.distributions import MultivariateNormal, Normal, Independent

from models.unet import *
from models.nn import *
from utils import *

from dataloader.dataloader_lds import *

torch.set_printoptions(precision=3)


class LatentDynamicalSystem(nn.Module):
    def __init__(self, latent_dim):
        """
        Initialize the latent dynamical system prior.
        
        Parameters:
        - latent_dim: Dimensionality of the latent space.
        """
        super(LatentDynamicalSystem, self).__init__()
        self.latent_dim = latent_dim
        self.mu1 = torch.zeros(latent_dim)  # Mean vector for z1
        self.log_Q1 = torch.log(torch.ones(latent_dim)*0.005) # torch.zeros(latent_dim)  # Log-diagonal of covariance matrix Q1
        delta = 0.45  # rotation angle per step (~2.5 cycles in 36 steps)
        A = torch.eye(latent_dim)
        # Apply rotation in the first 2 dimensions
        A[0, 0] = math.cos(delta)
        A[0, 1] = -math.sin(delta)
        A[1, 0] = math.sin(delta)
        A[1, 1] = math.cos(delta)
        self.A = A  # Rotation transition matrix A
        self.log_Q = torch.log(torch.ones(latent_dim)*0.02)  # Log-diagonal of covariance matrix Q for subsequent steps


    def _get_distribution(self, mu, log_var):
        """Create a diagonal Gaussian distribution given mean and log variance."""
        # restrict mu to (-1,1), restrict log_var to negative

        scale = torch.diag_embed(torch.exp(0.5 * log_var))
        dists = MultivariateNormal(mu, scale_tril=scale)
        return dists

    def rsample(self, n, S, device):
        """
        Generate a sequence of latent variables z_1, ..., z_T using reparameterization trick.
        Parameters:
        - T: Number of time steps.
        Returns:
        - zs: Sequence of latent variables (shape: [T, latent_dim])
        """
        zs = torch.zeros((S, n, self.latent_dim), device=device)
        # First latent variable z1 ~ N(mu1, Q1)
        dist_z1 = self._get_distribution(self.mu1.to(device), self.log_Q1.to(device))
        zs[0] = dist_z1.rsample((n,))  # Use rsample for reparameterization
        # Latent variables z_s+1 ~ N(A * z_s, Q)
        for s in range(1, S):
            dist_zs = self._get_distribution(
                torch.einsum("ij, nj -> ni", self.A.to(device), zs[s-1].clone()), self.log_Q.to(device))
            # print(dist_zs.rsample().shape)
            zs[s] = dist_zs.rsample()  # Use rsample for reparameterization
        
        return zs # (S, n, latent_dim)

    def sample(self, n, S, device):
        """
        Generate a sequence of latent variables z_1, ..., z_T using reparameterization trick.
        Parameters:
        - T: Number of time steps.
        Returns:
        - zs: Sequence of latent variables (shape: [T, latent_dim])
        """
        zs = torch.zeros((S, n, self.latent_dim)).to(device)
        # First latent variable z1 ~ N(mu1, Q1)
        dist_z1 = self._get_distribution(self.mu1.to(device), self.log_Q1.to(device))
        zs[0] = dist_z1.sample((n,)).to(device) 
        # Latent variables z_s+1 ~ N(A * z_s, Q)
        for s in range(1, S):
            mus = self.A.to(device).expand(n,self.latent_dim,self.latent_dim) @ zs[s-1].unsqueeze(-1)
            dist_zs = self._get_distribution(
                mus.squeeze(), 
                self.log_Q.to(device))
            zs[s] = dist_zs.sample().to(device)  # Use rsample for reparameterization

        return zs # (S, n, latent_dim)

    def _log_prob(self, z_s, z_s_minus_1):
        """
        Evaluate the log probability of z_s given z_{s-1}.
        
        Parameters:
        - z_s: The current latent variable z_s (shape: [latent_dim])
        - z_s_minus_1: The previous latent variable z_{s-1} (shape: [latent_dim])
        
        Returns:
        - log_prob: Log probability of z_s given z_{s-1}.
        """
        if z_s_minus_1 is not None:
            # Compute the means A * z_{s-1} for all steps in parallel
            means = torch.einsum('ij,ni->nj', self.A, z_s_minus_1)  # Shape: [n, T-1, latent_dim]
            # Get the conditional distribution P(z_s | z_{s-1}) for all steps
            dist_zs = self._get_distribution(means, self.log_Q)
            # Compute log-probability of z_s under P(z_s | z_{s-1})
            log_prob = dist_zs.log_prob(z_s) # [n,]
        else:
            dist_zs = self._get_distribution(self.mu1, self.log_Q1)
            log_prob = dist_zs.log_prob(z_s)

        return log_prob.squeeze()


class sGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, S, F, dsemb:int =8, bidirectional=True, activation="tanh"):
        super(sGRU, self).__init__()
        self.S = S
        self.F = F
        self.dsemb = dsemb
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.positional_encoding = timestep_embedding
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=False)

    def forward(self, x, indices=None):
        batch_size = x.shape[1]
        if self.bidirectional:
            # Initial hidden state
            h0 = torch.zeros(2*self.num_layers, batch_size, self.hidden_size).to(x.device)
            # Forward propagate through RNN
            out, hs = self.gru(x, h0)
            # compute embeded x
            emb = torch.concatenate([hs[0], hs[1]], dim=-1).view(batch_size, self.hidden_size*2)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            # Forward propagate through RNN
            out, hs = self.gru(x, h0)
            emb = hs
        # print("emb", emb.squeeze().shape)
        return out, emb

    def recurse(self, xs, hsminus, s):
        batch_size = xs.shape[0]

        xs = xs.unsqueeze(0).clone()

        # with new sample coming in compute the new embedding
        if not self.bidirectional:
            if hsminus is not None:
                _, hs = self.gru(xs, hsminus)
                # print("hs", hs.shape)
                emb = hs # + hsminus
            else:
                _, hs = self.gru(xs, hsminus)
                emb = hs # .view(-1, self.hidden_size)
            # _, hs = self.gru(xs, hsminus)
        return emb


class fullGauss(nn.Module):
    def __init__(
        self, x_features: int, z_features: int, S: int, F: int,
        dsemb: int = 4, freqs: int = 2, num_hidden=32, nonlinearity="tanh",
        cnn=False, num_layers=2, emb_dim=32, in_ch=1, rnn=False, attention=False,
        **model_kwargs
    ):
        super().__init__()
        self.S = S
        self.F = F
        self.num_hidden = num_hidden
        self.num_hidden_z = num_hidden
        self.num_layers = num_layers
        self.dsemb = dsemb
        self.dtemb = 2 * freqs
        self.x_features = x_features
        self.z_features = z_features

        self.frame_enc = nn.Sequential(
            nn.LayerNorm(x_features),
            nn.Linear(x_features, num_hidden),
        )

        self.rnnz = sGRU(z_features, self.num_hidden_z, num_layers,
                         S=S, F=F, dsemb=dsemb, bidirectional=False)

        self.q_proj = nn.Linear(self.num_hidden_z * num_layers, num_hidden)
        self.k_proj = nn.Linear(num_hidden, num_hidden)
        self.v_proj = nn.Linear(num_hidden, num_hidden)
        self.attn_scale = num_hidden ** -0.5

        #   input: cross-attn context + z-context + time emb + scale emb
        self.num_l_elements = z_features * (z_features + 1) // 2
        mlp_output_dim = z_features + self.num_l_elements
        mlp_input_dim = num_hidden + self.num_hidden_z * num_layers + self.dtemb + self.dsemb

        self.fc = MLP(mlp_input_dim, mlp_output_dim, **model_kwargs)

        with torch.no_grad():
            last_layer = [m for m in self.fc.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

        diag_indices = torch.arange(z_features)
        self.register_buffer('diag_indices', diag_indices)
        off_tril = torch.tril_indices(row=z_features, col=z_features, offset=-1)
        self.register_buffer('off_tril_r', off_tril[0])
        self.register_buffer('off_tril_c', off_tril[1])

        self._hx_all = None

    def _time_embed(self, t, like):
        t_inp = t if t.dim() >= 1 else t.unsqueeze(0)
        temb = timestep_embedding(t_inp, self.dtemb).to(like.device)
        temb = temb.expand(*like.shape[:-1], -1)
        return temb

    def _scale_embed(self, s, like):
        if self.dsemb == 0:
            return like.new_zeros(*like.shape[:-1], 0)
        s_inp = s if s.dim() >= 1 else s.unsqueeze(0)
        semb = timestep_embedding(s_inp, self.dsemb).to(like.device)
        semb = semb.expand(*like.shape[:-1], -1)
        return semb

    def _cross_attend(self, hz, hx_all):
        """
        hz:     (B, num_hidden_z * num_layers) — z-context
        hx_all: (S, B, num_hidden) — per-frame x encodings
        returns: (B, num_hidden) — attended x-context
        """
        q = self.q_proj(hz).unsqueeze(1)            # (B, 1, H)
        k = self.k_proj(hx_all).permute(1, 0, 2)    # (B, S, H)
        v = self.v_proj(hx_all).permute(1, 0, 2)    # (B, S, H)

        attn = (q @ k.transpose(-1, -2)) * self.attn_scale  # (B, 1, S)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).squeeze(1)                           # (B, H)
        return out

    def _build_scale_tril(self, phi, min_variance):
        B = phi.shape[0]
        mu = (phi[:, :self.z_features])

        raw_diag = phi[:, self.z_features:self.z_features * 2]
        raw_off  = phi[:, self.z_features * 2:]
        L = torch.zeros(B, self.z_features, self.z_features,
                        device=phi.device, dtype=phi.dtype)
        L[:, self.off_tril_r, self.off_tril_c] = (raw_off)
        sqrt_min = torch.as_tensor(min_variance, device=phi.device, dtype=phi.dtype).sqrt()
        L[:, self.diag_indices, self.diag_indices] = F.softplus((raw_diag)) + sqrt_min
        return mu, L

    def forward(self, t, zcemb, xcemb, s, min_variance=1e-4):
        """
        t: scalar or (B,)
        zcemb: (B, Hz*num_layers) or None
        xcemb: ignored — we use self._hx_all instead
        s: scalar or (B,)
        """
        B = self._hx_all.shape[1]
        device = self._hx_all.device
        dtype = self._hx_all.dtype

        temb = self._time_embed(t, self._hx_all[0])       # (B, dtemb)
        semb = self._scale_embed(s, self._hx_all[0])       # (B, dsemb)

        if zcemb is None:
            zcemb = torch.zeros(B, self.num_hidden_z * self.num_layers,
                                device=device, dtype=dtype)

        # z-context cross-attends into per-frame x encodings
        cx = self._cross_attend(zcemb, self._hx_all)       # (B, num_hidden)

        inp = torch.cat([temb, semb, zcemb, cx], dim=-1)
        phi = self.fc(inp)
        mu, L = self._build_scale_tril(phi, min_variance)
        return MultivariateNormal(loc=mu, scale_tril=L)
    
    def _encode_x_window(self, xS, indices):
        S, B = xS.shape[0], xS.shape[1]
        hx = self.frame_enc(xS)  # (S, B, num_hidden)

        idx = torch.arange(S, device=xS.device, dtype=xS.dtype) / S  # (S,)
        pos = timestep_embedding(idx, self.num_hidden).to(xS.device)  # (S, num_hidden)
        pos = pos.unsqueeze(1).expand(-1, B, -1)  # (S, B, num_hidden)

        self._hx_all = hx + pos

    def decode(self, zcemb, xcemb, s, t, deterministic=False):
        if t is None:
            t = torch.zeros((), device=xcemb.device, dtype=xcemb.dtype)
        dist = self(t, zcemb, xcemb, s)
        return dist.mean if deterministic else (dist.mean, dist.rsample())

    def decode_sequence(self, zS, xS, t=None, indices=None, deterministic=False):
        S, B = xS.shape[0], xS.shape[1]
        if t is None:
            t = torch.zeros((), device=xS.device, dtype=xS.dtype)
        if indices is None:
            indices = torch.arange(S, device=xS.device, dtype=xS.dtype) / max(S, 1)

        z0S = zS.clone().detach()
        mu0S = zS.clone().detach()
        xcemb = self._encode_x_window(xS, indices)  # sets self._hx_all

        zSemb = [torch.zeros((self.num_layers, B, self.num_hidden_z), device=xS.device)]

        out = self.decode(zSemb[-1].view(-1, self.num_hidden_z * self.num_layers),
                          xcemb, indices[0], t, deterministic)
        if not deterministic:
            z0S[0], mu0S[0] = out[1], out[0]
        else:
            z0S[0] = out

        for s in range(1, S):
            z_input = mu0S[s-1] if not deterministic else z0S[s-1]
            zcemb = self.rnnz.recurse(z_input, zSemb[-1], indices[s])
            zSemb.append(zcemb)

            out = self.decode(zSemb[-1].view(-1, self.num_hidden_z * self.num_layers),
                              xcemb, indices[s], t, deterministic)
            if not deterministic:
                z0S[s], mu0S[s] = out[1], out[0]
            else:
                z0S[s] = out

        return z0S

    def log_prob(self, z, zcemb, xcemb, s, t, prior=None):
        if t is None:
            t = torch.zeros((), device=xcemb.device, dtype=xcemb.dtype)
        return self(t, zcemb, xcemb, s).log_prob(z)

    def log_prob_sequence(self, zS, xS, t, prior=None, indices=None):
        if indices is None:
            indices = torch.arange(self.S, device=xS.device, dtype=xS.dtype) / max(self.S, 1)
        S, B = xS.shape[0], xS.shape[1]
        xcemb = self._encode_x_window(xS, indices)  # sets self._hx_all

        zSemb = [None]
        out = self.log_prob(zS[0], zSemb[-1], xcemb, indices[0], t, prior)
        for s in range(1, S):
            zcemb = self.rnnz.recurse(zS[s-1], zSemb[-1], indices[s])
            zSemb.append(zcemb)
            out = out + self.log_prob(zS[s], zSemb[-1].view(-1, self.num_hidden_z * self.num_layers),
                                      xcemb, indices[s], t, prior)
        return out


class FlowMatchingLossSeq(nn.Module):
    """Unified sequence flow matching loss.

    Merges FlowMatchingLoss (non-const), FlowMatchingLossc (const),
    and FlowMatchingLosscnn (const CNN) into a single class.

    Args:
        const: If True, vt is shared across all time steps (does not receive
               sequence position s). If False, vt is position-aware and called
               with s=indices[s] at each step (non-const LDS).
        flowcnn: If True, rt receives unflattened x. If False, rt gets flattened x.
    """
    def __init__(self, vt: nn.Module, rt: nn.Module, prior,
                 alpha=0.01, sig_min=1e-4, const=False, flowcnn=False, upperbound=False):
        super().__init__()

        self.vt = vt
        self.rt = rt
        self.prior = prior
        self.sig_min = sig_min
        self.alpha = alpha
        self.const = const
        self.flowcnn = flowcnn
        self.upperbound = upperbound

    def forward_entropy(self, zS, xS, t):
        return - self.rt.log_prob_sequence(zS, xS, t, self.prior).mean()

    def forward(self, x: Tensor, indices=None) -> Tensor:
        S, nbatch = x.shape[0], x.shape[1]
        if indices is None:
            indices = torch.arange(S, device=x.device) / S

        _t = torch.rand(1).to(x.device)
        t = torch.ones_like(x[..., 0, None]) * _t

        x1 = torch.randn_like(x).to(device=x.device)
        xt = (1 - t) * x + (self.sig_min + (1 - self.sig_min) * t) * x1
        ut = (1 - self.sig_min) * x1 - x

        # Sample latent z
        z1 = self.prior.rsample(nbatch, S, device=x.device)

        # rt receives flattened or unflattened x based on flowcnn flag
        _xt_rt = xt if self.flowcnn else xt.flatten(start_dim=2)
        zt = self.rt.decode_sequence(z1, _xt_rt, t=_t[0], indices=indices)

        # Compute flow matching loss (accumulate residuals, then square)
        reg = 0
        reg_sq = 0
        if self.upperbound:
            # Per-step squared norm (from FlowMatchingLossc/FlowMatchingLosscnn)
            loss = 0
            for s in range(S):
                residual = self.vt(_t, xt[s], zt[s]) - ut[s]
                rdims = tuple(range(1, residual.ndim))
                loss += torch.mean(torch.norm(residual, p=2, dim=rdims) ** 2)
                if s > 0:
                    reg += (zt[s] - zt[s - 1]) ** 2
                if s > 1:
                    reg_sq += ((zt[s] - zt[s - 1]) - (zt[s - 1] - zt[s - 2])) ** 2
            loss = loss / S
        else:
            fm_loss = 0
            for s in range(S):
                if self.const:
                    fm_loss += self.vt(_t, xt[s], zt[s]) - ut[s]
                else:
                    fm_loss += self.vt(_t, xt[s], zt[s], s=indices[s]) - ut[s]
                if s > 0:
                    reg += (zt[s] - zt[s - 1]) ** 2
                if s > 1:
                    reg_sq += ((zt[s] - zt[s - 1]) - (zt[s - 1] - zt[s - 2])) ** 2
            loss = fm_loss.square().mean(-1).mean() / S


        # Smoothness regularization only
        reg_loss2 = reg_sq.mean() / S if S > 2 else torch.zeros(1, device=x.device)
        beta2 = loss.detach() / (torch.abs(reg_loss2.detach()) + 1e-8)
        return loss + reg_loss2 * self.alpha * beta2


