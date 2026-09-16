"""Unified LLK (log-likelihood / vector field) model.

Consolidates 7 LLK classes from sfa.py, sfa_discrete.py, and sfa_lds.py
into a single class with a `model=` parameter to select the architecture.

Architecture mapping:
    "mlp"      <- sfa.LLK, sfa_discrete.LLK_R, sfa_lds.LLKc
    "mlp_high" <- sfa.LLK_high, sfa_lds.LLK (non-const)
    "cnn"      <- sfa_discrete.cnnLLK (default mode)
    "cnn_film" <- sfa_lds.LLKcnn
    "resnet"   <- sfa_discrete.cnnLLK (resnet mode)
    "unet"     <- sfa_discrete.cnnLLK (unet mode)
"""

import itertools
from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint

from models.nn import MLP, sum_except_batch, log_normal
from models.unet import ResBlock_nzt, Unet_nzt


class LLK(nn.Module):
    """Unified log-likelihood / vector field model.

    Args:
        x_features: Spatial size (H=W) for images, or flat feature dim for vectors.
        z_features: Latent dimension.
        freqs: Number of Fourier frequencies for time embedding.
        in_ch: Number of input channels (for image data).
        mod_ch: Base channel count for ResNet/UNet architectures.
        hidden_dim: Intermediate MLP dimension. Auto-computed if None.
        num_blocks: Number of ResNet blocks (for "resnet" model).
        model: Architecture type. One of "mlp", "mlp_high", "cnn", "cnn_film",
               "resnet", "unet".
        droprate: Dropout rate (for "resnet" model).
        is_image: Whether input x is image-shaped (batch, C, H, W).
                  Auto-set True for CNN models. For MLP models defaults to False.
        S: Sequence length (for LDS). Stored for decode_sequence/log_prob_sequence.
        F: Total frames in sequence (for LDS). Used in positional embedding.
        dsemb: Sequence embedding dimension (for non-const LDS).
        **kwargs: Passed to MLP layers (hidden_features, fct, batch_norm, etc.)
    """

    def __init__(self, x_features: int, z_features: int, freqs: int = 2, in_ch: int = 1,
                 mod_ch: int = 128, hidden_dim=None, num_blocks: int = 4,
                 model: str = "cnn", droprate: float = 0.2,
                 is_image=None,
                 S=None, F=None, dsemb: int = 2,
                 **kwargs):
        super().__init__()

        self.model_type = model
        self.x_features = x_features
        self.in_ch = in_ch
        self.dtemb = 2 * freqs

        # Sequence support (for LDS)
        self.S = S
        self.F_total = F
        self.dsemb = dsemb
        # Sequence positional embedding only for non-const LDS (mlp_high with S)
        self.has_seq_emb = (S is not None and model == "mlp_high")

        if self.has_seq_emb:
            freqs_s = torch.exp(torch.linspace(
                0, torch.log(torch.tensor(float(S))), dsemb // 2))
            self.register_buffer('freqs_s', 2 * torch.pi / F * freqs_s)

        self.register_buffer('freqs_t', torch.arange(1, freqs + 1) * torch.pi)

        # Image vs vector — auto-detect for CNN models, default False for MLP
        if is_image is None:
            is_image = model not in ("mlp", "mlp_high")
        self.is_image = is_image

        # Flat x dimension (used by MLP models)
        if is_image:
            self.x_flat_dim = in_ch * x_features ** 2
        else:
            self.x_flat_dim = x_features

        # Default hidden_dim per model type
        if hidden_dim is None:
            if model == "mlp":
                hidden_dim = x_features**2 # z_features
            elif model == "mlp_high":
                hidden_dim = 800
            elif model == "cnn_film":
                hidden_dim = 16
            else:
                hidden_dim = x_features ** 2
        self.hidden_dim = hidden_dim

        # Build architecture
        builders = {
            "mlp": self._build_mlp,
            "mlp_film": self._build_mlp_film,
            "mlp_high": self._build_mlp_high,
            "cnn": self._build_cnn,
            "cnn_film": self._build_cnn_film,
            "resnet": self._build_resnet,
            "unet": self._build_unet,
        }
        builder = builders.get(model)
        if builder is None:
            raise ValueError(f"Unknown model type: {model}. "
                             f"Available: {list(builders.keys())}")
        builder(x_features, z_features, in_ch, freqs, mod_ch, num_blocks, droprate, **kwargs)

    # ─── Architecture builders ─────────────────────────────────────────

    def _build_mlp(self, x_features, z_features, in_ch, freqs, mod_ch, num_blocks, droprate, **kwargs):
        self.embz = nn.Sequential(
            # nn.Linear(z_features, z_features),
            # nn.Tanh(),
            # nn.LayerNorm(z_features),
            nn.Linear(z_features, self.hidden_dim),
            # nn.Tanh(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.LayerNorm(self.hidden_dim),
        )
        self.embx = nn.Sequential(
            # nn.LayerNorm(self.x_flat_dim),
            # nn.Linear(self.x_flat_dim, self.hidden_dim),
            # nn.Tanh(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # self.embt = nn.Sequential(
        #     # nn.LayerNorm(self.x_flat_dim),
        #     nn.Linear(self.dtemb, self.hidden_dim)
        # )
        seq_dim = self.dsemb if self.has_seq_emb else 0
        self.fc1 = MLP(
            # self.hidden_dim + self.x_flat_dim + self.dtemb + seq_dim,
            # z_features + self.x_flat_dim + self.dtemb + seq_dim,
            self.hidden_dim*2+ self.dtemb + seq_dim,
            # self.hidden_dim*3 + seq_dim,
            self.x_flat_dim, **kwargs)
    
    def _build_mlp_film(self, x_features, z_features, in_ch, freqs, mod_ch, num_blocks, droprate, **kwargs):
        self.embz_scale = nn.Sequential(
            nn.Linear(z_features, self.hidden_dim),
            # nn.Tanh()
        )
        self.embz_shift = nn.Sequential(
            nn.Linear(z_features, self.hidden_dim),
            # nn.Tanh()
        )

        self.fc_pre = nn.Linear(self.hidden_dim + self.dtemb, self.hidden_dim)
        
        self.fc_post = nn.Sequential(
            MLP(
            self.hidden_dim,
            # self.hidden_dim*2+ self.dtemb + seq_dim,
            self.x_flat_dim, **kwargs),
            # nn.LayerNorm(self.x_flat_dim)
        ) 
        

    def _build_mlp_high(self, x_features, z_features, in_ch, freqs, mod_ch, num_blocks, droprate, **kwargs):
        self.embx = MLP(self.x_flat_dim, self.hidden_dim,
                        hidden_features=[self.hidden_dim], fct=nn.Softplus())
        self.embz = nn.Sequential(
            nn.Linear(z_features, self.hidden_dim),
            nn.Tanh()
            # nn.LayerNorm(self.hidden_dim),
        )
        seq_dim = self.dsemb if self.has_seq_emb else 0
        self.fc1 = MLP(
            2 * self.hidden_dim + self.dtemb + seq_dim,
            self.x_flat_dim, **kwargs)

    def _build_cnn(self, x_features, z_features, in_ch, freqs, mod_ch, num_blocks, droprate, **kwargs):
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_fc = nn.Linear(64 * (x_features // 4) ** 2, self.hidden_dim)
        self.embz = nn.Sequential(
            nn.Linear(z_features, self.hidden_dim),
            nn.Tanh()
            # nn.LayerNorm(self.hidden_dim),
        )
        self.combine_fc = MLP(
            self.hidden_dim * 2 + self.dtemb,
            x_features ** 2 * in_ch, **kwargs)

    def _build_cnn_film(self, x_features, z_features, in_ch, freqs, mod_ch, num_blocks, droprate, **kwargs):
        # FiLM conditioning: z modulates CNN features via scale and shift
        self.embz_scale = nn.Linear(z_features, self.hidden_dim)
        self.embz_shift = nn.Linear(z_features, self.hidden_dim)
        # Zero-init so z starts as identity modulation
        nn.init.zeros_(self.embz_scale.weight)
        nn.init.zeros_(self.embz_scale.bias)
        nn.init.zeros_(self.embz_shift.weight)
        nn.init.zeros_(self.embz_shift.bias)

        cnn_out_dim = 32 * (x_features // 4) ** 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(cnn_out_dim, self.hidden_dim),
        )
        self.fc_pre = nn.Linear(self.hidden_dim + self.dtemb, self.hidden_dim)
        self.fc_post = nn.Linear(self.hidden_dim, x_features ** 2 * in_ch)

    def _build_resnet(self, x_features, z_features, in_ch, freqs, mod_ch, num_blocks, droprate, **kwargs):
        self.base_ch = mod_ch
        self.input_proj = nn.Conv2d(in_ch, mod_ch, 3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResBlock_nzt(mod_ch, mod_ch, droprate) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            nn.Conv2d(mod_ch, in_ch, 3, padding=1))
        self.embz = nn.Sequential(
            nn.Linear(z_features, x_features ** 2 * in_ch),
            nn.Tanh()
            # nn.LayerNorm(self.hidden_dim),
        )
        self.temb_fc_res = nn.Linear(2 * freqs, x_features ** 2)
        self.combine_fc = nn.Sequential(
            nn.Conv2d(in_ch * 2 + 1, in_ch, kernel_size=3, padding=1))

    def _build_unet(self, x_features, z_features, in_ch, freqs, mod_ch, num_blocks, droprate, **kwargs):
        self.unet_net = Unet_nzt(in_ch, mod_ch, in_ch)
        self.cnn_fc = nn.Linear(in_ch * x_features ** 2, self.hidden_dim)
        self.embz = nn.Sequential(
            nn.Linear(z_features, self.hidden_dim),
            nn.Tanh()
            # nn.LayerNorm(self.hidden_dim),
        )
        self.combine_fc = MLP(
            self.hidden_dim * 2 + self.dtemb,
            x_features ** 2 * in_ch, **kwargs)

    # ─── Forward ───────────────────────────────────────────────────────

    def forward(self, t: Tensor, x: Tensor, z: Tensor, s=None) -> Tensor:
        # Time embedding
        _t = self.freqs_t * t[..., None]
        temb = torch.cat((_t.cos(), _t.sin()), dim=-1)

        # Sequence embedding (optional, for non-const LDS)
        semb = None
        if s is not None and self.has_seq_emb:
            _s = self.freqs_s * s[..., None]
            semb = torch.cat((_s.cos(), _s.sin()), dim=-1)

        dispatch = {
            "mlp_film": self._forward_mlp_film,
            "mlp": self._forward_mlp,
            "mlp_high": self._forward_mlp_high,
            "cnn": self._forward_cnn,
            "cnn_film": self._forward_cnn_film,
            "resnet": self._forward_resnet,
            "unet": self._forward_unet,
        }
        return dispatch[self.model_type](temb, x, z, semb)

    def _forward_mlp(self, temb, x, z, semb=None):
        is_img = (x.ndim >= 4)
        input_shape = x.shape

        x_flat = x.flatten(start_dim=1) if is_img else x
        temb = temb.expand(*x_flat.shape[:-1], -1)
        # temb = self.embt(temb)
        zemb = self.embz(z)
        xemb = self.embx(x_flat)
        # xemb = x_flat

        parts = [temb, zemb, xemb]
        if semb is not None:
            semb = semb.expand(*x_flat.shape[:-1], -1)
            parts.append(semb)

        out = self.fc1(torch.cat(parts, dim=-1))
        return out.view(input_shape) if is_img else out

    def _forward_mlp_film(self, temb, x, z, semb=None):
        is_img = (x.ndim >= 4)
        input_shape = x.shape
        x_flat = x.flatten(start_dim=1) if is_img else x
        temb = temb.expand(*x_flat.shape[:-1], -1)

        h = self.fc_pre(torch.cat([x_flat, temb], dim=-1))
        h = h * (1 + self.embz_scale(z)) + self.embz_shift(z)
        out = self.fc_post(h)
        return out.view(input_shape) if is_img else out
        
    def _forward_mlp_high(self, temb, x, z, semb=None):
        is_img = (x.ndim >= 4)
        input_shape = x.shape

        x_flat = x.flatten(start_dim=1) if is_img else x
        temb = temb.expand(*x_flat.shape[:-1], -1)
        xemb = self.embx(x_flat)
        zemb = self.embz(z)

        parts = [temb, xemb, zemb]
        if semb is not None:
            semb = semb.expand(*x_flat.shape[:-1], -1)
            parts.append(semb)

        out = self.fc1(torch.cat(parts, dim=-1))
        return out.view(input_shape) if is_img else out

    def _forward_cnn(self, temb, x, z, semb=None):
        B, C, H, W = x.shape
        temb = temb.expand(B, -1)

        xemb = self.cnn(x)
        xemb = self.cnn_fc(xemb.view(B, -1))
        zemb = self.embz(z)

        out = self.combine_fc(torch.cat([xemb, zemb, temb], dim=-1))
        return out.view(B, C, H, W)

    def _forward_cnn_film(self, temb, x, z, semb=None):
        n, p = x.shape[0], x.shape[-1]
        temb = temb.expand(n, -1)

        xemb = self.cnn(x)
        h = self.fc_pre(torch.cat([xemb, temb], dim=-1))
        h = h * (1 + self.embz_scale(z)) + self.embz_shift(z)
        out = self.fc_post(h)
        return out.view(n, self.in_ch, p, p)

    def _forward_resnet(self, temb, x, z, semb=None):
        xemb = self.input_proj(x)
        xemb = self.res_blocks(xemb)
        xemb = self.output_proj(xemb)
        zemb = self.embz(z)[:, :, None, None]
        temb = temb[:, :, None, None]
        return self.combine_fc(torch.cat([xemb, zemb, temb], dim=1))

    def _forward_unet(self, temb, x, z, semb=None):
        B, C, H, W = x.shape
        temb = temb.expand(B, -1)

        xemb = self.unet_net(x)
        xemb = self.cnn_fc(xemb.view(B, -1))
        zemb = self.embz(z)

        out = self.combine_fc(torch.cat([xemb, zemb, temb], dim=-1))
        return out.view(B, C, H, W)

    # ─── torchdiffeq interface ─────────────────────────────────────────

    def _forward(self, t: Tensor, x: Tensor, z: Tensor, s=None) -> Tensor:
        out = self.forward(t, x, z, s)
        return out, out

    # ─── ODE methods ───────────────────────────────────────────────────

    def encode(self, x: Tensor) -> Tensor:
        return odeint_adjoint(
            self, x, torch.tensor([0.0, 1.0]),
            adjoint_params=tuple(self.parameters()),
            method="dopri5", atol=1e-5, rtol=1e-5)[-1]

    def decode(self, x: Tensor, z: Tensor, t=None, s=None) -> Tensor:
        if t is None:
            t = 1.
        z = z.clone().detach().requires_grad_(True)
        f = partial(self, z=z, s=s)
        t_span = torch.tensor([float(t), 0.0], device=x.device)
        try:
            xt = odeint_adjoint(
                f, x, t_span,
                adjoint_params=self.parameters(),
                method="dopri5",
                atol=1e-5, rtol=1e-5)[-1]
        except AssertionError:
            # dopri5 dt underflow -- fall back to fixed-step solver
            xt = odeint_adjoint(
                f, x, t_span,
                adjoint_params=self.parameters(),
                method="midpoint",
                options={"step_size": 0.01})[-1]
        return xt

    def decode_with_trajectory(self, x: Tensor, z: Tensor, t=None, num_points=100):
        if t is None:
            t = 1.
        z = z.clone().detach().requires_grad_(True)
        time_points = torch.linspace(0., float(t), num_points, device=x.device)

        trajectory = []
        recorded_times = []

        def ode_func_with_recording(t_val, state):
            trajectory.append(state.detach().clone())
            recorded_times.append(t_val.item())
            return self(t_val, state, z=z)

        xt = odeint_adjoint(
            ode_func_with_recording, x, time_points,
            adjoint_params=self.parameters(),
            atol=1e-5, rtol=1e-5)

        return xt[-1], torch.stack(trajectory), torch.tensor(recorded_times)

    def decode_sequence(self, xS: Tensor, zS: Tensor, t=None, indices=None):
        """Decode a sequence of observations.

        If has_seq_emb (non-const LDS): loops over S with positional embeddings.
        Otherwise: batches all S positions together for efficiency.
        """
        if t is None:
            t = 1.
        S = xS.shape[0]

        if self.has_seq_emb:
            if indices is None:
                indices = torch.arange(S, device=xS.device) / S
            x0S = xS.clone().detach()
            for s_idx in range(S):
                x0S[s_idx] = self.decode(xS[s_idx], zS[s_idx], t=t, s=indices[s_idx])
            return x0S
        else:
            B = xS.shape[1]
            x_flat = xS.reshape(S * B, *xS.shape[2:])
            z_flat = zS.reshape(S * B, *zS.shape[2:])
            x0_flat = self.decode(x_flat, z_flat, t=t)
            return x0_flat.reshape(S, B, *xS.shape[2:])

    # ─── Log probability (Hutchinson trace, universal) ─────────────────

    @staticmethod
    def hutch_trace(f, y, e):
        """Hutchinson's estimator for the Jacobian trace."""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        return sum_except_batch(e_dzdx_e)

    def log_prob(self, x: Tensor, z: Tensor, t=0, source=None, s=None) -> Tensor:
        """Compute log p(x|z) using Hutchinson trace estimator.

        Args:
            x: Input tensor (image or vector).
            z: Latent tensor.
            t: Start time for ODE integration (default 0).
            source: Prior distribution for x. If None, uses standard normal.
            s: Sequence position (for non-const LDS models).
        """
        z = z.clone().detach().requires_grad_(True)
        e = torch.randint(low=0, high=2, size=x.size(), device=x.device).to(x.dtype) * 2 - 1

        def augmented(t_val: Tensor, state):
            x_aug, adj = state
            with torch.enable_grad():
                x_aug = x_aug.requires_grad_(True)
                dx = self(t_val, x_aug, z, s=s)
                trace = self.hutch_trace(dx, x_aug, e)
            return dx, trace * 1e-3

        ladj = x.new_zeros(x.shape[0])
        x0, ladj = odeint_adjoint(
            augmented, (x, ladj),
            torch.tensor([float(t), 1.0], device=x.device),
            adjoint_params=self.parameters(),
            atol=1e-5, rtol=1e-5)

        if source is not None:
            lp = source.log_prob(x0[-1])
            if lp.ndim > 1:
                lp = lp.sum(dim=tuple(range(1, lp.ndim)))
            return lp + ladj[-1] * 1e3
        else:
            # Flatten to (batch, -1) so log_normal sums over all non-batch dims
            x_final = x0[-1].reshape(x0[-1].shape[0], -1)
            return log_normal(x_final) + ladj[-1] * 1e3

    def log_prob_sequence(self, xS: Tensor, zS: Tensor, source=None):
        """Compute log probability over a sequence."""
        S = xS.shape[0]

        if self.has_seq_emb:
            indices = torch.arange(S, device=xS.device) / S
            out = 0
            for s_idx in range(S):
                out += self.log_prob(xS[s_idx], zS[s_idx], source=source, s=indices[s_idx])
            return out
        else:
            B = xS.shape[1]
            x_flat = xS.reshape(S * B, *xS.shape[2:])
            z_flat = zS.reshape(S * B, *zS.shape[2:])
            lp = self.log_prob(x_flat, z_flat, source=source)
            return lp.reshape(S, B).sum(dim=0)
