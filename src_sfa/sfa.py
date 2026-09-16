from functools import partial
from typing import *

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torchdiffeq import odeint, odeint_adjoint

from dataloader.dataloader_pinwheel import *
from utils import *
from models.nn import *

torch.set_printoptions(precision=3)

class GaussianPrior(nn.Module):
    def __init__(self, x_features: int, z_features: int, freqs: int=2, **kwargs):
        super().__init__()
        self.embx = nn.Sequential(
            nn.LayerNorm(x_features),
            # nn.Tanh()
        )
        self.hyper =  nn.Sequential(
            MLP(x_features+2*freqs, z_features*2, **kwargs),
            # nn.LayerNorm(z_features*2)
            )
        self.z_features = z_features
        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)
        
    def forward(self, x:Tensor, t:Tensor=torch.Tensor([0]), min_variance=1e-5):
        x = x.flatten(start_dim=1) if x.ndim >= 4 else x
        t = self.freqs * t[..., None]
        xemb = self.embx(x)
        temb = torch.cat((t.cos(), t.sin()), dim=-1)
        temb = temb.expand(*x.shape[:-1], -1)
        phi = self.hyper(torch.cat((temb, xemb), dim=-1))
        
        mu, log_sigma = phi.chunk(2, dim=-1)
        return Normal((mu), F.softplus(log_sigma)+min_variance)

    def rsample(self, x:Tensor,t:Tensor=torch.Tensor([0])):
        dist = self(x, t)
        return dist.rsample()

    def sample(self, x:Tensor, t:Tensor=torch.Tensor([0])):
        dist = self(x, t)
        return dist.sample()

    def log_prob(self, x:Tensor, z:Tensor, t:Tensor=torch.Tensor([0])):
        dist = self(x, t)
        return dist.log_prob(z).sum(-1)





class CNF(nn.Module):
    def __init__(self, x_features: int, z_features: int, freqs: int = 2, hidden_dim=784, **kwargs):
        super().__init__()

        self.embx = nn.Linear(x_features, hidden_dim)
        self.fc1 = nn.Sequential(
            MLP(2 * freqs + hidden_dim + z_features, z_features, **kwargs),
        )
        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t: Tensor, z: Tensor, x: Tensor) -> Tensor:
        t = self.freqs * t[..., None]
        temb = torch.cat((t.cos(), t.sin()), dim=-1)
        temb = temb.expand(*z.shape[:-1], -1)
        xemb = self.embx(x)
        out = self.fc1(torch.cat((z, temb, xemb), dim=-1))
        return -out

    def encode(self, x: Tensor) -> Tensor:
        t = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
        return odeint(partial(self, x=x), x, t, atol=1e-8, rtol=1e-8)[-1]

    def decode(self, z: Tensor, x: Tensor, t=None) -> Tensor:
        if t is None:
            t = 0.
        t_span = torch.tensor([1.0, float(t)], device=x.device, dtype=z.dtype)
        return odeint_adjoint(
            partial(self, x=x), z, t_span,
            adjoint_params=tuple(self.parameters()),
            atol=1e-8, rtol=1e-8,
        )[-1]

    def log_prob(self, z: Tensor, x: Tensor, t, prior) -> Tensor:
        I = torch.eye(z.shape[-1], dtype=z.dtype, device=z.device)
        I = I.expand(*z.shape, z.shape[-1]).movedim(-1, 0)

        def augmented(t: Tensor, state):
            z, ladj = state
            with torch.enable_grad():
                z = z.requires_grad_(True)
                dz = self(t, z, x)
                jacobian = torch.autograd.grad(
                    dz, z, I, create_graph=True, is_grads_batched=True
                )[0]
            trace = torch.einsum('i...i', jacobian)
            return dz, trace * 1e-2

        ladj = torch.zeros_like(z[..., 0])
        t_span = torch.tensor([float(t), 1.0], device=x.device, dtype=z.dtype)
        zt, ladj = odeint(
            augmented, (z, ladj), t_span,
            atol=1e-8, rtol=1e-8,
        )
        priorlog = prior.log_prob(zt[-1]).sum(-1)
        return priorlog + ladj[-1] * 1e2
    

class FlowMatchingLoss(nn.Module):
    def __init__(self, vt: nn.Module, rt: nn.Module, prior, alpha=0.001, sig_min=1e-4, fixz=False):
        super().__init__()

        self.vt = vt
        self.rt = rt
        self.prior = prior
        self.sig_min = sig_min
        self.alpha = alpha
        self.fixz = fixz

    def forward(self, x: Tensor) -> Tensor:
        if self.fixz:
            _t = torch.rand(len(x), device=x.device)
        else:
            _t = torch.rand(1, device=x.device)
        t = _t.reshape(-1, *([1] * (x.ndim - 1)))
        x0 = torch.randn_like(x)
        xt = (1 - t) * x + (self.sig_min + (1 - self.sig_min) * t) * x0
        ut = (1 - self.sig_min) * x0 - x

        if self.fixz:
            # same time 
            zt = self.rt.forward(xt, _t).rsample()
        else:
            z0 = self.prior.sample((len(x),))
            zt = self.rt.decode(z0, xt, _t)

        fm_loss = (self.vt(_t, xt, zt) - ut).square().mean()
        reg_loss = zt.square().mean()
        return fm_loss + self.alpha * reg_loss


